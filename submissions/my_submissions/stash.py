"""Template submission module.

See https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md#allowed-submissions
and https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md#disallowed-submissions
for guidelines.
"""
from typing import Dict, Iterator, List, Tuple, Any, Callable, NamedTuple, Optional, Union
from algorithmic_efficiency import spec
import chex
from flax import jax_utils
import jax
import jax.numpy as jnp
from jax import lax
import optax
import functools
import flax
import flax.linen as nn
import copy
import jraph


from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils

import numpy as np

# jax.config.update('jax_log_compiles', True)

NamedSharding = jax.sharding.NamedSharding 

mesh_shape = (jax.device_count(),)
mesh = mesh_utils.create_device_mesh(
    mesh_shape, devices=jax.devices()
)
mesh = jax.sharding.Mesh(mesh, axis_names=('data',))
# Utils for jit sharding.



def get_sharding(x, y):
  if len(x.shape) == 1:
    return NamedSharding(mesh, P(None))
  else:
    if x.shape[0] % 8 == 0:
      if x.shape[1] % 8 == 0:
        return NamedSharding(mesh, P('data',))
      else:
        return NamedSharding(mesh, P('data', None))
    else:
      if x.shape[1] % 8 == 0:
        return NamedSharding(mesh, P(None, 'data'))
      else:
        return NamedSharding(mesh, P(None))

# LR Schedule

def jax_cosine_warmup(step_hint: int, hyperparameters):
  # Create learning rate schedule.
  warmup_steps = int(hyperparameters.warmup_steps_fraction * step_hint)
  warmup_fn = optax.linear_schedule(
      init_value=0.,
      end_value=hyperparameters.learning_rate,
      transition_steps=warmup_steps)
  cosine_steps = max(step_hint - warmup_steps, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=hyperparameters.learning_rate, decay_steps=cosine_steps)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[warmup_steps])
  return schedule_fn

# Nadam_EMA implementation

def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)

class ScaleByNadamW_EMAState(NamedTuple):
  """State for the AdaProp algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  params_ema: optax.Updates
  mu: optax.Updates
  nu: optax.Updates

def scale_by_nadam_ema(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    power: float = 2.0,
    decay=0.999,
) -> optax.GradientTransformation:
  """Rescale updates according to the NAdam algorithm.

  References:
  There seem to be multiple versions of NAdam. The original version is here
  https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ (the pytorch imp. also
  follows this)

  Current code implements a simpler version with no momentum decay and slightly
  different bias correction terms. The exact description can be found here
  https://arxiv.org/pdf/1910.05446.pdf (Table 1)

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: whether to use bias correction.
    power: the power to use in the preconditioner (the value determines the
      power to which the absolute value of the grads are raised).

  Returns:
    An (init_fn, update_fn) tuple.
  """

  raise_power = (
      jnp.sqrt if power == 2.0 else lambda x: jnp.power(x, 1.0 / power)
  )
  sharding_map = {}


  def init_fn(params):    
    def zeros_like_params(params):
      return jax.tree_map(jnp.zeros_like, params)

    def copy_params(params):
      return jax.tree_map(jnp.copy, params)

    # Get pytree with param key names mapped to sharding.
    pytree_sharding = nn.get_sharding(params, mesh)

    # Overwrite sharding based on `get_sharding` method.
    pytree_sharding = jax.tree_util.tree_map(
      lambda x, y: get_sharding(x, y),
      params, pytree_sharding)

    jitted_zeros_like = jax.jit(zeros_like_params, out_shardings=pytree_sharding)
    # jitted_copy = jax.jit(copy_params, out_shardings=pytree_sharding)

    params_ema = jitted_zeros_like(params)
    mu = jitted_zeros_like(params)
    nu = jitted_zeros_like(params)

    sharding_map['mu'] = pytree_sharding
    sharding_map['nu'] = pytree_sharding
    sharding_map['params_ema'] = pytree_sharding

    return ScaleByNadamW_EMAState(
        count=jnp.zeros([], jnp.int32),
        params_ema=params_ema,
        mu=mu,
        nu=nu,
    )

  def update_fn(updates, state, params):
    # Update params EMA for evals.
    ema_decay = jnp.minimum(decay, (1. + state.count) / (10. + state.count))

    def update_func(old_v, new_v):
      if old_v.dtype == jnp.bool_ or jnp.issubdtype(old_v, jnp.integer):
        # If it is integer, we directly return the new variable
        # This is mainly supported for non_trainable
        return new_v
      else:
        return old_v - (1.0 - ema_decay) * (old_v - new_v)

    params_ema = jax.tree_map(update_func, state.params_ema, params)
    params_ema = jax.lax.with_sharding_constraint(params_ema, sharding_map["params_ema"])

    # Nadam update rule.
    mu = jax.tree_map(lambda g, t: (1-b1)*g + b1*t, updates, state.mu)
    mu_hat = jax.tree_map(lambda g, t: (1-b1)*g + b1*t, updates, mu)
    mu_hat = jax.lax.with_sharding_constraint(mu_hat, sharding_map["mu"])


    nu = jax.tree_map(lambda g, t: (1-b2)*(g**2) + b2*t, updates, state.nu)
    nu = jax.lax.with_sharding_constraint(nu, sharding_map["nu"])

    count = state.count + jnp.array(1, dtype=jnp.int32)

    mu_hat = _bias_correction(mu_hat, b1, count)
    mu_hat = jax.lax.with_sharding_constraint(mu_hat, sharding_map["mu"])

    nu_hat = _bias_correction(nu, b2, count)
    nu_hat = jax.lax.with_sharding_constraint(nu_hat, sharding_map["nu"])

    # New updates.    
    updates = jax.tree_map(
        lambda m, v: m / (raise_power(v + eps_root) + eps), mu_hat, nu_hat
    )

    return updates, ScaleByNadamW_EMAState(
        count=count,
        params_ema=params_ema,
        mu=mu,
        nu=nu,
    )

  return optax.GradientTransformation(init_fn, update_fn)

def nadamw_ema(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
  """Rescale updates according to the NAdam algorithm.

  References:
  There seem to be multiple versions of NAdam. The original version is here
  https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ (the pytorch imp. also
  follows this)

  Current code implements a simpler version with no momentum decay and slightly
  different bias correction terms. The exact description can be found here
  https://arxiv.org/pdf/1910.05446.pdf (Table 1)

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: whether to use bias correction.
    weight_decay: strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent with
      other frameworks such as PyTorch, but different from (Loshchilov et al,
      2019) where the weight decay is only multiplied with the "schedule
      multiplier", but not the base learning rate.

  Returns:
    An (init_fn, update_fn) tuple.
  """
  return optax.chain(
      scale_by_nadam_ema(b1, b2, eps, eps_root, debias),
      optax.add_decayed_weights(weight_decay),
      scale_by_learning_rate(learning_rate))


def scale_by_learning_rate(learning_rate, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a NAdamW optimizer and a learning rate schedule."""
  del model_params
  del model_state
  del rng

  target_setting_step_hint = int(hyperparameters.cosine_decay_fraction * workload.step_hint)
  lr_schedule_fn = jax_cosine_warmup(target_setting_step_hint, hyperparameters)

  # Create optimizer.
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)

  opt_init_fn, opt_update_fn = nadamw_ema(
      learning_rate=lr_schedule_fn,
      b1=hyperparameters.beta1,
      b2=hyperparameters.beta2,
      weight_decay=hyperparameters.weight_decay)

  optimizer_state = opt_init_fn(params_zeros_like)

  return optimizer_state, opt_update_fn

@functools.partial(jax.jit, donate_argnums=(3, 5), static_argnums=(0, 1, 7))
def train_step(
  workload,
  opt_update_fn,
  model_state,
  optimizer_state,
  current_param_container,
  batch,
  rng,
  label_smoothing):

  def _loss_fn(params):
    """Loss function used for training."""
    logits, new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True)
    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits,
        mask_batch=batch.get('weights'),
        label_smoothing=label_smoothing)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    return summed_loss, (n_valid_examples, new_model_state)

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(
      current_param_container)
  # Get correct global mean loss and grad.
  # (summed_loss, n_valid_examples, grad) = lax.psum(
  #     (summed_loss, n_valid_examples, grad), axis_name='batch')
  loss = summed_loss / n_valid_examples
  grad = jax.tree_map(lambda x: x / n_valid_examples, grad)
  
  grad_norm = jnp.sqrt(
      sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

  current_param_container = jax.tree_util.tree_map(lambda x: jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(None))), current_param_container)

  updates, optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
  current_param_container = jax.tree_util.tree_map(
      lambda p, u: p + u, current_param_container, updates)

  current_param_container = jax.tree_util.tree_map(lambda x: jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(None))), current_param_container)
  
  return optimizer_state, current_param_container, new_model_state, loss, grad_norm

submission_global_step = 0

def convert_host_local_array_to_global_array(arr):
  """Converts a host local array from pmap to global jax.Array.

  Args:
    arr: Input host local array produced by pmap.

  Returns:
    A global array similar to GDA.
  """
  # input `arr` is fully replicated, so it's shape is the global shape.
  global_shape = arr.addressable_data(0).shape
  # Create a 1D mesh to create fully replicated global jax.Array.
  mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=('x',))
  partition_spec = (
      jax.sharding.PartitionSpec(None)
      if global_shape
      else jax.sharding.PartitionSpec()
  )
  # pmap-produced Array has a "scrambled" device order.
  dbs = sorted(
      [shard.data for shard in arr.addressable_shards],
      key=lambda x: list(x.devices())[0].id,
  )
  return jax.make_array_from_single_device_arrays(
      global_shape, jax.sharding.NamedSharding(mesh, partition_spec), dbs
  )

def convert_fully_replicated_array_to_pmap_array(arr):
  """Converts a fully replicated Array to Array with PmapSharding.

  Args:
    arr: Fully replicated jax.Array.

  Returns:
    Fully replicated jax.Array with PmapSharding. This is suitable as an
    input to pmap.
  """
  assert isinstance(arr, jax.Array)
  # with jax.transfer_guard('disallow'):
  local_shape = (jax.local_device_count(),) + arr.shape
  device_buffers = [shard.data for shard in arr.addressable_shards]
  devices = np.array([shard.device for shard in arr.addressable_shards])

  s = jax.sharding.PmapSharding.default(
      local_shape, sharded_dim=0, devices=devices
  )
  return jax.make_array_from_single_device_arrays(local_shape, s,
                                                  device_buffers)

def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del eval_results

  optimizer_state, opt_update_fn = optimizer_state
  per_device_rngs = jax.random.split(rng, jax.local_device_count())
  if hasattr(hyperparameters, 'label_smoothing'):
    label_smoothing = hyperparameters.label_smoothing
  else:
    label_smoothing = 0.0

  # Convert pmap array to jit fully replicated Array.
  current_param_container = jax.tree_util.tree_map(
    lambda x: convert_host_local_array_to_global_array(x), current_param_container)

  (optimizer_state, 
      current_param_container, 
      model_state, 
      loss, 
      grad_norm) = train_step( # pylint: disable=line-too-long
      workload=workload, 
      opt_update_fn=opt_update_fn, 
      label_smoothing=label_smoothing,
          model_state=model_state, 
          optimizer_state=optimizer_state,
          current_param_container=current_param_container, 
          batch=batch, 
          rng=rng)

  global submission_global_step

  if ((submission_global_step <= 100 or submission_global_step % 500 == 0) and
    workload.metrics_logger is not None):
      workload.metrics_logger.append_scalar_metrics(
          {
              'loss': loss,
              'grad_norm': grad_norm,
          }, submission_global_step)

  submission_global_step = submission_global_step + 1
  
  # Convert jit fully replicated array to pmap array.
  current_param_container = jax.tree_util.tree_map(
    lambda x: convert_fully_replicated_array_to_pmap_array(x), current_param_container)

  return (optimizer_state, opt_update_fn), current_param_container, model_state


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262144
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
    return 1024
  elif workload_name == 'imagenet_resnet_silu':
    return 512
  elif workload_name == 'imagenet_resnet_gelu':
    return 512
  elif workload_name == 'imagenet_vit':
    return 1024
  elif workload_name == 'imagenet_vit_glu':
    return 512
  elif workload_name == 'librispeech_conformer':
    return 128
  elif workload_name == 'librispeech_deepspeech':
    return 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 128
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')

def print_device_buffers():
    print('len = ', len(jax.live_arrays('gpu')))
    
    # for device_live_arr in jax.live_arrays('gpu'):
    #    print('shape = ' , device_live_arr.shape)



def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   model_state: spec.ModelAuxiliaryState,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.
    Each element of the queue is a batch of training examples and labels.
    Tip:
    If you would just like the next batch from the input queue return next(input_queue).

    Returns:
      batch: next batch of input data
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del rng

  batch = next(input_queue)

  for key in batch.keys():
    print('key = ', key)

  def shard_arr_helper(arr):
    array_shape = arr.shape

    if len(array_shape) == 1:
      return arr
    else:
      leading_batch_dim = array_shape[0] * array_shape[1]
      
      if len(array_shape) == 2:
        return shard_arr(arr.reshape(leading_batch_dim, ))
      else:
        return shard_arr(arr.reshape(leading_batch_dim, -1))
    



  for key in batch.keys():
    if isinstance(batch[key], jraph.GraphsTuple):
      print('key = ', key)

      print('nodes = ', batch[key].nodes.shape)
      print('edges = ', batch[key].edges.shape)
      print('receivers = ', batch[key].receivers.shape)
      print('senders = ', batch[key].senders.shape)
      print('n_node = ', batch[key].n_node.shape)
      print('n_edge = ', batch[key].n_edge.shape)
      
      batch[key] = batch[key]._replace(nodes = shard_arr_helper(batch[key].nodes))
      batch[key] = batch[key]._replace(edges = shard_arr_helper(batch[key].edges))
      batch[key] = batch[key]._replace(receivers = shard_arr_helper(batch[key].receivers))
      batch[key] = batch[key]._replace(senders = shard_arr_helper(batch[key].senders))
      batch[key] = batch[key]._replace(n_node = shard_arr_helper(batch[key].n_node))
      batch[key] = batch[key]._replace(n_edge = shard_arr_helper(batch[key].n_edge))

      print('nodes = ', batch[key].nodes.shape)
      print('edges = ', batch[key].edges.shape)
      print('receivers = ', batch[key].receivers.shape)
      print('senders = ', batch[key].senders.shape)
      print('n_node = ', batch[key].n_node.shape)
      print('n_edge = ', batch[key].n_edge.shape)
    
    elif isinstance(batch[key], tuple):
      elem1, elem2 = batch[key]
      batch[key] = (shard_arr_helper(elem1), shard_arr_helper(elem2))
    else:
      array_shape = batch[key].shape

      leading_batch_dim = array_shape[0] * array_shape[1]
      batch[key] = shard_arr(batch[key].reshape(leading_batch_dim, -1))

      print('key = ', key, ' - shape = ', batch[key].shape)


  return batch
