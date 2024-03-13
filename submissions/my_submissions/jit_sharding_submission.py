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

from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils

import numpy as np

NamedSharding = jax.sharding.NamedSharding 

mesh_shape = (jax.device_count(),)
mesh = mesh_utils.create_device_mesh(
    mesh_shape, devices=jax.devices()
)
mesh = jax.sharding.Mesh(mesh, axis_names=('data',))
p = NamedSharding(mesh, P('data'))

print(mesh)

def jax_cosine_warmup(step_hint: int, hyperparameters):
  # Create learning rate schedule.
  warmup_fn = optax.linear_schedule(
      init_value=0.,
      end_value=hyperparameters.learning_rate,
      transition_steps=hyperparameters.warmup_steps)
  cosine_steps = max(step_hint - hyperparameters.warmup_steps, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=hyperparameters.learning_rate, decay_steps=cosine_steps)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[hyperparameters.warmup_steps])
  return schedule_fn

def get_shard_array_fn(sharding):
  shard_array_fn = jax.jit(lambda x:x, out_shardings =sharding)
  return shard_array_fn

def shard_pytree(pytree):
  pytree_sharding = nn.get_sharding(pytree, mesh)
  pytree_sharding = jax.tree_util.tree_map(lambda x: p, pytree_sharding)

  shard_array_fn = get_shard_array_fn(pytree_sharding)
  return shard_array_fn(pytree), pytree_sharding

def replicate_pytree(pytree):
  pytree_sharding = nn.get_sharding(pytree, mesh)
  pytree_sharding = jax.tree_util.tree_map(lambda x: NamedSharding(mesh, P(None)), pytree_sharding)

  shard_array_fn = get_shard_array_fn(pytree_sharding)
  return shard_array_fn(pytree)

def shard_array(arr):
  shard_array_fn = get_shard_array_fn(p)
  return shard_array_fn(arr)


class ScaleByAdapropState(NamedTuple):
  """State for the AdaProp algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  pp: optax.Updates
  mu: optax.Updates
  nu: optax.Updates
  gain: optax.Updates

def scale_by_adaprop(
    alpha: float = 1.0,
    b1: float = 0.9,
    b3: float = 1.0,
    b4: float = 0.9,
    eps: float = 1e-8,
    use_nesterov: str = 'True',
    dtype: jnp.dtype = jnp.float32,
) -> optax.GradientTransformation:
  """Rescale updates according to the AdaProp algorithm.

  Args:
    alpha: upper bound on bet.
    b1: decay rate for the exponentially weighted average of grads.
    # b2: decay rate for the exponentially weighted average of absolute grads
    #     is omitted because it is calculated from alpha and b1.
    b3: decay rate for the exponentially weighted average of max grads.
    b4: decay rate for the exponentially weighted average of reward.
    eps: term added to the denominator to improve numerical stability.
    use_nesterov: Whether to use Nesterov-style update.
    dtype: type of the  input. Allowed options are
      jnp.float16 and jnp.float32. If floating-point type is specified,
      accumulators are stored as such type, instead of quantized integers.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  sharding_map = {}

  def init_fn(params):
    prev_params = jax.tree_map(
        lambda p: jnp.zeros_like(p, dtype=dtype), params
    )
    mu = jax.tree_map(
        lambda p: jnp.zeros_like(p, dtype=dtype), params
    )
    nu = jax.tree_map(
        lambda p: jnp.zeros_like(p, dtype=dtype), params
    )
    gain = jax.tree_map(
        lambda p: jnp.zeros_like(p, dtype=dtype), params
    )

    prev_params, prev_params_sharding = shard_pytree(prev_params)
    mu, mu_sharding = shard_pytree(mu)
    nu, nu_sharding = shard_pytree(nu)
    gain, gain_sharding = shard_pytree(gain)

    sharding_map['mu'] = mu_sharding
    sharding_map['nu'] = nu_sharding
    sharding_map['gain'] = gain_sharding
    sharding_map['pp'] = prev_params_sharding
    
    # print(sharding_map['pp'])


    return ScaleByAdapropState(
        count=jnp.zeros([], jnp.int32),
        pp=prev_params,
        mu=mu,
        nu=nu,
        gain=gain,
    )

  def update_fn(updates, state, params):
    new_count = optax.safe_int32_increment(state.count)
    b2 = 1.0 - (1.0 - b1)/alpha

    mu = jax.tree_map(lambda g, t: (1-b1)*g + b1*t, updates, state.mu)
    mu = jax.lax.with_sharding_constraint(mu, sharding_map["mu"])

    if use_nesterov == 'True':
      mu2 = jax.tree_map(lambda g, t: (1-b1)*g + b1*t, updates, mu)
      mu_hat = _bias_correction(mu2, b1, new_count)
    else:
      mu_hat = _bias_correction(mu, b1, new_count)

    nu = jax.tree_map(lambda g, t: (1-b2)*jnp.abs(g) + b2*t, updates, state.nu)
    nu = jax.lax.with_sharding_constraint(nu, sharding_map["nu"])

    nu_hat = _bias_correction(nu, b2, new_count)
    
    pp = jax.tree_map(lambda p, t: (1-b4)*p + b4*t, params, state.pp)
    pp = jax.lax.with_sharding_constraint(pp, sharding_map["pp"])

    pp_hat = _bias_correction(pp, b4, new_count)
    param_change = jax.tree_map(lambda p, i: p - i, params, pp_hat)
    g_max = jax.tree_map(lambda g, n: jnp.maximum(jnp.abs(g), n),
                         updates, nu_hat)
    gain = jax.tree_map(
        lambda r, p, g, x: jnp.maximum(b3*r - p*g/(x + eps), 0.0),
        state.gain, param_change, updates, g_max)
    gain = jax.lax.with_sharding_constraint(gain, sharding_map["gain"])
    wealth = jax.tree_map(lambda g: 1.0 + g, gain)

    bet_factor = jax.tree_map(
        lambda m, n: m / (n + eps),
        mu_hat,
        nu_hat,
    )
    new_updates = jax.tree_map(lambda b, w: b * w,
                               bet_factor, wealth)
    return new_updates, ScaleByAdapropState(
        count=new_count,
        pp=pp,
        mu=mu,
        nu=nu,
        gain=gain,
    )

  return optax.GradientTransformation(init_fn, update_fn)

def nadamw(
    learning_rate: Union[float, optax.Schedule],
    b1: float = 0.9769389763078179,
    b2: float = 0.9580544644542972,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[optax.Params],
                                                    Any]]] = None,
) -> optax.GradientTransformation:
  """Rescale updates according to the NAdam algorithm.
  References:
  There seem to be multiple versions of NAdam. The original version is here
  https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ (the official PyTorch
  implementation also follows this).
  Current code implements a simpler version with no momentum decay and slightly
  different bias correction terms. The exact description can be found here
  https://arxiv.org/pdf/1910.05446.pdf (Table 1).
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
    weight_decay_mask: a tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Nadam gradient transformations are applied to all parameters.
  Returns:
    An (init_fn, update_fn) tuple.
  """
  return optax.chain(
      scale_by_adaprop(b1=0.9769389763078179, b3=1.0, eps=eps),
      optax.add_decayed_weights(weight_decay, weight_decay_mask),
      scale_by_learning_rate(learning_rate))


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)


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

  target_setting_step_hint = int(0.75 * workload.step_hint)
  lr_schedule_fn = jax_cosine_warmup(target_setting_step_hint,
                                                   hyperparameters)

  # Create optimizer.
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  epsilon = (
      hyperparameters.epsilon if hasattr(hyperparameters, 'epsilon') else 1e-8)
  opt_init_fn, opt_update_fn = nadamw(
      learning_rate=lr_schedule_fn,
      b1=hyperparameters.beta1,
      b2=hyperparameters.beta2,
      eps=epsilon,
      weight_decay=hyperparameters.weight_decay)
  optimizer_state = opt_init_fn(params_zeros_like)

  return optimizer_state, opt_update_fn


@functools.partial(jax.jit, static_argnums=(0, 1, 7))
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

  updates, optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
  current_param_container = jax.tree_util.tree_map(
      lambda p, u: p + u, current_param_container, updates)
  
  return optimizer_state, current_param_container, new_model_state, loss, grad_norm


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

  # partial_train_step = functools.partial(
  #   train_step, 
  #   )


  current_param_container = jax_utils.unreplicate(current_param_container)
  current_param_container = replicate_pytree(current_param_container)
    
  if global_step == 0:
    print('optimizer state pp sharding = ')
    jax.debug.visualize_array_sharding(optimizer_state[0].pp['shared_embedding']['embedding'])

    print('params sharding = ')

    print(current_param_container['shared_embedding']['embedding'].shape)
    jax.debug.visualize_array_sharding(current_param_container['shared_embedding']['embedding'])

    print('batch input sharding = ')
    jax.debug.visualize_array_sharding(batch['inputs'])


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

  current_param_container = jax_utils.replicate(current_param_container)

  # print('loss = ', loss)
  # print('grad norm = ', grad_norm)

  # Log loss, grad_norm.
  if ((global_step <= 100 or global_step % 500 == 0) and
      workload.metrics_logger is not None):
    workload.metrics_logger.append_scalar_metrics(
        {
            'loss': loss,
            'grad_norm': grad_norm,
        }, global_step)
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
    return 256
  elif workload_name == 'librispeech_deepspeech':
    return 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 128
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')


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
  del global_step
  del rng
  batch = next(input_queue)
  
  # Shard batch correctly 
  for key in batch.keys():
    array_shape = batch[key].shape
    sharded_array = jax.jit(lambda x: x, out_shardings=p)(batch[key].reshape(
      array_shape[0] * array_shape[1], -1
    ))

    batch[key] = sharded_array

  return batch
