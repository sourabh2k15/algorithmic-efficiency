"""
Example Usage:
python run_workloads.py \
--workload_config_path workload_config.json \
--framework jax \
--experiment_name my_first_experiment \
--docker_image_url us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/algoperf_jax_dev \
--run_percentage 10 \
--workload_config_path workload_config.json \
--dry_run 
"""

import json
import os
import struct
import time

from absl import app
from absl import flags
from absl import logging

import docker

flags.DEFINE_string(
    'docker_image_url',
    'us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/algoperf_jax_dev',
    'URL to docker image')
flags.DEFINE_integer('run_percentage',
                     100,
                     'Percentage of max num steps to run for.')
flags.DEFINE_string('experiment_name',
                    'my_experiment',
                    'Name of top sub directory in experiment dir.')
flags.DEFINE_boolean('rsync_data',
                     True,
                     'Whether or not to transfer the data from GCP w rsync.')
flags.DEFINE_boolean('local', False, 'Mount local algorithmic-efficiency repo.')
flags.DEFINE_string('framework', 'jax', 'Can be either PyTorch or JAX.')
flags.DEFINE_boolean(
    'dry_run',
    False,
    'Whether or not to actually run the docker containers. '
    'If False, simply print the docker run commands. ')
flags.DEFINE_integer('num_studies', 1, 'Number of studies to run')
flags.DEFINE_integer('study_start_index', None, 'Start index for studies.')
flags.DEFINE_integer('study_end_index', None, 'End index for studies.')
flags.DEFINE_integer('num_tuning_trials', 1, 'Number of tuning trials.')
flags.DEFINE_integer('hparam_start_index',
                     None,
                     'Start index for tuning trials.')
flags.DEFINE_integer('hparam_end_index', None, 'End index for tuning trials.')
flags.DEFINE_integer('seed', None, 'Random seed for evaluating a submission.')
flags.DEFINE_integer('submission_id',
                     0,
                     'Submission ID to generate study and hparam seeds.')
flags.DEFINE_string(
    'workload_config_path',
    'workload_confing.json',
    'Path to config containing dataset and maximum number of steps per workload.'
    'The default values of these are set to the full budgets as determined '
    'via the target-setting procedure. '
    'Note that training will be interrupted at either the set maximum number '
    'of steps or the fixed workload maximum run time, whichever comes first. '
    'If your algorithm has a smaller per step time than our baselines '
    'you may want to increase the number of steps per workload.')

FLAGS = flags.FLAGS


def read_workloads(filename):
  with open(filename, "r") as f:
    held_out_workloads = json.load(f)
  return held_out_workloads


def container_running():
  docker_client = docker.from_env()
  containers = docker_client.containers.list()
  if len(containers) == 0:
    return False
  else:
    return True


def wait_until_container_not_running(sleep_interval=5 * 60):
  while container_running():
    time.sleep(sleep_interval)
  return


def main(_):
  # What Docker image to run the container with
  docker_image_url = FLAGS.docker_image_url

  # Framework
  framework = FLAGS.framework

  #
  run_fraction = FLAGS.run_percentage / 100.
  experiment_name = FLAGS.experiment_name

  # Get study and trial interval arguments
  num_studies = FLAGS.num_studies
  study_start_index = FLAGS.study_start_index if FLAGS.study_start_index else 0
  study_end_index = FLAGS.study_end_index if FLAGS.study_end_index else num_studies - 1

  # Get trial arguments
  num_tuning_trials = FLAGS.num_tuning_trials
  hparam_start_index_flag = ''
  hparam_end_index_flag = ''
  if FLAGS.hparam_start_index:
    hparam_start_index_flag = f'--hparam_start_index {FLAGS.hparam_start_index} '
  if FLAGS.hparam_end_index:
    hparam_end_index_flag = f'--hparam_end_index {FLAGS.hparam_end_index} '

  # Generate rng keys from submission_id and seed
  submission_id = FLAGS.submission_id
  rng_seed = FLAGS.seed

  if not rng_seed:
    rng_seed = struct.unpack('I', os.urandom(4))[0]

  logging.info('Using RNG seed %d', rng_seed)

  # Read workload specifications to run
  with open(FLAGS.workload_config_path) as f:
    workload_config = json.load(f)
  workloads = [w for w in workload_config.keys()]

  for study_index in range(study_start_index, study_end_index + 1):
    print('-' * 100)
    print('*' * 40, f'Starting study {study_index + 1}/{num_studies}', '*' * 40)
    print('-' * 100)
    study_dir = os.path.join(experiment_name, f'study_{study_index}')

    for workload in workloads:
      # For each runnable workload check if there are any containers running
      wait_until_container_not_running()

      # Clear caches
      os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
      print('=' * 100)

      # Get workload dataset, max step, algorithm path and tuning search space
      dataset = workload_config[workload]['dataset']
      max_steps = int(workload_config[workload]['max_steps'] * run_fraction)
      submission_path = workload_config[workload]['submission_path']
      tuning_search_space = workload_config[workload]['tuning_search_space']

      # Optionally, define flag to mount local algorithmic-efficiency repo
      mount_repo_flag = ''
      if FLAGS.local:
        mount_repo_flag = '-v $HOME/algorithmic-efficiency:/algorithmic-efficiency '

      command = ('docker run -t -d -v $HOME/data/:/data/ '
                 '-v $HOME/experiment_runs/:/experiment_runs '
                 '-v $HOME/experiment_runs/logs:/logs '
                 f'{mount_repo_flag}'
                 '--gpus all --ipc=host '
                 f'{docker_image_url} '
                 f'-d {dataset} '
                 f'-f {framework} '
                 f'-s {submission_path} '
                 f'-w {workload} '
                 f'-t {tuning_search_space} '
                 f'-e {study_dir} '
                 f'-m {max_steps} '
                 f'--num_tuning_trials {num_tuning_trials} '
                 f'{hparam_start_index_flag} '
                 f'{hparam_end_index_flag} '
                 f'--rng_seed {rng_seed} '
                 '-c false '
                 '-o true '
                 '-i true ')
      if not FLAGS.dry_run:
        print('Running docker container command')
        print('Container ID: ')
        return_code = os.system(command)
      else:
        return_code = 0
      if return_code == 0:
        print(
            f'SUCCESS: container for {framework} {workload} launched successfully'
        )
        print(f'Command: {command}')
        print(f'Results will be logged to {experiment_name}')
      else:
        print(
            f'Failed: container for {framework} {workload} failed with exit code {return_code}.'
        )
        print(f'Command: {command}')
      wait_until_container_not_running()
      os.system(
          "sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")  # clear caches

      print('=' * 100)


if __name__ == '__main__':
  app.run(main)
