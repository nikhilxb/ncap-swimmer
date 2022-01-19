"""
Train swimmer body using ES algorithm (from ESTorch).
"""
import click
import os
import sys
import time
import shutil
import torch
import numpy as np
from mpi4py import MPI
from dowel import logger, tabular
from dm_control import suite
from acme import wrappers
from garage import wrap_experiment, rollout
from garage.envs.dm_control import DMControlEnv
from garage.experiment import deterministic
from garage.experiment.snapshotter import Snapshotter

from garage.torch.policies import GaussianMLPPolicy
from neurobo.ncap.swimmer import SwimmerPolicy
import neurobo.ncap.envs
import estorch


@click.command(context_settings={'show_default': True})
@click.option('--job', default='0', help='Job id on cluster.')
@click.option('--task', default='swim', type=click.Choice(['swim', 'navigate']), help='Task name.')
@click.option('--n_epochs', default=2, help='Num epochs total.')
@click.option('--n_proc', default=1, help='Num of processes to use for distributed training.')
@click.option('--population_size', default=8, help='Num perturbations per epoch. Must be even and multiple of number of processes.')
@click.option('--sigma', default=0.02, help='Standard deviation of Gaussian noise perturbation.')
@click.option('--optim', default='adam', type=click.Choice(['adam', 'sgd']), help='Name of optimizer.')
@click.option('--lr', default=0.01, help='Learning rate of optimizer.')
@click.option('--l2_coeff', default=0.005, help='L2 regularization coefficient.')
@click.option('--n_evals', default=1, help='Num evaluation episodes per epoch.')
@click.option('--policy', default='mlp', type=click.Choice(['mlp', 'ncap']), help='Policy network architecture.')
@click.option('--policy_hidden_sizes', default='64,64', help='Policy network hidden layers sizes.')
@click.option('--ncap_opts', default='111', help='Bit string for NCAP options [ws, sign, mag].')
@click.option('--osc_period', default=60, help='Swimmer oscilator period.')
@click.option('--seed', default=1, help='Random seed.')
@wrap_experiment(
  prefix='experiments/swimmer/swimmer_es',
  name_parameters='passed',
  snapshot_mode='gap_and_last',
  snapshot_gap=5,
)
def swimmer_es(
  ctxt,
  job,
  task,
  n_epochs,
  n_proc,
  population_size,
  sigma,
  optim,
  lr,
  l2_coeff,
  n_evals,
  policy,
  policy_hidden_sizes,
  ncap_opts,
  osc_period,
  seed,
):
  assert population_size % n_proc == 0 and population_size % 2 == 0

  # Extract MPI multi-process info.
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  seed = 1000 * seed + rank  # Worker-specific seed based on task id.
  deterministic.set_seed(seed)

  env = DMControlEnv(suite.load('swimmer', task, task_kwargs={'random': seed}))

  assert policy in ('mlp', 'ncap')
  if policy == 'mlp':
    Policy = GaussianMLPPolicy
    policy_kwargs = dict(
      env_spec=env.spec,
      hidden_sizes=tuple(int(h) for h in policy_hidden_sizes.split(',')),
      hidden_nonlinearity=torch.tanh,
      output_nonlinearity=None,
    )
  elif policy == 'ncap':
    assert len(ncap_opts) == 3
    sharing, constraints, constant_init = tuple(bool(int(x)) for x in ncap_opts)
    Policy = SwimmerPolicy
    policy_kwargs = dict(
      env_spec=env.spec,
      n_joints=env.spec.action_space.shape[0],
      oscillator_period=osc_period,
      use_task=None,
      use_weight_sharing=sharing,
      use_weight_constraints=constraints,
      use_weight_constant_init=constant_init,
    )

  assert optim in ('adam', 'sgd')
  if optim == 'adam':
    optimizer = torch.optim.Adam
  elif optim == 'sgd':
    optimizer = torch.optim.SGD

  class Agent():
    def __init__(self, device=torch.device('cpu')):
      self.device = device
      self.env = DMControlEnv(suite.load('swimmer', task, task_kwargs={'random': seed}))

    def rollout(self, policy):
      total_reward = 0.
      for _ in range(n_evals):
        data = rollout(self.env, policy, deterministic=True)
        total_reward += sum(data['rewards'])
      return total_reward / n_evals

  start_time = time.time()
  epoch_time = time.time()

  snapshotter = Snapshotter(
    snapshot_dir=ctxt.snapshot_dir,
    snapshot_mode=ctxt.snapshot_mode,
    snapshot_gap=ctxt.snapshot_gap,
  )

  class ES(estorch.ES):
    def log(self):
      nonlocal epoch_time, start_time
      now = time.time()
      logger.log(f'Time {now - start_time:.2f} s')
      logger.log(f'EpochTime {now - epoch_time:.2f} s')
      epoch_time = now
      epoch = self.step + 1

      tabular.record('Epoch', epoch)
      tabular.record('TotalEnvSteps', epoch * population_size * n_evals * env.spec.max_episode_length)
      tabular.record('CurrentReturn', self.episode_reward)
      tabular.record('PopulationAvgReturn', np.mean(self.population_returns))
      tabular.record('PopulationMinReturn', np.min(self.population_returns))
      tabular.record('PopulationMaxReturn', np.max(self.population_returns))
      logger.log(tabular)

      logger.dump_all(epoch)
      tabular.clear()

      snapshotter.save_itr_params(
        epoch,
        dict(
          policy_cls=Policy,
          policy_kwargs=policy_kwargs,
          policy_dict=es.policy.state_dict(), 
          best_policy_dict=es.best_policy_dict, 
        ),
      )      

  es = ES(
    policy=Policy,
    agent=Agent,
    optimizer=torch.optim.Adam,
    population_size=population_size,
    sigma=sigma,
    device=torch.device('cpu'),
    policy_kwargs=policy_kwargs,
    agent_kwargs=dict(),
    optimizer_kwargs=dict(lr=lr, weight_decay=l2_coeff),
  )

  if rank == 0 and os.getenv('MPI_PARENT') is not None:
    # Manager process is only one that logs.
    logger.log(f'Training started for {n_epochs} epochs on worker {rank} of {size} ...')
  elif n_proc > 1:
    # Worker processes shouldn't have a log dir.
    shutil.rmtree(ctxt.snapshot_dir)

  es.train(n_steps=n_epochs, n_proc=n_proc, hwthread=True)
  logger.log(f'Training done.')


if __name__ == '__main__':
  swimmer_es()
