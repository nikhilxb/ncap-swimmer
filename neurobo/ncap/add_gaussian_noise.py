import akro
import numpy as np
from dowel import tabular
from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent
from garage.torch.policies.stochastic_policy import StochasticPolicy


class AddGaussianNoise(StochasticPolicy):
  """Add Gaussian noise to the action taken by the deterministic policy.

  Args:
    env_spec (garage.EnvSpec): Environment specification.
    policy (garage.torch.policies.Policy): Policy to wrap.
    total_timesteps (int): Total steps in the training, equivalent to
      max_episode_length * n_epochs.
    max_sigma (float): Action noise standard deviation at the start of
      exploration.
    min_sigma (float): Action noise standard deviation at the end of the
      decay period.
    decay_ratio (float): Fraction of total steps for epsilon decay.
    suffix (str): Suffic to append to policy name.
  """
  def __init__(
    self,
    env_spec,
    policy,
    total_timesteps,
    max_sigma=1.0,
    min_sigma=0.1,
    decay_ratio=1.0,
    suffix='AddGaussianNoise',
  ):
    assert isinstance(env_spec.action_space, akro.Box)
    assert len(env_spec.action_space.shape) == 1
    super().__init__(env_spec, f'{policy.name}{suffix}')
    self._policy = policy
    self._max_sigma = max_sigma
    self._min_sigma = min_sigma
    self._decay_period = int(total_timesteps * decay_ratio)
    self._action_space = env_spec.action_space
    self._decrement = (self._max_sigma - self._min_sigma) / self._decay_period
    self._total_env_steps = 0

  def forward(self, observations):
    """Compute the action distributions from the observations.

    Args:
      observations (torch.Tensor): Batch of observations, shape (N, observation_dim).
    
    Returns:
      (torch.distributions.Distribution): Batch distribution of actions, shape (N, action_dim).
      (dict[str, torch.Tensor]): Additional agent_info, as torch.Tensors
    """
    actions = self._policy(observations)
    self._total_env_steps += actions.shape[0]
    dist = Normal(actions, self._sigma())
    # Reinterprets some of the batch dims of a distribution as event dims.
    # This is mainly useful for changing the shape of the result of log_prob().
    dist = Independent(dist, 1)
    mean = dist.mean.cpu()
    log_std = (dist.variance.sqrt()).log().cpu()
    return dist, dict(mean=mean, log_std=log_std)

  def _sigma(self):
    """Get the current sigma.

    Returns:
      (float): Sigma.
    """
    if self._total_env_steps >= self._decay_period: return self._min_sigma
    return self._max_sigma - self._decrement * self._total_env_steps

  def get_param_values(self):
    """Get parameter values.

    Returns:
      (list or dict): Values of each parameter.
    """
    return {
      'total_env_steps': self._total_env_steps,
      'inner_params': self._policy.get_param_values(),
    }

  def set_param_values(self, params):
    """Set param values.
    
    Args:
      params (np.ndarray): A numpy array of parameter values.
    """
    self._total_env_steps = params['total_env_steps']
    self._policy.set_param_values(params['inner_params'])
