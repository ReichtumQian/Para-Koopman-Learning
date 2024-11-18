import torch
from tqdm import tqdm


class TransitionFunction:

  def step(self, x, u):
    return NotImplementedError


class DiscreteDynamics:

  def __init__(self, trans_func, dim, param_dim=0):
    self._trans_func = trans_func
    self._dim = dim
    self._param_dim = param_dim

  def step(self, x, u):
    return self._trans_func.step(x, u)

  @property
  def dim(self):
    return self._dim

  @property
  def param_dim(self):
    return self._param_dim

  def traj(self, x0, u0, traj_len):
    x = [x0]
    if u0.size(0) == 1:
      u = u0.expand(traj_len, -1)
    else:
      assert (u0.size(0) == traj_len - 1)
      u = u0
    for i in range(traj_len - 1):
      x.append(self.step(x[-1], u[i].unsqueeze(0)))
    return torch.stack(x, dim=1)  # size: (N, traj_len, number of state)


class KoopmanODEDynamics(DiscreteDynamics):

  def __init__(self, ode, koopman, dictionary, state_dim, param_dim=0):
    super().__init__(koopman, state_dim, param_dim)
    self._koopman = koopman
    self._dictionary = dictionary
    self._ode = ode

  def step(self, x, u):
    psi = self._dictionary(x)
    return self._trans_func.step(psi, u)

  def traj(self, x0, u0, traj_len):
    if u0.size(0) == 1:
      u = u0.expand(traj_len, -1)
    else:
      assert u0.size(0) == traj_len - 1
      u = u0
    x = [self._dictionary(x0)]
    state = x0
    for i in tqdm(range(traj_len - 1), desc='Generating trajectory'):
      x.append(self.step(state, u[i].unsqueeze(0)))
      state = self._ode.step(state, u[i].unsqueeze(0))
    return torch.stack(x, dim=1)  # size: (N, traj_len, number of state)


class KoopmanStateDynamics(DiscreteDynamics):

  def __init__(self, koopman, dictionary, state_pos, state_dim, param_dim=0):
    super().__init__(koopman, state_dim, param_dim)
    self._dictionary = dictionary
    self._state_pos = state_pos

  def step(self, x, u):
    psi = self._dictionary(x)
    return self._trans_func.step(psi, u)[:, self._state_pos]
