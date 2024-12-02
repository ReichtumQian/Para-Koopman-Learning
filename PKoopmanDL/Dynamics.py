import torch
from tqdm import tqdm


class DiscreteDynamics:

  def __init__(self, trans_func, dim, param_dim=0):
    """
    Initializes the dynamics object with a transition function,  state dimension, and optional parameter dimension.

    Args:
      trans_func (torch.Tensor -> torch.Tensor): The transformation function that defines the dynamics.
      dim (int): The dimension of the state space.
      param_dim (int, optional): The dimension of the parameter space. 
    """
    self._trans_func = trans_func
    self._dim = dim
    self._param_dim = param_dim

  def step(self, x, u):
    """ Advances the system dynamics by one time step using the given state and parameter/control input.

    Args:
      x (torch.Tensor): The current state of the system.
      u (torch.Tensor): The control input to be applied.

    Returns:
      torch.Tensor: The next state of the system after applying the control input.
    """
    return self._trans_func.step(x, u)

  @property
  def dim(self):
    """Returns the dimension of the system.

    Returns:
      int: The dimension of the system.
    """
    return self._dim

  @property
  def param_dim(self):
    """Returns the dimension of the parameters.

    Returns:
      int: The dimension of the parameters.
    """
    return self._param_dim

  def traj(self, x0, u0, traj_len):
    """Simulate trajectory of a dynamical system.

    Args:
      x0 (torch.Tensor): The initial state of the system. Expected shape is (N, number of state).
      u0 (torch.Tensor or None): The control input or parameter. If None, a zero tensor is used.
                     If time-independent, should have shape (1, number of control inputs).
                     If time-dependent, should have shape (traj_len - 1, number of control inputs).
      traj_len (int): The length of the trajectory to simulate.

    Returns:
      torch.Tensor: A tensor containing the simulated trajectory. The shape is (N, traj_len, number of state).
    """
    x = [x0]
    # if do not need parameter
    if u0 == None:
      u0 = torch.zeros(1, 1)
    # if the parameters are time-independent
    if u0.size(0) == 1:
      u = u0.expand(traj_len - 1, -1)
    else:
      assert (u0.size(0) == traj_len - 1)
      u = u0
    for i in range(traj_len - 1):
      x.append(self.step(x[-1], u[i].unsqueeze(0)))
    return torch.stack(x, dim=1)  # size: (N, traj_len, number of state)


class KoopmanDynamics(DiscreteDynamics):
  """Koopman Dynamics of the form $\\Psi_{n+1} = K \\Psi_n$.
     The states are updated from the dictionary's output.
  """

  def __init__(self, koopman, dictionary, state_pos, state_dim, param_dim=0):
    """
    Initializes the Dynamics class with the given parameters.

    Args:
      koopman (Koopman or ParamKoopman): An instance representing the Koopman operator.
      dictionary (Dictionary or TrainableDictionary): A dictionary or function used for state transition.
      state_pos (list[int]): The position or index of the state in the input data.
      state_dim (int): The dimensionality of the state space.
      param_dim (int): The dimensionality of the parameter space. Defaults to 0.
    """
    super().__init__(koopman, state_dim, param_dim)
    self._dictionary = dictionary
    self._state_pos = state_pos

  def step(self, x, u):
    """
    Advances the system dynamics by one time step.

    Args:
      x (torch.Tensor): The current state of the system.
      u (torch.Tensor): The control input applied to the system.

    Returns:
      torch.Tensor: The next state of the system after applying the control input.
    """
    psi = self._dictionary(x)
    return self._trans_func.step(psi, u)[:, self._state_pos]


class KoopmanODEDynamics(DiscreteDynamics):
  """Koopman Dynamics of the form \\Psi_{n+1} = K \Psi_n.
     The states x are obtained from the ODESolver.
  """

  def __init__(self, ode_solver, koopman, dictionary, state_dim, param_dim=0):
    super().__init__(koopman, state_dim, param_dim)
    self._koopman = koopman
    self._dictionary = dictionary
    self._ode = ode_solver

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
