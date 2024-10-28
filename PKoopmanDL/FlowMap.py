

class FlowMap:
  
  def __init__(self, t_step, dt = 1e-3):
    """Initialize the FlowMap instance.

    Args:
        dt (float): The time step size of the flow map. Defaults to 1e-3.
    """
    self._t_step = t_step
    self._dt = dt

  def step(self, ode, x, u):
    return NotImplementedError

class ForwardEuler(FlowMap):

  def step(self, ode, x, u):
    """Apply one step of the Forward Euler method.

    Args:
        ode (AbstractODE): The ODE system.
        x (tensor): The current state.
        u (tensor): The parameter of the ODE system.

    Returns:
        tensor: The state after one step of the Forward Euler method.
    """
    n_step = int(self._t_step / self._dt)
    for _ in range(n_step):
      x = x + self._dt * ode.rhs(x, u)
    return x
