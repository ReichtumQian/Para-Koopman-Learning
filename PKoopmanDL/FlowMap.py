import numpy as np
import torch
from scipy.optimize import fsolve
from .Factory import *


class FlowMap:

  def __init__(self, t_step, dt=1e-3):
    """Initialize the FlowMap instance.

    Args:
        dt (float): The time step size of the flow map. Defaults to 1e-3.
    """
    self._t_step = t_step
    self._dt = dt

  def generate_traj_data(self, ode, x0, u, traj_len):
    x = [x0]
    for t in range(traj_len - 1):
      x.append(self.step(ode, x[-1], u))
    x = torch.cat(x, dim=0)
    return x

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


class BackwardEuler(FlowMap):

  def step(self, ode, x, u):
    """Apply one step of the Backward Euler method.

    Args:
        ode (AbstractODE): The ODE system.
        x (tensor): The current state.
        u (tensor): The parameter of the ODE system.

    Returns:
        tensor: The state after one step of the Backward Euler method.
    """
    n_step = int(self._t_step / self._dt)
    x_numpy = x.detach().numpy()
    N = x_numpy.shape[0]
    d = x_numpy.shape[1]
    for _ in range(n_step):

      def equ(x_next):
        x_next = np.reshape(x_next, (N, d))
        result = x_next - x_numpy - self._dt * ode.rhs(torch.from_numpy(x_next),
                                                       u).numpy()
        return result.flatten()

      x_numpy = fsolve(equ, x_numpy)
      x_numpy = np.reshape(x_numpy, (N, d))
    return torch.from_numpy(x_numpy).to(torch.float32)


class RungeKutta4(FlowMap):

  def step(self, ode, x, u):
    """Apply one step of the 4th-order Runge-Kutta method.

    Args:
        ode (AbstractODE): The ODE system.
        x (tensor): The current state.
        u (tensor): The parameter of the ODE system.

    Returns:
        tensor: The state after one step of the 4th-order Runge-Kutta method.
    """
    n_step = int(self._t_step / self._dt)
    for _ in range(n_step):
      k1 = self._dt * ode.rhs(x, u)
      k2 = self._dt * ode.rhs(x + 0.5 * k1, u)
      k3 = self._dt * ode.rhs(x + 0.5 * k2, u)
      k4 = self._dt * ode.rhs(x + k3, u)
      x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


# Factory
FLOWMAPFACTORY = Factory()
FLOWMAPFACTORY.register("forward euler", ForwardEuler)
FLOWMAPFACTORY.register("backward euler", BackwardEuler)
FLOWMAPFACTORY.register("rk4", RungeKutta4)
