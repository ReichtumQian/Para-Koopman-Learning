import numpy as np
import scipy.integrate
import scipy.optimize
import torch
import scipy
import joblib
from tqdm import tqdm
from scipy.optimize import fsolve
from .Factory import *
from .Log import *
from . import Parallel


class ODESolver:

  def __init__(self, ode, t_step, dt=1e-3):
    """
    Initializes the ODESolver with the given ordinary differential equation (ODE), 
    time step, and optional time increment.

    Args:
      ode (AbstractODE): The ordinary differential equation to be solved.
      t_step (float): The time step for the solver.
      dt (float, optional): The time increment for each step. Defaults to 1e-3.
    """
    self._ode = ode
    self._t_step = t_step
    self._dt = dt

  def step(self, x, u):
    """Perform a single step of the ODE solver.

    Args:
      x (torch.Tensor) : The current state of the system.
      u (torch.Tensor): The control input or parameters on the system.

    Returns:
      NotImplementedError: This method is intended to be overridden by subclasses to implement specific ODE solving logic.
    """
    return NotImplementedError


class ForwardEuler(ODESolver):

  def step(self, x, u):
    n_step = int(self._t_step / self._dt)
    for _ in range(n_step):
      x = x + self._dt * self._ode.rhs(x, u)
    return x


class BackwardEuler(ODESolver):

  def step(self, x, u):
    n_step = int(self._t_step / self._dt)
    x_numpy = x.detach().numpy()
    N = x_numpy.shape[0]
    d = x_numpy.shape[1]
    for _ in range(n_step):

      def equ(x_next):
        x_next = np.reshape(x_next, (N, d))
        result = x_next - x_numpy - self._dt * self._ode.rhs(
            torch.from_numpy(x_next), u).numpy()
        return result.flatten()

      x_numpy = fsolve(equ, x_numpy)
      x_numpy = np.reshape(x_numpy, (N, d))
    return torch.from_numpy(x_numpy).to(torch.float32)


class RungeKutta4(ODESolver):

  def step(self, x, u):
    ode = self._ode
    n_step = int(self._t_step / self._dt)
    for _ in range(n_step):
      k1 = self._dt * ode.rhs(x, u)
      k2 = self._dt * ode.rhs(x + 0.5 * k1, u)
      k3 = self._dt * ode.rhs(x + 0.5 * k2, u)
      k4 = self._dt * ode.rhs(x + k3, u)
      x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


class ScipyODESolver(ODESolver):

  def step(self, x, u):
    debug_message(f"[{self._solver_type}] Start stepping...")
    ode = self._ode

    def rhs(t, y, param):
      return ode.rhs(torch.from_numpy(y).unsqueeze(0),
                     param.unsqueeze(0)).squeeze(0).numpy()

    def solve_ivp(i):
      return scipy.integrate.solve_ivp(rhs, (0, self._t_step),
                                       x[i, :].detach().numpy(),
                                       args=(param[i, :], ),
                                       method=self._solver_type)

    data_size = x.size(0)
    if u.size(0) == 1:
      param = u.expand(data_size, u.size(1))
    else:
      param = u
    y_list = []
    results = joblib.Parallel(n_jobs=Parallel.get_n_jobs())(
        joblib.delayed(solve_ivp)(i)
        for i in tqdm(range(x.size(0)), desc="Stepping...", leave=False))
    for result in results:
      y_list.append(result)
    y = np.stack([y_list[i].y[:, -1] for i in range(x.size(0))])
    debug_message(f"[{self._solver_type}] Finish stepping...")
    return torch.from_numpy(y).float()


class RungeKutta23(ScipyODESolver):

  def __init__(self, ode, t_step, dt=1e-3):
    super().__init__(ode, t_step, dt)
    self._solver_type = 'RK23'


class RungeKutta45(ScipyODESolver):

  def __init__(self, ode, t_step, dt=1e-3):
    super().__init__(ode, t_step, dt)
    self._solver_type = 'RK45'


# Factory
ODESOLVERFACTORY = Factory()
ODESOLVERFACTORY.register("forward euler", ForwardEuler)
ODESOLVERFACTORY.register("backward euler", BackwardEuler)
ODESOLVERFACTORY.register("rk4", RungeKutta4)
ODESOLVERFACTORY.register("rk23", RungeKutta23)
ODESOLVERFACTORY.register("rk45", RungeKutta45)


def register_ode_solver(name, solver):
  ODESOLVERFACTORY.register(name, solver)
