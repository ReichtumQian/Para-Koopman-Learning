import torch
import scipy
import numpy as np
import numbers
from tqdm import tqdm
from .Log import *


class KoopmanMPCSolver:

  def __init__(self, dynamics, koopman, dictionary, ref_traj, time_horizon,
               observable_pos, lambda_param):
    """Initializes an instance of the KoopmanMPCSolver class.

    Args:
        dynamics (KoopmanDynamics): The Koopman dynamics system.
        koopman (ParamKoopman): The parametric Koopman operator.
        dictionary (TrainableDictionary or Dictionary): A function that maps the state to the dictionary of observables.
        ref_traj (torch.Tensor): A tensor containing the reference trajectory, with shape (trajectory_length, state_dimension).
        time_horizon (int): The time horizon for model predictive control.
        observable_pos (int): The position(s) of target observables in dictionary.
        lambda_param (float): A regularization parameter used to balance the control error and tracking error.
    """

    self._time_horizon = time_horizon
    self._ref_traj = ref_traj
    self._observable_pos = observable_pos
    self._dynamics = dynamics

    def loss_func(control_in, x_in, start_time):
      x = torch.from_numpy(x_in).unsqueeze(0)
      control = torch.from_numpy(control_in).unsqueeze(0).to(torch.float32)
      control = control.reshape(time_horizon, -1)
      ref = ref_traj[start_time:start_time + time_horizon, :]
      m = torch.ones(1, time_horizon)
      psi = dictionary(x)
      for i in range(time_horizon):
        psi = koopman(psi, control[i].unsqueeze(0))
        obs_next = psi[:, observable_pos]
        m[:, i] = obs_next
      ref_loss = torch.sum(torch.square(m - ref)).item()
      control_loss = torch.sum(torch.square(control)).item()
      result = ref_loss + lambda_param * control_loss
      return result

    self._loss_func = loss_func

  def solve(self,
            state0,
            control_min,
            control_max,
            method='powell',
            disp=False):
    """Solve the optimal control problem for a given initial state and control bounds.
    Args:
        state0 (torch.Tensor): The initial state of the system.
        control_min (float or list of floats): Minimum allowable value(s) for the control input(s).
        control_max (float or list of floats): Maximum allowable value(s) for the control input(s).
        method (str, optional): Optimization method to use. Defaults to 'powell'.
        disp (bool, optional): Whether to print convergence messages. Defaults to False.

    Returns:
        torch.Tensor: The optimal control sequence over the given time horizon.
    """
    if isinstance(control_min, numbers.Number):
      control_min = [control_min] * self._dynamics.param_dim
    if isinstance(control_max, numbers.Number):
      control_max = [control_max] * self._dynamics.param_dim

    tau = self._time_horizon
    control_dim = self._dynamics.param_dim
    state_traj = [state0.detach().numpy()]
    bounds = []
    # generate init controls
    for i in range(tau):
      for j in range(control_dim):
        bounds.append((control_min[j], control_max[j]))
    lower_bounds, upper_bounds = zip(*bounds)
    bounds = scipy.optimize.Bounds(lower_bounds, upper_bounds)
    np.random.seed(0)  # for debugging
    control_init = np.random.uniform(0, 1, size=(tau, control_dim))
    for i in range(control_dim):
      control_init[:,
                   i] = control_init[:, i] * (control_max[i] -
                                              control_min[i]) + control_min[i]
    control_init = control_init.flatten()
    traj_len = self._ref_traj.size(0)

    # solve the optimal control problem
    opt_control_list = []
    info_message("[KoopmanMPCSolver] Solving the optimal control problem...")
    if traj_len <= tau:
      raise ValueError(
          "The reference trajectory is smaller than or equal to the time horizon!"
      )
    pbar = tqdm(range(traj_len - tau), desc="Solving")
    for t in pbar:
      results = scipy.optimize.minimize(self._loss_func,
                                        x0=control_init,
                                        args=(state_traj[-1], t),
                                        bounds=bounds,
                                        method=method,
                                        options={'disp': disp})
      if not results.success:
        warning_message(
            f"[KoopmanMPCSolver] Optimization failed! Reason: {results.message}"
        )
        # print(results.x)
        # warning_message(f"Jac = {results.jac}")
      loss = f"{results.fun:.2e}"
      pbar.set_postfix(loss=loss)
      if t == traj_len - tau - 1:
        controls = results.x.reshape(tau, control_dim)
        for it in controls:
          control = torch.from_numpy(it).unsqueeze(0)
          opt_control_list.append(control)
          state = torch.from_numpy(state_traj[-1]).unsqueeze(0)
          state_traj.append(
              self._dynamics.step(state, control).squeeze(0).detach().numpy())
        continue
      control = torch.from_numpy(results.x.reshape(tau,
                                                   control_dim)[0]).unsqueeze(0)
      control_init = results.x.reshape((tau * control_dim, ))
      state = torch.from_numpy(state_traj[-1]).unsqueeze(0)
      state_traj.append(
          self._dynamics.step(state, control).squeeze(0).detach().numpy())
      opt_control_list.append(control)

    opt_control_list = torch.cat(opt_control_list, dim=0)
    state_traj = torch.from_numpy(np.array(state_traj))
    return opt_control_list
