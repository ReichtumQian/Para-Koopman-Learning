
Ths class `KoopmanMPCSolver` is used to solve the optimal control problem using Koopman operator and model predictive control (MPC).

## Attributes

- `_dynamics`: The discrete dynamics of the optimal control problem.
- `_loss_func`: The loss function of the optimal control problem.
- `_time_horizon`: The time horizon parameter $\tau$ of the MPC algorithm.
- `_ref_traj`: The reference trajectory of the size `(traj_len, 1)`.
- `_observable_pos`: The position of the observable in the dictionary, corresponding to the reference trajectory.

## Methods

- `__init__(self, dynamics, koopman, dictionary, ref_traj, time_horizon, observable_pos, lambda_param)`
    - `dynamics` (DiscreteDynamics): The discrete dynamics of the system.
    - `koopman` (ParamKoopman): The Koopman operator of the system.
    - `dictionary` (Dictionary): The dictionary of the system.
    - `ref_traj` (tensor): The reference trajectory of the size `(traj_len, 1)`.
    - `time_horizon` (int): The time horizon $\tau$ of the MPC algorithm.
    - `observable_pos` (int): The position of the observable in the dictionary.
    - `lambda_param` (float): The regularization parameter $\lambda$ of the MPC algorithm.
- `solve(self, x0, control_min, control_max)`
    - `x0` (tensor): Initial state of the size `(1, state_dim)`
    - `control_min` (list or float): The minimum control input of the length `control_dim`
    - `control_max` (list or float): The maximum control input of the length `control_dim`
    - Return (tensor): The optimal control of the size `(traj_len, control_dim)`, where `traj_len` is the length of the reference trajectory.

