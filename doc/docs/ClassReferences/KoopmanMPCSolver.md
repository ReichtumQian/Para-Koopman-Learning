
## Attributes


## Methods

- `__init__(self, dynamics, koopman, dictionary, ref_traj, time_horizon, observable_pos, lambda_param)`
  - `ref_traj` (tensor): The reference trajectory of the size `(traj_len, 1)`, where `traj_len` is the length of the reference trajectory.
- `solve(self, x0, control_min, control_max)`
  - `x0` (tensor): Initial state of the size `(1, state_dim)`
  - `control_min` (list): The minimum control input of the length `control_dim`
  - `control_max` (list): The maximum control input of the length `control_dim`
  - Return1 (tensor): The optimal control input of the size `(traj_len, control_dim)`, where `traj_len` is the length of the reference trajectory.
  - Return2 (tensor): The optimal observable trajectory of the size `(traj_len, 1)`, where `traj_len` is the length of the reference trajectory.

