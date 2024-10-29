
The class `ParamODEDataSet` is a subclass of `ODEDataSet`,
it provides data for `ParamKoopmanDLSolver`.

## Attributes

- `_data_param` (tensor): The parameter data.

## Methods


- `__getitem__(self, idx)`
    - `idx` (int): The index of the data to be returned.
    - Returns: A tuple containing `data_x`, `data_param` and `labels`.
- `generate_data(self, n_traj, traj_len, x_min, x_max, param_min, param_max, seed_x = 11, seed_param = 22)`: Generates the data for the parameter-dependent ODE.
    - `n_traj` (int): Number of trajectories.
    - `traj_len` (int): Length of each trajectory.
    - `x_min` (float): Minimum value for the state variable.
    - `x_max` (float): Maximum value for the state variable.
    - `param_min` (float): Minimum value for the parameter.
    - `param_max` (float): Maximum value for the parameter.
    - `seed_x` (int): Seed for the state variable.
    - `seed_param` (int): Seed for the parameter.
    - Effects: Generates the data for the parameter-dependent ODE and stores it in `_data_x`, `_labels` and `_data_param`.


