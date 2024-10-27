
The class `ODEDataSet` is a subclass of `torch.utils.data.Dataset`,
it provides data for `EDMDSolver`, `EDMDDLSolver` and `ParamKoopmanDLSolver`.

## Attributes

- `__ode` (AbstractODE): The ODE system.
- `__flowmap` (FlowMap): The flow map of the ODE system.
- `__data_x` (ndarray): The state data of the system.
- `__data_param` (ndarray): The parameter data of the system.
- `__labels` (ndarray): The label data of the system.

## Methods

- `__init__(self, ode, flowmap)`
- `__len__(self)`: Returns the number of dataset.
- `__getitem__(self, idx)`: Returns the data at index `idx`, including `x`, `param` and `label`.
- `data_x(self)`: Returns the state data.
- `data_param(self)`: Returns the parameter data.
- `labels(self)`: Returns the label data.
- `generate_data(self, n_traj, traj_len, x_min, x_max, param_min, param_max, seed_x=11, seed_param=22)`
    - `n_traj` (int): The number of trajectories to generate.
    - `traj_len` (int): The length of each trajectory.
    - `x_min`, `x_max` (float): The range of the initial state.
    - `param_min`, `param_max` (float): The range of the parameter.
    - `seed_x`, `seed_param` (int): The seed for the random number generator.
    - Effects: Generate data and store in `self.__data_x`, `self.__data_param` and `self.__labels`.
