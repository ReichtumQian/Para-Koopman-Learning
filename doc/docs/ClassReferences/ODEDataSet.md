
The class `ODEDataSet` is a subclass of `torch.utils.data.Dataset`,
it provides data for `EDMDSolver` and `EDMDDLSolver`.

## Attributes

- `_ode` (AbstractODE): The ODE system.
- `_flowmap` (FlowMap): The flow map of the ODE system.
- `_generated` (bool): Whether the data has been generated.
- `_data_x` (tensor): The state data of the system.
- `_labels` (tensor): The label data of the system.

## Methods

- `__init__(self, ode, flowmap)`
- `__len__(self)`: Returns the number of dataset.
- `__getitem__(self, idx)`: Returns the data at index `idx`, including `x` and `label`.
- `data_x(self)`: Returns the state data.
- `labels(self)`: Returns the label data.
- `generate_data(self, n_traj, traj_len, x_min, x_max, param, seed_x=11)`
    - `n_traj` (int): The number of trajectories to generate.
    - `traj_len` (int): The length of each trajectory.
    - `x_min`, `x_max` (float): The range of the initial state.
    - `param` (tensor): The parameter of the ODE system.
    - `seed_x` (int): The seed for the random number generator.
    - Effects: Generate data and store in `self._data_x` and `self._labels`.
