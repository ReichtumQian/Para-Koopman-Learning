
The class `SolverWrapper` is a wrapper for each solver, it offers a common interface for reading the config file, setting up the solver and solving the problem.


## Attributes

- `_data` (dict): The dict read from the config file.

### ODE

- `equ_type` (str): The type of the ODE.
- `ode` (AbstractODE): The ODE object.

### Flow map

- `dt`: (float): The time step size of the flow map.
- `t_step` (float): The time step size of the trajectory.
- `flowmap` (FlowMap): The flow map object.

### Dataset

- `n_traj` (int): The number of trajectories.
- `traj_len` (int): The length of each trajectory.
- `x_min, x_max` (int or list): The minimum and maximum values of the initial conditions.
- `seed_x` (int): The seed for the initial conditions.
- `dataset` (ODEDataSet or ParamODEDataSet): The dataset object.

### Dictionary

- `dim_output` (int): The dimension of the output.
- `dim_nontrain`
- `dictionary`

### Solver

- `solver_type`
- `solver`

## Methods

- `__init__(self, config_file)`: Read the config file and initialize the attributes.
- `setup(self, nontrain_func)`: Setup the solver, including `ode`, `flowmap`, `dataset`, `dictionary` and `solver`.
- `solve(self)`
- `_read_ode_config(self, config_file)`
- `_read_flowmap_config(self, config_file)`
- `_read_dataset_config(self, config_file)`
- `_read_dictionary_config(self, config_file)`
- `_read_solver_config(self, config_file)`
- `_init_ode(self)`
- `_init_flowmap(self)`
- `_init_dataset(self)`
- `_init_dictionary(self, nontrain_func)`
- `_init_solver(self)`

!!! note
    You can manually change the attributes before calling `setup`.


