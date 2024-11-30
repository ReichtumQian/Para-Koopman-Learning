
We explain in detail the parameters listed in the JSON files in `example/simple examples`.

## General Parameters

- `equ_type (string)`: Indicates the type of equation. 
  This type must be registered in the `ODEFACTORY`, see [[CustomizeODE | Customize ODE]] for more details.
- `ode_solver`:
  - `type (string)`: Indicates the type of ODE solver.
    This type must be registered in the `ODESOLVERFACTORY`, see [[CustomizeODESolver.md | Customize ODE Solver]] for more details.
  - `dt (float)`: Time step $k$ of the ODE Solver.
  - `t_step (float)`: Time step of the trajectories.
- `dataset`:
  - `n_traj (int)`: Number of trajectories.
  - `traj_len (int)`: Length of each trajectory.
  - `x_min/x_max (float or list)`: Range of the states.
    If a `float` is given, it means every state has the same range.
  - `seed_x (int)`: Seed for generating the states.
- `dictionary`:
  - `dim_output (int)`: Output dimension $N_{\Psi}$ of the dictionary.

## EDMD-RBF Algorithm

- `dataset`:
  - `param (list)`: Fixed parameters $u$ of the dataset.
- `dictionary`:
  - `reg (float)`: Regularization parameter $\lambda$ for generating the RBF functions.    

## EDMDDL Algorithm

- `dataset`:
  - `param (list)`: Fixed parameters $u$ of the dataset.
  - `train_ratio (float)`: Ratio of the training set.
- `dictionary`
  - `dic_layer_sizes (list)`: Hidden layer sizes of the dictionary.
- `solver`:
  - `reg (float)`: Regularization parameter $\lambda$ when training the Koopman operator.
  - `reg_final (float)`: Regularization parameter $\lambda$ when outputting the Koopman operator.
  - `n_epochs (int)`: Number of training epochs.
  - `batch_size (int)`: Batch size for training.
  - `tol (float)`: Tolerance of training.
  - `dic_lr (float)`: Learning rate for training the dictionary.

## Parametric Koopman Learning  

- `dataset`
  - `n_traj_per_param (int)`: Number of trajectories per parameter setting.
  - `param_min/param_max (list)`: Range of the parameters.
  - `seed_param (int)`: Seed for generating the parameters.
  - `param_time_dependent (bool)`: Indicates whether the parameters are time-dependent.
- `dictionary`:
  - `dic_layer_sizes (list)`: Hidden layer sizes of the dictionary.
- `solver`:
  - `n_epochs (int)`: Number of training epochs.
  - `batch_size (int)`: Batch size for training.
  - `tol (float)`: Tolerance of training.
  - `dic_lr (float)`: Learning rate for training the dictionary.
  - `koopman_layer_sizes (list)`: Hidden layer sizes for the Koopman operator.
  - `koopman_lr (float)`: Learning rate for training the Koopman operator.

