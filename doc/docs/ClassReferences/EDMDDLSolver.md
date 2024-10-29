
The class `EDMDDLSolver` implements the EDMD-DL algorithm.
It's a subclass of [[EDMDSolver.md | EDMDSolver]].

## Attributes

- `_reg` (float): The regularization factor $\lambda$ used in the computation of $K$.
- `_reg_final` (float): The final regularization factor $\lambda$ used in the computation of $K$.

## Methods

- `__init__(self, dictionary, reg, reg_final)`
    - `dictionary` (TrainableDictionary)
    - `reg` (float)
- `solve(self, dataset, n_epochs, batch_size, tol = 1e-8, lr = 1e-4)`: 
  Applies the EDMD-DL algorithm to solve the system.
    - `dataset` (ODEDataSet): The dataset to solve.
    - `n_epochs` (int): The number of epochs.
    - `batch_size` (int): The batch size.
    - `tol` (float): The tolerance of the solver.
    - `lr` (float): The learning rate.
    - Returns (Koopman): The Koopman operator with a nerual network function.




