
The class `EDMDDLSolver` implements the EDMD-DL algorithm.

## Attributes

- `__reg`: The regularization factor $\lambda$ used in the computation of $K$.

## Methods

- `__init__(self, dictionary, reg)`
    - `dictionary` (TrainableDictionary)
    - `reg` (float)
- `solve(self, dataset, n_epochs, batch_size, tol)`: 
  Applies the EDMD-DL algorithm to solve the system.
    - `dataset` (ODEDataSet): The dataset to solve.
    - `n_epochs` (int): The number of epochs.
    - `batch_size` (int): The batch size.
    - `tol` (float): The tolerance of the solver.
    - `return` (Koopman): The Koopman operator with a neural network function.




