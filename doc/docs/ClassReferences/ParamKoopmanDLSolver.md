
The class `ParaKoopmanDLSolver` implements the algorithm of
learning parametric Koopman decomposition.

## Attributes

- `_dictionary` (TrainableDictionary)

## Methods

- `__init__(self, dictionary)`
- `solve(self, dataset, paramkoopman, n_epochs, batch_size, tol, lr_dic, lr_koop)`:
    - `dataset` (ParamODEDataSet): The dataset to solve.
    - `paramkoopman` (ParamKoopman): The initial parametric Koopman operator.
    - `n_epochs` (int): The number of epochs.
    - `batch_size` (int): The batch size.
    - `tol` (float): The tolerance for early stopping.
    - `lr_dic` (float): The learning rate for the dictionary.
    - `lr_koop` (float): The learning rate for the Koopman operator.
    - Returns (ParamKoopman): The trained parametric Koopman operator.
