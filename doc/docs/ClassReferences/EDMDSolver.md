
The class `EDMDSolver` implements the EDMD algorithm.

## Attributes

- `_dictionary` (Dictionary): A dictionary of basis functions.

## Methods

- `__init__(self, dictionary)`
- `solve(self, dataset)`: 
  Applies the EDMD algorithm to solve the system,
    - `dataset` (ODEDataSet): The dataset to solve.
    - Returns (Koopman): The Koopman operator corresponding to the dictionary.




