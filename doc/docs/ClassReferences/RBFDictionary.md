
The class `RBFDictionary` is the implementation of the RBF dictionary,
it's a subclass of [[Dictionary.md | Dictionary]].

## Methods

- `__init__(self, data_x, nontrain_func, dim_input, dim_output, dim_nontrain, reg)`
    - `data_x` (numpy array): The input data of the states, used to compute the RBF centers.
    - `nontrain_func` (tensor -> tensor): The non-trainable function that maps the input data to the output data.
    - `dim_input` (int): The dimension of the input data.
    - `dim_output` (int): The dimension of the output data.
    - `dim_nontrain` (int): The dimension of the non-trainable part of the output data.
    - `reg` (float): The regularization parameter used in the RBF dictionary.

