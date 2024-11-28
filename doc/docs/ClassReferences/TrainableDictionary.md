
The `TrainableDictionary` class is a subclass of [[Dictionary.md|Dictionary]] class.
It contains a trainable neural network and a set of $N_y$ non-trainable observable functions 
(see [[ObservableFunction.md|ObservableFunction]]).
The dictionary can be represented by the mapping
$\Psi: \mathbb{R}^{N \times N_x} \rightarrow \mathbb{R}^{N \times N_{\psi}}$.

!!! Note
	The constant observable function $\mathbf{1}$ is included in the `TrainableDictionary` by default,
	so users don't need to define it explicitly.

## Attributes

- `_dim_nontrain` (int): The number of non-trainable outputs.
- `_network` (torch.nn.Module): The trainable neural network.

!!! info
    The attribute `_function`, inherited from `Dictionary`, is a function that combines the neural network with non-trainable outputs. The first `dim_nontrain` outputs are non-trainable.

## Methods

- `__init__(self, network, nontrain_func, dim_input, dim_output, dim_nontrain)`
    - `network` (torch.nn.Module): The trainable neural network,
      which is a mapping $\Psi: \mathbb{R}^{N \times N_x} \rightarrow \mathbb{R}^{N \times (N_{\psi} - N_y)}$. 
    - `nontrain_func` (ndarray -> ndarray): The trainable neural network,
      which is a mapping $\Psi: \mathbb{R}^{N \times N_x} \rightarrow \mathbb{R}^{N \times N_y}$.
    - `dim_input` (int): The dimension of the input $N_x$.
    - `dim_output` (int): The dimension of the output $N_{\psi}$.
    - `dim_nontrain` (int): The number of non-trainable outputs $N_y$.
- `parameters(self)`: Return the trainable parameters of the network `__network`.
- `train(self)`: Set the network to training mode.
- `eval(self)`: Set the network to evaluation mode.


