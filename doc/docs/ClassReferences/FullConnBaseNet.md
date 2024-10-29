
The class `FullConnBaseNet` is a subclass of `torch.nn.Module`.
It implements a basic structure of fully-connected network
without the `forward` method.

## Attributes

- `_input_layer` (torch.nn.Linear): The input layer.
- `_hidden_layers` (torch.nn.Module): The list of hidden layers.
- `_output_layer` (torch.nn.Linear): The output layer.
- `_activation_type` (str): The activation function type.

## Methods

- `__init__(self, input_dim=1, output_dim=1, layer_size=[64, 64], activation='tanh')`
    - `input_dim` (int): The input dimension.
    - `layer_size` (list): The list of hidden layer sizes.
    - `output_dim` (int): The output dimension.
    - `activation` (str): The activation function to use.
    - Effects: Initialize the network, set the `__input_layer`, `__hidden_layer`, and `__output_layer`.
- `_apply_activation(self, x)`: Applies the activation function to the input `x`.
    - `x` (tensor): The input to the activation function.
    - Returns: The output of the activation function.
- `forward(self, inputs)`: Applies the network to the input `inputs`, which must be in the form $\mathbb{R}^{N \times N_x}$.
    - `inputs` (tensor): The input to the network.
    - Note: This method must be implemented by subclasses.