
The class `FullConnResNet` is a subclass of `torch.nn.Module`.
It represents a fully-connected residual network.

## Attributes

- `_input_layer` (torch.nn.Linear): The input layer.
- `_hidden_layer` (torch.nn.module): The list of hidden layers.
- `_output_layer` (torch.nn.Linear): The output layer.

## Methods

- `__init__(self, input_dim=1, layer_size=[64, 64], output_dim=1, activation='tanh')`
    - `input_dim` (int): The input dimension.
    - `layer_size` (list): The list of hidden layer sizes.
    - `output_dim` (int): The output dimension.
    - `activation` (str): The activation function to use.
    - Effects: Initialize the network, set the `__input_layer`, `__hidden_layer`, and `__output_layer`.
- `forward(self, inputs)`: Applies the network to the input `inputs`, which must be in the form $\mathbb{R}^{N \times N_x}$.
    - `inputs` (torch.Tensor): The input to the network.
