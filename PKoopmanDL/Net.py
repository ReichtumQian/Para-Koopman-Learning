import torch


class FullConnBaseNet(torch.nn.Module):

  def __init__(self,
               input_dim=1,
               output_dim=1,
               layer_size=[64, 64],
               activation='tanh'):
    """ Initialize the FullConnResNet instance.

    Args:
        input_dim (int, optional): The input dimension. Defaults to 1.
        layer_size (list, optional): The list of layer sizes. Defaults to [64, 64].
        output_dim (int, optional): The output dimension. Defaults to 1.
        activation (str, optional): The activation function. Defaults to 'tanh'.
    """
    super().__init__()
    # input layer
    self._input_layer = torch.nn.Linear(input_dim, layer_size[0], bias=False)
    # hidden layers
    self._hidden_layers = torch.nn.ModuleList()
    for in_features, out_features in zip(layer_size[:-1], layer_size[1:]):
      self._hidden_layers.append(torch.nn.Linear(in_features, out_features))
    # output layer
    self._output_layer = torch.nn.Linear(layer_size[-1], output_dim)
    self._activation_type = activation

  def _apply_activation(self, x):
    if self._activation_type == 'tanh':
      return torch.tanh(x)
    elif self._activation_type == 'relu':
      return torch.relu(x)
    else:
      raise NotImplementedError

  def forward(self, inputs):
    """Apply the network to the input `inputs`

    Args:
        inputs (tensor): The input $\mathbb{R}^{N \times N_x}$.

    Returns:
        tensor: The output $\mathbb{R}^{N \times N_y}$.
    """
    raise NotImplementedError


class FullConnResNet(FullConnBaseNet):

  def forward(self, inputs):
    hidden_u = self._input_layer(inputs)
    for layer in self._hidden_layers:
      residual = hidden_u
      hidden_u = self._apply_activation(layer(hidden_u))
      hidden_u = hidden_u + residual
    return self._output_layer(hidden_u)


class FullConnNet(FullConnBaseNet):

  def forward(self, inputs):
    hidden_u = self._input_layer(inputs)
    for layer in self._hidden_layers:
      hidden_u = self._apply_activation(layer(hidden_u))
    return self._output_layer(hidden_u)
