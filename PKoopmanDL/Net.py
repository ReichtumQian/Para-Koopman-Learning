
import torch

class FullConnResNet(torch.nn.Module):

  def __init__(self, input_dim=1, layer_size=[64, 64], output_dim=1, activation='tanh'):
    """ Initialize the FullConnResNet instance.

    Args:
        input_dim (int, optional): The input dimension. Defaults to 1.
        layer_size (list, optional): The list of layer sizes. Defaults to [64, 64].
        output_dim (int, optional): The output dimension. Defaults to 1.
        activation (str, optional): The activation function. Defaults to 'tanh'.
    """
    super().__init__()
    # input layer
    self._input_layer = torch.nn.Linear(input_dim, layer_size[0], bias = False)
    # hidden layers
    self._hidden_layers = torch.nn.ModuleList()
    for in_features, out_features in zip(layer_size[:-1], layer_size[1:]):
      self._hidden_layers.append(torch.nn.Linear(in_features, out_features))
    # output layer
    self._output_layer = torch.nn.Linear(layer_size[-1], output_dim)
    self._activation_type = activation

  def forward(self, inputs):
    """Apply the network to the input `inputs`

    Args:
        inputs (torch.Tensor): The input $\mathbb{R}^{N \times N_x}$.

    Returns:
        torch.Tensor: The output $\mathbb{R}^{N \times N_y}$.
    """
    hidden_u = self.input_layer(inputs)

    for layer in self._hidden_layers:
      hidden_u = layer(hidden_u)
      if self._activation_type == 'tanh':
        hidden_u = torch.tanh(hidden_u)
      elif self._activation_type == 'relu':
        hidden_u = torch.relu(hidden_u)
      else:
        raise NotImplementedError
      
    return self._output_layer(hidden_u)



