import numpy as np
import torch


class Koopman:

  def __init__(self, K):
    """Initialize the Koopman instance.

    Args:
        K (torch.Tensor): The Koopman matrix of the size $(N_{\\psi}, N_{\\psi})$
    """
    func = lambda x: (K @ x.t()).t()
    self._K = K
    self._func = func

  def __call__(self, x, u):
    """ Apply the Koopman operator $K(x)$, here the input $u$ is ignored.
    
    Args:
      x (torch.Tensor): The input dictionary basis of the size $(N, N_{\\psi})$.
      u (torch.Tensor): Ignored
    
    Returns:
      torch.Tensor: Applied the Koopman operator, it's of the size $(N, N_{\\psi})$.
    """
    return self._func(x)

  def step(self, x, u):
    """Consider the Koopman operator as a step function \\Psi_{n+1} = K \\Psi_n.

    Args:
        x (torch.Tensor): The input dictionary basis of the size $(N, N_{\\psi})$.
        u (torch.Tensor): Ignored

    Returns:
        torch.Tensor: Applied the Koopman operator, it's of the size $(N, N_{\\psi})$.
    """
    return self(x, u)

  def save(self, path):
    """Save the Koopman operator to a file.

    Args:
        path (str): The path to save the Koopman operator.
    """
    torch.save(self._K, path)

  def load(self, path):
    """Load the Koopman operator from a file.

    Args:
        path (str): The path to load the Koopman operator.
    """
    K = torch.load(path)
    self.__init__(K)


class ParamKoopman:

  def __init__(self, size_K, network):
    """Initialize the ParamKoopman instance.

    Args:
        size_K (int): The size of the Koopman matrix.
        network (torch.nn.Module): The network to generate the Koopman operator.
    """
    self._size = size_K
    self._network = network

  def __call__(self, x, u):
    """
    Applies the Koopman operator to the input tensor `x` using the parameter tensor `u`.

    Args:
      x (torch.Tensor): The input tensor of shape $(N, N_\\psi)$.
      u (torch.Tensor): The parameter tensor of shape $(1, N_u)$ or $(N, N_u)$. If `u` has a size of 1 in the first dimension,
                it will be expanded to match the batch size of `x`.

    Returns:
      torch.Tensor: The result of applying the Koopman operator, with the same batch size
              as `x` and the transformed dimension.
    """
    if u.size(0) == 1:
      net_param = u.expand(x.size(0), -1)
    else:
      net_param = u
    net_out = self._network(net_param)
    K = net_out.reshape(net_out.size(0), self._size, self._size)
    x = x.unsqueeze(2)
    result = torch.bmm(K, x).squeeze(2)
    return result

  def step(self, x, u):
    """Computes the next step in the system given the current state and parameter.

    Args:
      x (torch.Tensor): The current state of the system.
      u (torch.Tensor): The control input to the system.

    Returns:
      torch.Tensor: The next state of the system after applying the Koopman.
    """
    return self(x, u)

  def parameters(self):
    """Retrieve the parameters of the network.

    Returns:
      Iterator[torch.nn.Parameter]: An iterator over the parameters of the network.
    """
    return self._network.parameters()

  def train(self):
    """Sets the network to training mode.
    """
    self._network.train()

  def eval(self):
    """Sets the network to evaluation mode.
    """
    self._network.eval()

  def save(self, path):
    """Saves the current state of the network to a file.

    Args:
      path (str): The file path where the state dictionary and size will be saved.
    """
    data_to_save = {
        'state_dict': self._network.state_dict(),
        'size': self._size
    }
    torch.save(data_to_save, path)

  def load(self, path):
    """Loads the model state and size from a specified file path.

    Args:
      path (str): The file path from which to load the model data.
    """
    data_loaded = torch.load(path)
    self._size = data_loaded['size']
    self._network.load_state_dict(data_loaded['state_dict'])

  @property
  def size(self):
    """Returns the size of the Koopman matrix.

    Returns:
      int: The row/column size of the Koopman matrix.
    """
    return self._size
