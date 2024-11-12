import numpy as np
import torch
from .Dynamics import TransitionFunction


class Koopman(TransitionFunction):

  def __init__(self, K):
    """Initialize the Koopman instance.

    Args:
        func (tensor -> tensor): A mapping function representing the Koopman operator.
        K (tensor): The Koopman matrix.
    """
    func = lambda x: (K @ x.t()).t()
    self._K = K
    self._func = func

  def __call__(self, x, u):
    return self._func(x)

  def step(self, x, u):
    return self(x, u)

  def save(self, path):
    torch.save(self._K, path)

  def load(self, path):
    K = torch.load(path)
    self.__init__(K)


class ParamKoopman(TransitionFunction):

  def __init__(self, size_K, network):
    """Initialize the ParamKoopman instance.

    Args:
        network ((tensor) -> tensor): Given the parameters `u`, returns the generated Koopman operator matrix.
    """
    self._size = size_K
    self._network = network

  def __call__(self, x, u):
    """Generate a Koopman operator based on the given parameters.

    Args:
        para (tensor): The parameter data, expected to be of shape (N, N_u).

    Returns:
        Koopman: The Koopman operator corresponding to the given parameters.
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
    return self(x, u)

  def parameters(self):
    return self._network.parameters()

  def train(self):
    self._network.train()

  def eval(self):
    self._network.eval()

  def save(self, path):
    data_to_save = {
        'state_dict': self._network.state_dict(),
        'size': self._size
    }
    torch.save(data_to_save, path)

  def load(self, path):
    data_loaded = torch.load(path)
    self._size = data_loaded['size']
    self._network.load_state_dict(data_loaded['state_dict'])
