
import numpy as np
import torch

class Koopman:
  
  def __init__(self, func = None, K = None):
    """Initialize the Koopman instance.

    Args:
        func (tensor -> tensor): A mapping function representing the Koopman operator.
        K (tensor): The Koopman matrix.
    """
    if K is None and func is None:
      raise ValueError("Either K or func must be provided.")
    elif func is not None:
      self.__func = func
    elif K is not None:
      func = lambda x: (K @ x.t()).t()
      self.__func = func
  
  def __call__(self, x):
    """Apply the Koopman operator on the input `x`.

    Args:
        x (tensor): The input to apply the Koopman operator, expected to be of shape (N, N_psi).

    Returns:
        tensor : The output of the Koopman operator.
    """
    return self.__func(x)
  
  def predict(self, x0, dictionary, dim_nontrain, traj_len):
    y = []
    psi = dictionary(x0)
    y.append(psi[:, :dim_nontrain])
    for _ in range(traj_len - 1):
      psi = self(psi)
      y.append(psi[:, :dim_nontrain])
    return torch.stack(y, dim = 0).permute(1, 0, 2) # size: (N, traj_len, dim_nontrain)
  
class ParamKoopman:

  def __init__(self, size_K, network):
    """Initialize the ParamKoopman instance.

    Args:
        network ((tensor) -> tensor): Given the parameters `u`, returns the generated Koopman operator matrix.
    """
    self.__size = size_K
    self.__network = network

  def __call__(self, para, x):
    """Generate a Koopman operator based on the given parameters.

    Args:
        para (tensor): The parameter data, expected to be of shape (N, N_u).

    Returns:
        Koopman: The Koopman operator corresponding to the given parameters.
    """
    net_out = self.__network(para) 
    K = net_out.reshape(net_out.size(0), self.__size, self.__size)
    x = x.unsqueeze(2)
    result = torch.bmm(K, x).squeeze(2)
    return result

  def parameters(self):
    return self.__network.parameters()


