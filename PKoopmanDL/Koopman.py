
import numpy as np
import torch

class Koopman:
  
  def __init__(self, func):
    """Initialize the Koopman instance.

    Args:
        func (tensor -> tensor): A mapping function representing the Koopman operator.
    """
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

  def __init__(self, func):
    """Initialize the ParamKoopman instance.

    Args:
        func ((tensor, tensor) -> tensor): A mapping function representing the parametric Koopman operator.
    """
    self.__func = func

  def __call__(self, x, para):
    """Apply the parametric Koopman operator on the inputs `x` and `para`.

    Args:
        x (ndarray): The input data, expected to be of shape (N, N_psi).
        para (ndarray): The parameter data, expected to be of shape (N, N_u).

    Returns:
        ndarray: The result of applying the parametric Koopman operator, with the same shape as `x`.
    """
    return self.__func(x, para)




