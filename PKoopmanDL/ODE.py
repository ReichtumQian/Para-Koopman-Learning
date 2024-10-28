
import torch

class AbstractODE:

  def __init__(self, dim, param_dim, rhs):
    """
    Initialize the AbstractODE instance.

    Args:
        dim (int): The dimension of the ODE.
        param_dim (int): The dimension of the parameter.
        rhs ((tensor, tensor) -> tensor): The right-hand side function of the ODE.
    """
    self._dim = dim
    self._param_dim = param_dim
    self._rhs = rhs
    
  def rhs(self, x, u):
    """
    Computes the right-hand side function using the inputs `(x, u)`. If `_param_dim=0` then `u` will be ignored.

    Args:
        x (tensor): The state of the system.
        u (tensor): The parameter of the system.

    Returns:
        tensor: The right-hand side of the ODE.
    """
    return self._rhs(x, u)

  @property
  def dim(self):
    return self._dim
  
  @property
  def param_dim(self):
    return self._param_dim

class DuffingOscillator(AbstractODE):

  def __init__(self):
    dim = 2
    param_dim = 3
    def rhs(x, u):
      # u = (delta, beta, alpha)
      param = u
      data_size = x.size(0)
      if u.size(0) == 1:
        param = u.expand(data_size, param_dim)
      result = torch.stack(
        (x[:, 1], - param[:, 0] * x[:, 1] - x[:, 0] * (param[:, 1] + param[:, 2] * x[:, 0]**2)), axis = 1
      )
      return result
    super().__init__(dim, param_dim, rhs)
        
