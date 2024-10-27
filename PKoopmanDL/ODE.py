
import numpy as np

class AbstractODE:

  def __init__(self, dim, param_dim, rhs):
    """
    Initialize the AbstractODE instance.

    Args:
        dim (int): The dimension of the ODE.
        param_dim (int): The dimension of the parameter.
        rhs ((ndarray, ndarray) -> ndarray): The right-hand side function of the ODE.
    """
    self._dim = dim
    self._param_dim = param_dim
    self._rhs = rhs
    
  def rhs(self, x, u):
    """
    Computes the right-hand side function using the inputs `(x, u)`. If `_param_dim=0` then `u` will be ignored.

    Args:
        x (ndarray): The state of the system.
        u (ndarray): The parameter of the system.

    Returns:
        ndarray: The right-hand side of the ODE.
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
      data_size = x.shape[0]
      if u.shape[0] == 1:
        param = np.broadcast_to(u, (data_size, param_dim))
      result = np.stack(
        (x[:, 1], - param[:, 0] * x[:, 1] - x[:, 0] * (param[:, 1] + param[:, 2] * x[:, 0]**2)), axis = 1
      )
      return result
    super().__init__(dim, param_dim, rhs)
        
