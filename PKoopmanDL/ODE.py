

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
    assert(len(x.shape) == 2)
    assert(len(u.shape) == 2)
    assert(x.shape[0] == u.shape[0])
    assert(x.shape[1] == self._dim)
    assert(u.shape[1] == self._param_dim)
    return self._rhs(x, u)

  @property
  def dim(self):
    return self._dim
  
  @property
  def param_dim(self):
    return self._param_dim

