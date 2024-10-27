

class Koopman:
  
  def __init__(self, func):
    """Initialize the Koopman instance.

    Args:
        func (ndarray -> ndarray): A mapping function representing the Koopman operator.
    """
    self.__func = func
  
  def __call__(self, x):
    """Apply the Koopman operator on the input `x`.

    Args:
        x (ndarray): The input to apply the Koopman operator, expected to be of shape (N, N_psi).

    Returns:
        ndarray: The output of the Koopman operator.
    """
    return self.__func(x)
  
class ParamKoopman:

  def __init__(self, func):
    """Initialize the ParamKoopman instance.

    Args:
        func ((ndarray, ndarray) -> ndarray): A mapping function representing the parametric Koopman operator.
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




