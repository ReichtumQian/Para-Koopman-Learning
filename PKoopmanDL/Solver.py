
from .Koopman import *
import numpy as np

class EDMDSolver:

  def __init__(self, dictionary):
    """ Initialize the EDMDSolver instance.
    
    Args:
        dictionary (Dictionary): The dictionary used in the algorithm.
    """
    self._dictionary = dictionary

  def solve(self, dataset):
    """Applies the EDMD algorithm to compute the Koopman operator.

    Args:
        dataset (ODEDataSet): The dataset containing state and label data.

    Returns:
        Koopman: The Koopman operator as a linear mapping function.
    """
    data_x = dataset.data_x
    labels = dataset.labels
    X = self._dictionary(data_x).T
    Y = self._dictionary(labels).T
    K = (Y @ X.T) @ np.linalg.pinv(X @ X.T)
    K_func = lambda x: (K @ x.T).T
    return Koopman(K_func)

class EDMDDLSolver(EDMDSolver):

  def __init__(self, dictionary, regularizer):
    """ Initialize the EDMDDLSolver instance.
    
    Args:
        dictionary (TrainableDictionary): The dictionary used in the algorithm.
        regularizer (float): The regularization parameter used in the algorithm.
    """
    super().__init__(dictionary)
    self.__regularizer = regularizer

  def solve(self, dataset, n_epochs, batch_size, tol):
    def compute_K(data_x, labels):
      X = self._dictionary(data_x).T
      Y = self._dictionary(labels).T
      regularizer = np.eye(self._dictionary.dim_output) * self.__regularizer
      K = (Y @ X.T) @ np.linalg.pinv(X @ X.T + regularizer)
      K_func = lambda x: (K @ x.T).T
      return Koopman(K_func)
    

class ParamKoopmanDLSolver:
  def __init__(self, dictionary):
    self.__dictionary = dictionary

    
    
    
    
    
    
    