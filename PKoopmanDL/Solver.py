
from .Koopman import *
from .Device import DEVICE
from tqdm import tqdm
import numpy as np
import torch

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

  def solve(self, dataset, n_epochs, batch_size, tol, lr = 1e-3):
    def compute_K(data_x, labels):
      X = self._dictionary(data_x).T
      Y = self._dictionary(labels).T
      regularizer = np.eye(self._dictionary.dim_output) * self.__regularizer
      K = (Y @ X.T) @ np.linalg.pinv(X @ X.T + regularizer)
      # K_func = lambda x: (K @ x.T).T
      # return Koopman(K_func)
      return torch.from_numpy(K).to(DEVICE).detach()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_func = torch.nn.MSELoss()
    opt = torch.optim.Adam(self._dictionary.parameters(), lr = lr)
    self._dictionary.train()
    with tqdm(range(n_epochs), desc="Training") as pbar:
      for _ in pbar:
        K = compute_K(dataset.data_x, dataset.labels)
        for data, labels in data_loader:
          data = data.to(DEVICE)
          labels = labels.to(DEVICE)
          opt.zero_grad()
          X = K.t() @ self._dictionary(data)
          

    

class ParamKoopmanDLSolver:
  def __init__(self, dictionary):
    self.__dictionary = dictionary

    
    
    
    
    
    
    