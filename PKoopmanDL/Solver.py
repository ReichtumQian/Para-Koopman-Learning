
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
    X = self._dictionary(data_x).t()
    Y = self._dictionary(labels).t()
    K = (Y @ X.t()) @ np.linalg.pinv(X @ X.t())
    return Koopman(K)

class EDMDDLSolver(EDMDSolver):

  def __init__(self, dictionary, reg, reg_final = 0.01):
    """ Initialize the EDMDDLSolver instance.
    
    Args:
        dictionary (TrainableDictionary): The dictionary used in the algorithm.
        regularizer (float): The regularization parameter used in the algorithm.
    """
    super().__init__(dictionary)
    self.__reg = reg
    self.__reg_final = reg_final

  def solve(self, dataset, n_epochs, batch_size, tol = 1e-8, lr = 1e-4):
    def compute_K(data_x, labels, reg):
      X = self._dictionary(data_x).t()
      Y = self._dictionary(labels).t()
      regularizer = torch.eye(self._dictionary.dim_output) * reg 
      K = (Y @ X.t()) @ torch.linalg.pinv(X @ X.t() + regularizer)
      return K.detach()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_func = torch.nn.MSELoss()
    opt = torch.optim.Adam(self._dictionary.parameters(), lr = lr)
    self._dictionary.train()
    pbar = tqdm(range(n_epochs), desc="Training")
    for _ in pbar:
      K = compute_K(dataset.data_x, dataset.labels, self.__reg)
      K = K.to(DEVICE)
      total_loss = 0
      num_samples = 0
      for __ in range(2):
        for data, labels in data_loader:
          opt.zero_grad()
          X = self._dictionary(data).to(DEVICE) @ K.t()
          Y = self._dictionary(labels).to(DEVICE)
          loss = loss_func(X, Y)
          loss.backward()
          opt.step()
          total_loss = loss.item() * data.size(0)
          num_samples += data.size(0)
      total_loss /= num_samples
      loss_str = f"{total_loss:.2e}"
      pbar.set_postfix(loss=loss_str)
      if total_loss < tol:
        break
    K = compute_K(dataset.data_x, dataset.labels, self.__reg_final)
    return Koopman(K)

          
class ParamKoopmanDLSolver:
  def __init__(self, dictionary):
    self.__dictionary = dictionary
  
  def solve(self, dataset, paramkoopman, n_epochs, batch_size, tol = 1e-6, lr_dic = 1e-4, lr_koop = 1e-4):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_func = torch.nn.MSELoss()
    opt_dictionary = torch.optim.Adam(self.__dictionary.parameters(), lr = lr_dic)
    opt_koopman = torch.optim.Adam(paramkoopman.parameters(), lr = lr_koop)
    pbar = tqdm(range(n_epochs), desc="Training")
    for _ in pbar:
      total_loss = 0
      for data_x, data_param, labels in data_loader:
        opt_dictionary.zero_grad()
        opt_koopman.zero_grad()
        X = self.__dictionary(data_x)
        X = paramkoopman(data_param, X).to(DEVICE)
        Y = self.__dictionary(labels).to(DEVICE)
        loss = loss_func(X, Y)
        opt_dictionary.step()
        opt_koopman.step()
        total_loss += loss.item() * data_x.size(0)
      total_loss = total_loss / len(dataset)
      loss_str = f"{total_loss:.2e}"
      pbar.set_postfix(loss=loss_str)
      if total_loss < tol:
        break
    return paramkoopman

    
    
    
    
    
    
    