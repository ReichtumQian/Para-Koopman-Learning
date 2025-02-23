from .Koopman import *
from .Device import DEVICE
from .Log import *
from tqdm import tqdm
import torch
import math


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
        dataset (KoopmanDataSet): The dataset containing state and label data.

    Returns:
        Koopman: The Koopman operator as a linear mapping function.
    """
    info_message("[EDMDSolver] Solving...")
    data_x = dataset.data_x
    labels = dataset.labels
    X = self._dictionary(data_x).t()
    Y = self._dictionary(labels).t()
    A = Y @ X.t()
    G = X @ X.t()
    K = A @ torch.linalg.pinv(G)
    return Koopman(K)


class EDMDDLSolver(EDMDSolver):

  def __init__(self, dictionary, reg, reg_final=0.01):
    """ Initialize the EDMDDLSolver instance.
    
    Args:
        dictionary (TrainableDictionary): The dictionary used in the algorithm.
        reg (float): The regularization parameter used in the algorithm.
        reg_final (float): The final regularization parameter used in the algorithm.
    """
    super().__init__(dictionary)
    self._reg = reg
    self._reg_final = reg_final

  def solve(self,
            dataset_train,
            dataset_val,
            n_epochs,
            batch_size,
            tol=1e-8,
            lr=1e-4):
    """ Solves the Koopman operator learning problem using the provided training and validation datasets.

    Args:
      dataset_train (KoopmanDataset): The training dataset containing input data and labels.
      dataset_val (KoopmanDataset): The validation dataset containing input data and labels.
      n_epochs (int): The number of epochs to train the model.
      batch_size (int): The size of each batch for training and validation.
      tol (float): The tolerance for early stopping based on training loss. Defaults to 1e-8.
      lr (float): The learning rate for the optimizer. Defaults to 1e-4.

    Returns:
      Koopman: The learned Koopman operator.
    """

    def compute_K(data_x, labels, reg):
      X = self._dictionary(data_x).t()
      Y = self._dictionary(labels).t()
      regularizer = torch.eye(self._dictionary.dim_output) * reg
      K = (Y @ X.t()) @ torch.linalg.pinv(X @ X.t() + regularizer)
      return K.detach()

    info_message("[EDMDDLSolver] Solving...")

    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=batch_size,
                                                 shuffle=False)
    loss_func = torch.nn.MSELoss()
    opt = torch.optim.Adam(self._dictionary.parameters(), lr=lr)
    pbar = tqdm(range(n_epochs), desc="Training")
    for _ in pbar:
      # Training
      self._dictionary.train()
      K = compute_K(dataset_train.dataset.data_x, dataset_train.dataset.labels,
                    self._reg)
      K = K.to(DEVICE)
      total_loss = 0
      num_samples = 0
      for __ in range(2):
        for data, labels in dataloader_train:
          opt.zero_grad()
          X = self._dictionary(data).to(DEVICE) @ K.t()
          Y = self._dictionary(labels).to(DEVICE)
          loss = loss_func(X, Y)
          loss.backward()
          opt.step()
          total_loss += loss.item() * data.size(0)
          num_samples += data.size(0)
      total_loss /= num_samples
      total_loss = math.sqrt(total_loss)
      loss_str = f"{total_loss:.2e}"
      # Validation
      self._dictionary.eval()
      K = compute_K(dataset_val.dataset.data_x, dataset_val.dataset.labels,
                    self._reg)
      K = K.to(DEVICE)
      val_loss = 0
      val_num_samples = 0
      with torch.no_grad():
        for data, labels in dataloader_val:
          X = self._dictionary(data).to(DEVICE) @ K.t()
          Y = self._dictionary(labels).to(DEVICE)
          val_loss += loss_func(X, Y).item() * data.size(0)
          val_num_samples += data.size(0)
      val_loss /= val_num_samples
      val_loss = math.sqrt(val_loss)
      val_loss_str = f"{val_loss:.2e}"
      pbar.set_postfix(train_loss=loss_str, val_loss=val_loss_str)
      if total_loss < tol:
        break
    K = compute_K(dataset_train.dataset.data_x, dataset_train.dataset.labels,
                  self._reg_final)
    return Koopman(K)


class ParamKoopmanDLSolver:

  def __init__(self, dictionary):
    """Initializes the KoopmanSolver with a given dictionary.

    Args:
      dictionary (TrainableDictionary): A dictionary containing configuration or parameters for the Koopman solver.
    """
    self._dictionary = dictionary

  def solve(self,
            dataset_train,
            dataset_val,
            paramkoopman,
            n_epochs,
            batch_size,
            tol=1e-6,
            lr_dic=1e-4,
            lr_koop=1e-4,
            weight=None):
    """Solves the ParamKoopmanDL problem using the provided datasets and parameters.

    Args:
      dataset_train (KoopmanDataset): The training dataset.
      dataset_val (KoopmanDataset): The validation dataset.
      paramkoopman (ParamKoopman): The parameterized Koopman operator model.
      n_epochs (int): The number of epochs to train the model.
      batch_size (int): The batch size for training and validation.
      tol (float): The tolerance for early stopping based on training loss. Default is 1e-6.
      lr_dic (float): The learning rate for the dictionary optimizer. Default is 1e-4.
      lr_koop (float): The learning rate for the Koopman optimizer. Default is 1e-4.

    Returns:
      ParamKoopman: The trained parameterized Koopman operator model.
    """
    info_message("[ParamKoopmanDLSolver] Solving...")
    # normalize the dataset
    if weight == None:
      weight = torch.ones(1, paramkoopman.size)
    else:
      assert weight.size(0) == 1
      assert weight.size(1) == paramkoopman.size

    def apply_weight(data):
      weight_tmp = weight.expand(data.size(0), -1)
      return data * weight_tmp

    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=batch_size,
                                                 shuffle=False)
    loss_func = torch.nn.MSELoss()
    opt_dictionary = torch.optim.Adam(self._dictionary.parameters(), lr=lr_dic)
    opt_koopman = torch.optim.Adam(paramkoopman.parameters(), lr=lr_koop)
    pbar = tqdm(range(n_epochs), desc="Training")
    for _ in pbar:
      # Training
      total_loss = 0
      self._dictionary.train()
      paramkoopman.train()
      for data_x, data_param, labels in dataloader_train:
        opt_dictionary.zero_grad()
        opt_koopman.zero_grad()
        X = self._dictionary(data_x)
        X = paramkoopman(X, data_param)
        X = apply_weight(X)
        X = X.to(DEVICE)
        Y = self._dictionary(labels)
        Y = apply_weight(Y)
        Y = Y.to(DEVICE)
        if debug_level():
          error = torch.max(torch.abs(X - Y))
          debug_message(f"Train Max Error: {error:.2e}")
        loss = loss_func(X, Y)
        loss.backward()
        opt_dictionary.step()
        opt_koopman.step()
        total_loss += loss.item() * data_x.size(0)
      total_loss = total_loss / len(dataset_train)
      total_loss = math.sqrt(total_loss)
      loss_str = f"{total_loss:.2e}"
      # Validation
      self._dictionary.eval()
      paramkoopman.eval()
      if len(dataloader_val) == 0:
        pbar.set_postfix(train_loss=loss_str)
        continue
      val_loss = 0
      val_num_samples = 0
      with torch.no_grad():
        for data_x, data_param, labels in dataloader_val:
          X = self._dictionary(data_x)
          X = paramkoopman(X, data_param).to(DEVICE)
          Y = self._dictionary(labels).to(DEVICE)
          val_loss += loss_func(X, Y).item() * data_x.size(0)
          val_num_samples += data_x.size(0)
      val_loss /= val_num_samples
      val_loss = math.sqrt(val_loss)
      val_loss_str = f"{val_loss:.2e}"
      pbar.set_postfix(train_loss=loss_str, val_loss=val_loss_str)
      if total_loss < tol:
        break
    return paramkoopman
