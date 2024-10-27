import torch
import numpy as np


class ODEDataSet(torch.utils.data.Dataset):

  def __init__(self, ode, flowmap):
    self.__ode = ode
    self.__flowmap = flowmap
    self.__generated = False

  def generate_data(self,
                    n_traj,
                    traj_len,
                    x_min,
                    x_max,
                    param_min,
                    param_max,
                    seed_x=11,
                    seed_param=22):
    np.random.seed(seed_x)
    x0 = np.random.uniform(low=x_min, high=x_max, size=(n_traj, self.__ode.dim))

    np.random.seed(seed_param)
    param = np.random.uniform(low=param_min,
                              high=param_max,
                              size=(n_traj, self.__ode.param_dim))

    data_x = [x0]
    for t in range(traj_len - 1):
      data_x.append(self.__flowmap.step(self.__ode, data_x[t], param))

    # Reshape and transpose data_x for the correct format
    self.__data_x = np.array(data_x).reshape(n_traj * traj_len, self.__ode.dim)

    # Repeat parameters for each trajectory length
    repeats_constant = traj_len * np.ones(shape=(n_traj, ), dtype=np.int32)
    self.__data_param = np.repeat(param, repeats=repeats_constant, axis=0)

    self.__labels = self.__flowmap.step(
        self.__ode, self.__data_x, self.__data_param)

    self.__generated = True

  def __len__(self):
    if (not self.__generated):
      raise RuntimeError("The data has not been generated yet.")
    return self.__data_x.shape[0]

  def __getitem__(self, idx):
    if (not self.__generated):
      raise RuntimeError("The data has not been generated yet.")
    return self.__data_x[idx], self.__data_param[idx], self.__labels[idx]
  
  @property
  def data_x(self):
    if (not self.__generated):
      raise RuntimeError("The data has not been generated yet.")
    return self.__data_x
  
  @property
  def data_param(self):
    if (not self.__generated):
      raise RuntimeError("The data has not been generated yet.")
    return self.__data_param
  
  @property
  def labels(self):
    if (not self.__generated):
      raise RuntimeError("The data has not been generated yet.")
    return self.__labels



