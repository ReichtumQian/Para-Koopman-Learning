import torch
import numpy as np


class ODEDataSet(torch.utils.data.Dataset):

  def __init__(self, ode, flowmap):
    self._ode = ode
    self._flowmap = flowmap
    self._generated = False

  def generate_data(self,
                    n_traj,
                    traj_len,
                    x_min,
                    x_max,
                    param,
                    seed_x=11):
    np.random.seed(seed_x)
    x0 = np.random.uniform(low=x_min, high=x_max, size=(n_traj, self._ode.dim))

    data_x = [x0]
    for t in range(traj_len - 1):
      data_x.append(self._flowmap.step(self._ode, data_x[t], param))

    # Reshape and transpose data_x for the correct format
    self._data_x = np.array(data_x).reshape(n_traj * traj_len, self._ode.dim)
    self._labels = self._flowmap.step(
        self._ode, self._data_x, param)
    self._generated = True

  def __getitem__(self, idx):
    if (not self._generated):
      raise RuntimeError("The data has not been generated yet.")
    return self._data_x[idx], self._labels[idx]

  def __len__(self):
    if (not self._generated):
      raise RuntimeError("The data has not been generated yet.")
    return self._data_x.shape[0]

  @property
  def labels(self):
    if (not self._generated):
      raise RuntimeError("The data has not been generated yet.")
    return self._labels


  @property
  def data_x(self):
    if (not self._generated):
      raise RuntimeError("The data has not been generated yet.")
    return self._data_x


class ParamODEDataSet(ODEDataSet):


  def generate_data(self,
                    n_traj,
                    traj_len,
                    x_min,
                    x_max,
                    seed_x=11,
                    param_min=0,
                    param_max=0,
                    seed_param=22):
    np.random.seed(seed_x)
    x0 = np.random.uniform(low=x_min, high=x_max, size=(n_traj, self._ode.dim))

    np.random.seed(seed_param)
    param = np.random.uniform(low=param_min,
                              high=param_max,
                              size=(n_traj, self._ode.param_dim))

    data_x = [x0]
    for t in range(traj_len - 1):
      data_x.append(self._flowmap.step(self._ode, data_x[t], param))

    # Reshape and transpose data_x for the correct format
    self._data_x = np.array(data_x).reshape(n_traj * traj_len, self._ode.dim)

    # Repeat parameters for each trajectory length
    repeats_constant = traj_len * np.ones(shape=(n_traj, ), dtype=np.int32)
    self._data_param = np.repeat(param, repeats=repeats_constant, axis=0)

    self._labels = self._flowmap.step(
        self._ode, self._data_x, self._data_param)

    self._generated = True

  def __getitem__(self, idx):
    if (not self._generated):
      raise RuntimeError("The data has not been generated yet.")
    return self._data_x[idx], self._data_param[idx], self._labels[idx]
  
  
  @property
  def data_param(self):
    if (not self._generated):
      raise RuntimeError("The data has not been generated yet.")
    return self._data_param
  



