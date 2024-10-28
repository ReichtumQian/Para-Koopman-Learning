import torch
import numpy as np


class ODEDataSet(torch.utils.data.Dataset):

  def __init__(self, ode, flowmap):
    self._ode = ode
    self._flowmap = flowmap
    self._generated = False

  def generate_data(self, n_traj, traj_len, x_min, x_max, param, seed_x=11):
    np.random.seed(seed_x)
    x0 = np.random.uniform(low=x_min, high=x_max, size=(n_traj, self._ode.dim))
    x0 = torch.from_numpy(x0).to(dtype=torch.float32).detach()

    data_x = [x0]
    for t in range(traj_len - 1):
      data_x.append(self._flowmap.step(self._ode, data_x[t], param))

    # Reshape and transpose data_x for the correct format
    self._data_x = torch.cat(data_x, dim=0)
    self._labels = self._flowmap.step(self._ode, self._data_x, param)
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
                    param_min,
                    param_max,
                    seed_x=11,
                    seed_param=22):
    np.random.seed(seed_x)
    x0 = np.random.uniform(low=x_min, high=x_max, size=(n_traj, self._ode.dim))
    x0 = torch.from_numpy(x0).to(dtype=torch.float32).detach()

    np.random.seed(seed_param)
    param = np.random.uniform(low=param_min,
                              high=param_max,
                              size=(n_traj, self._ode.param_dim))
    param = torch.from_numpy(param).to(dtype=torch.float32).detach()

    data_x = [x0]
    for t in range(traj_len - 1):
      data_x.append(self._flowmap.step(self._ode, data_x[t], param))

    # Reshape and transpose data_x for the correct format
    self._data_x = torch.cat(data_x, dim=0)

    # Repeat parameters for each trajectory length
    repeats_constant = traj_len * torch.ones((n_traj, ), dtype=torch.int32)
    self._data_param = param.repeat_interleave(repeats_constant, dim=0)

    self._labels = self._flowmap.step(self._ode, self._data_x, self._data_param)

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
