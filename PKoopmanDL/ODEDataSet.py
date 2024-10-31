import torch
import numpy as np
import numbers

class ODEDataSet(torch.utils.data.Dataset):

  def __init__(self, ode, flowmap):
    self._ode = ode
    self._flowmap = flowmap
    self._generated = False

  def generate_data(self, n_traj, traj_len, x_min, x_max, param, seed_x=11):
    if isinstance(x_min, numbers.Number):
      x_min = np.ones((1, self._ode.dim)) * x_min
    if isinstance(x_max, numbers.Number):
      x_max = np.ones((1, self._ode.dim)) * x_max
    if isinstance(x_min, torch.Tensor):
      x_min = x_min.numpy()
    if isinstance(x_max, torch.Tensor):
      x_max = x_max.numpy()
    x_min = np.broadcast_to(x_min, (n_traj, self._ode.dim))
    x_max = np.broadcast_to(x_max, (n_traj, self._ode.dim))
    np.random.seed(seed_x)
    x0 = np.random.uniform(low=0, high=1, size=(n_traj, self._ode.dim))
    x0 = x0 * (x_max - x_min) + x_min
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
                    n_traj_per_param,
                    traj_len,
                    x_min,
                    x_max,
                    param_min,
                    param_max,
                    seed_x=11,
                    seed_param=22):
    # generate x
    if isinstance(x_min, numbers.Number):
      x_min = np.ones((1, self._ode.dim)) * x_min
    if isinstance(x_max, numbers.Number):
      x_max = np.ones((1, self._ode.dim)) * x_max
    if isinstance(x_min, torch.Tensor):
      x_min = x_min.numpy()
    if isinstance(x_max, torch.Tensor):
      x_max = x_max.numpy()
    x_min = np.broadcast_to(x_min, (n_traj, self._ode.dim))
    x_max = np.broadcast_to(x_max, (n_traj, self._ode.dim))
    np.random.seed(seed_x)
    x0 = np.random.uniform(low=0, high=1, size=(n_traj, self._ode.dim))
    x0 = x0 * (x_max - x_min) + x_min
    x0 = torch.from_numpy(x0).to(dtype=torch.float32).detach()

    # generate param
    if isinstance(param_min, numbers.Number):
      param_min = np.ones((1, self._ode.param_dim)) * param_min
    if isinstance(param_max, numbers.Number):
      param_max = np.ones((1, self._ode.param_dim)) * param_max
    if isinstance(param_min, torch.Tensor):
      param_min = param_min.numpy()
    if isinstance(param_max, torch.Tensor):
      param_max = param_max.numpy()
    param_min = np.broadcast_to(param_min, (n_traj, self._ode.param_dim))
    param_max = np.broadcast_to(param_max, (n_traj, self._ode.param_dim))
    np.random.seed(seed_param)
    param = np.random.uniform(low=0,
                              high=1,
                              size=(int(n_traj/n_traj_per_param), self._ode.param_dim))
    param = np.repeat(param, n_traj_per_param, axis=0)
    param = param * (param_max - param_min) + param_min
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
