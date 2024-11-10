import torch
import numpy as np
import numbers


class ODEDataSet(torch.utils.data.Dataset):

  def __init__(self, ode, flowmap, x_sample_func=torch.rand):
    self._ode = ode
    self._flowmap = flowmap
    self._generated = False
    self._x_sample_func = x_sample_func

  def generate_data(self, n_traj, traj_len, x_min, x_max, param, seed_x=11):
    if isinstance(x_min, numbers.Number):
      x_min = torch.ones((1, self._ode.dim)) * x_min
    if isinstance(x_max, numbers.Number):
      x_max = torch.ones((1, self._ode.dim)) * x_max
    x_min = x_min.expand(n_traj, self._ode.dim)
    x_max = x_max.expand(n_traj, self._ode.dim)
    torch.manual_seed(seed_x)
    x0 = self._x_sample_func(n_traj, self._ode.dim)
    x0 = x0 * (x_max - x_min) + x_min

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

  def __init__(self,
               ode,
               flowmap,
               x_sample_func=torch.rand,
               param_sample_func=torch.rand):
    super().__init__(ode, flowmap, x_sample_func)
    self._param_sample_func = param_sample_func

  def generate_data(self,
                    n_traj,
                    n_traj_per_param,
                    traj_len,
                    x_min,
                    x_max,
                    param_min,
                    param_max,
                    seed_x=11,
                    seed_param=22,
                    param_time_dependent=False):
    # generate x
    if isinstance(x_min, numbers.Number):
      x_min = torch.ones((1, self._ode.dim)) * x_min
    if isinstance(x_max, numbers.Number):
      x_max = torch.ones((1, self._ode.dim)) * x_max
    x_min = x_min.expand(n_traj, self._ode.dim)
    x_max = x_max.expand(n_traj, self._ode.dim)
    torch.manual_seed(seed_x)
    x0 = self._x_sample_func(n_traj, self._ode.dim)
    x0 = x0 * (x_max - x_min) + x_min

    # generate param
    if isinstance(param_min, numbers.Number):
      param_min = torch.ones((1, self._ode.param_dim)) * param_min
    if isinstance(param_max, numbers.Number):
      param_max = torch.ones((1, self._ode.param_dim)) * param_max
    param_min = param_min.expand(n_traj, self._ode.param_dim)
    param_max = param_max.expand(n_traj, self._ode.param_dim)
    torch.manual_seed(seed_param)

    def generate_param():
      param = self._param_sample_func(int(n_traj / n_traj_per_param),
                                      self._ode.param_dim)
      param = param.repeat_interleave(n_traj_per_param, dim=0)
      param = param * (param_max - param_min) + param_min
      return param

    data_x = [x0]
    param = generate_param()
    data_param = [param]
    for t in range(traj_len - 1):
      data_x.append(self._flowmap.step(self._ode, data_x[t], data_param[t]))
      if param_time_dependent:
        param = generate_param()
      data_param.append(param)

    # Reshape and transpose data_x for the correct format
    self._data_x = torch.cat(data_x, dim=0)

    # Repeat parameters for each trajectory length
    self._data_param = torch.cat(data_param, dim=0)

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
