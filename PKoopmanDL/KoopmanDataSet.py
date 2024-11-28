import torch
import numpy as np
import numbers
import pickle
from .Log import *
from tqdm import tqdm


class KoopmanDataSet(torch.utils.data.Dataset):

  def __init__(self, dynamics, x_sample_func=torch.rand):
    self._dynamics = dynamics
    self._generated = False
    self._x_sample_func = x_sample_func

  def generate_data(self, n_traj, traj_len, x_min, x_max, param, seed_x=11):
    dim = self._dynamics.dim
    if isinstance(x_min, numbers.Number):
      x_min = torch.ones((1, dim)) * x_min
    if isinstance(x_max, numbers.Number):
      x_max = torch.ones((1, dim)) * x_max
    x_min = x_min.expand(n_traj, dim)
    x_max = x_max.expand(n_traj, dim)
    torch.manual_seed(seed_x)
    x0 = self._x_sample_func(n_traj, dim)
    x0 = x0 * (x_max - x_min) + x_min

    data_x = [x0]
    for t in range(traj_len - 1):
      data_x.append(self._dynamics.step(data_x[t], param))

    # Reshape and transpose data_x for the correct format
    self._data_x = torch.cat(data_x, dim=0)
    self._labels = self._dynamics.step(self._data_x, param)
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

  def save(self, file):
    if not self._generated:
      raise RuntimeError("The data has not been generated yet.")
    with open(file, 'wb') as f:
      pickle.dump({'data_x': self._data_x, 'labels': self._labels}, f)

  def load(self, file):
    with open(file, 'rb') as f:
      data = pickle.load(f)
    self._data_x = data['data_x']
    self._labels = data['labels']
    self._generated = True


class ParamKoopmanDataSet(KoopmanDataSet):

  def __init__(self,
               dynamics,
               x_sample_func=torch.rand,
               param_sample_func=torch.rand):
    super().__init__(dynamics, x_sample_func)
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
    dim = self._dynamics.dim
    param_dim = self._dynamics.param_dim
    info_message("[ParamKoopmanDataSet] Start generating data...")
    # generate x
    if isinstance(x_min, numbers.Number):
      x_min = torch.ones((1, dim)) * x_min
    if isinstance(x_max, numbers.Number):
      x_max = torch.ones((1, dim)) * x_max
    x_min = x_min.expand(n_traj, dim)
    x_max = x_max.expand(n_traj, dim)
    torch.manual_seed(seed_x)
    x0 = self._x_sample_func(n_traj, dim)
    x0 = x0 * (x_max - x_min) + x_min

    # generate param
    if isinstance(param_min, numbers.Number):
      param_min = torch.ones((1, param_dim)) * param_min
    if isinstance(param_max, numbers.Number):
      param_max = torch.ones((1, param_dim)) * param_max
    param_min = param_min.expand(n_traj, param_dim)
    param_max = param_max.expand(n_traj, param_dim)
    torch.manual_seed(seed_param)

    def generate_param():
      param = self._param_sample_func(int(n_traj / n_traj_per_param), param_dim)
      param = param.repeat_interleave(n_traj_per_param, dim=0)
      param = param * (param_max - param_min) + param_min
      return param

    data_x = [x0]
    param = generate_param()
    data_param = [param]
    info_message("[ParamKoopmanDataSet] Start generating trajectories...")
    for t in tqdm(range(traj_len - 1), desc="Generating trajectories"):
      data_x.append(self._dynamics.step(data_x[t], data_param[t]))
      if param_time_dependent:
        param = generate_param()
      data_param.append(param)

    # Reshape and transpose data_x for the correct format
    self._data_x = torch.cat(data_x, dim=0)

    # Repeat parameters for each trajectory length
    self._data_param = torch.cat(data_param, dim=0)

    info_message("[ParamKoopmanDataSet] Start generating labels...")
    self._labels = self._dynamics.step(self._data_x, self._data_param)

    self._generated = True
    info_message("[ParamKoopmanDataSet] Data generated.")

  def __getitem__(self, idx):
    if (not self._generated):
      raise RuntimeError("The data has not been generated yet.")
    return self._data_x[idx], self._data_param[idx], self._labels[idx]

  @property
  def data_param(self):
    if (not self._generated):
      raise RuntimeError("The data has not been generated yet.")
    return self._data_param

  def save(self, file):
    if not self._generated:
      raise RuntimeError("The data has not been generated yet.")
    with open(file, 'wb') as f:
      pickle.dump(
          {
              'data_x': self._data_x,
              'labels': self._labels,
              'data_param': self._data_param
          }, f)

  def load(self, file):
    with open(file, 'rb') as f:
      data = pickle.load(f)
    self._data_x = data['data_x']
    self._labels = data['labels']
    self._data_param = data['data_param']
    self._generated = True
