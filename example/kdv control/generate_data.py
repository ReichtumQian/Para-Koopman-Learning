import PKoopmanDL as pkdl
import numpy as np
import matplotlib.pyplot as plt
import torch

pkdl.set_n_jobs(32)

# read the config file
config_file = "kdv.json"
solver = pkdl.ParamKoopmanDLSolverWrapper(config_file)

# set up the grid
Nx = 128
x_max = torch.pi
x_min = -torch.pi
x_step = (x_max - x_min) / (Nx - 1)
x_grid = torch.linspace(x_min, x_max, Nx).unsqueeze(0)


def sample_func(row_size, col_size):
  assert (col_size == Nx)
  x = x_grid.expand(row_size, Nx)
  # torch.manual_seed(0)
  uniform = torch.tensor([1.0, 1.0, 1.0])
  b = torch.distributions.Dirichlet(uniform).sample((row_size, ))
  b1 = b[:, 0].view(-1, 1)
  b2 = b[:, 1].view(-1, 1)
  b3 = b[:, 2].view(-1, 1)
  eta = b1 * torch.exp(-(x - torch.pi / 2)**2) - b2 * torch.sin(
      x / 2)**2 + b3 * torch.exp(-(x + torch.pi / 2)**2)
  return eta


def observable_func(x):
  mass = torch.sum(x, dim=1, keepdim=True) * x_step
  momentum = torch.sum(x**2, dim=1, keepdim=True) * x_step
  return torch.cat((mass, momentum), dim=1)


# set up the solver
solver.setup(observable_func, sample_func)
solver.save_dataset("data/kdv_dataset.pt")
