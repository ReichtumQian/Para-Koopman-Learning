import torch
import scipy
import numpy as np
from .Factory import *


class AbstractODE:

  def __init__(self, dim, param_dim, rhs):
    self._dim = dim
    self._param_dim = param_dim
    self._rhs = rhs

  def rhs(self, x, u):
    return self._rhs(x, u)

  @property
  def dim(self):
    return self._dim

  @property
  def param_dim(self):
    return self._param_dim


class DuffingOscillator(AbstractODE):

  def __init__(self):
    dim = 2
    param_dim = 3

    def rhs(x, u):
      # u = (delta, beta, alpha)
      param = u
      data_size = x.size(0)
      if u.size(0) == 1:
        param = u.expand(data_size, param_dim)
      result = torch.stack((x[:, 1], -param[:, 0] * x[:, 1] - x[:, 0] *
                            (param[:, 1] + param[:, 2] * x[:, 0]**2)),
                           dim=1)
      if torch.isnan(result).any() or torch.isinf(result).any():
        print(torch.nonzero(torch.isnan(result)))
        print(torch.nonzero(torch.isinf(result)))
        raise ValueError("NaN or Inf detected in the result.")
      return result

    super().__init__(dim, param_dim, rhs)


class VanderPolMathieu(AbstractODE):

  def __init__(self):
    dim = 2
    param_dim = 2
    k1 = 2
    k2 = 2
    k3 = 1
    w0 = 1

    def rhs(x, u):
      # u = (mu, u)
      param = u
      data_size = x.size(0)
      if u.size(0) == 1:
        param = u.expand(data_size, param_dim)
      y1 = x[:, 1]
      y2 = (k1 - k2 * x[:, 0]**2) * x[:, 1] - (
          w0**2 + 2 * param[:, 0] * param[:, 1]**2 -
          param[:, 0]) * x[:, 0] + k3 * param[:, 1]
      result = torch.stack((y1, y2), dim=1)
      if torch.isnan(result).any() or torch.isinf(result).any():
        print(torch.nonzero(torch.isnan(result)))
        print(torch.nonzero(torch.isinf(result)))
        raise ValueError("NaN or Inf detected in the result.")
      return result

    super().__init__(dim, param_dim, rhs)


class FitzHughNagumo(AbstractODE):

  def __init__(self):
    Nx = 10  # in total Nx - 1 intervals
    x_max = 10
    x_min = -10
    x_step = (x_max - x_min) / (Nx - 1)
    x_grid = torch.linspace(x_min, x_max, Nx).unsqueeze(0)

    # A PDE is equivalent to (Nx * number of dependent variables) ODEs
    dim = 2 * Nx
    param_dim = 1
    delta = 4
    epsilon = 0.03
    a0 = -0.03
    a1 = 2
    k1 = -5
    k2 = 0
    k3 = 5

    def rhs(vw, u):
      param = u
      data_size = vw.size(0)
      if u.size(0) == 1:
        param = u.expand(data_size, param_dim)
      v = vw[:, :Nx]
      w = vw[:, Nx:]
      x = x_grid.expand(data_size, Nx)
      # zero Neumann boundary conditions
      v_minus_1 = v[:, 1].unsqueeze(1)
      v_N_plus_1 = v[:, -2].unsqueeze(1)
      v_ghost = torch.cat((v_minus_1, v, v_N_plus_1), dim=1)
      v_xx = (v_ghost[:, 2:] - 2 * v + v_ghost[:, :-2]) / x_step**2

      w_minus_1 = w[:, 1].unsqueeze(1)
      w_N_plus_1 = w[:, -2].unsqueeze(1)
      w_ghost = torch.cat((w_minus_1, w, w_N_plus_1), dim=1)
      w_xx = (w_ghost[:, 2:] - 2 * w + w_ghost[:, :-2]) / x_step**2

      param_term = param * 1e3 * (torch.exp(-(x - k1)**2 / 2) + torch.exp(
          -(x - k2)**2 / 2) + torch.exp(-(x - k3)**2 / 2))

      v_t = v_xx + v - v**3 - w + param_term
      w_t = delta * w_xx + epsilon * (v - a1 * w - a0)
      result = torch.cat((v_t, w_t), dim=1)
      if torch.isnan(result).any() or torch.isinf(result).any():
        print(torch.nonzero(torch.isnan(result)))
        print(torch.nonzero(torch.isinf(result)))
        raise ValueError("NaN or Inf detected in the result.")
      return result

    super().__init__(dim, param_dim, rhs)


class KortewegDeVries(AbstractODE):

  def __init__(self):
    x_min = -torch.pi
    x_max = torch.pi
    Nx = 128  # in total 127 intervals
    # x_step = (x_max - x_min) / (Nx - 1)
    x_grid = torch.linspace(x_min, x_max, Nx).unsqueeze(0)
    period = 2 * torch.pi

    dim = Nx
    param_dim = 3
    c1 = -torch.pi / 2
    c2 = 0
    c3 = torch.pi / 2
    v1 = lambda x: torch.exp(-25 * (x - c1)**2)
    v2 = lambda x: torch.exp(-25 * (x - c2)**2)
    v3 = lambda x: torch.exp(-25 * (x - c3)**2)

    def rhs(y, u):
      param = u
      data_size = y.size(0)
      if u.size(0) == 1:
        param = u.expand(data_size, param_dim)
      x = x_grid.expand(data_size, Nx)
      # left side
      yx = []
      yxxx = []
      for i in range(data_size):
        yx.append(scipy.fftpack.diff(y[i, :], period=period))
        yxxx.append(scipy.fftpack.diff(y[i, :], period=period, order=3))
      yx = torch.from_numpy(np.array(yx))
      yxxx = torch.from_numpy(np.array(yxxx))
      rhs1 = -y * yx - yxxx
      # right side
      param1 = param[:, 0].view(-1, 1)
      param2 = param[:, 1].view(-1, 1)
      param3 = param[:, 2].view(-1, 1)
      # rhs2 = v1(x) * torch.sin(torch.pi * param1) + v2(x) * torch.sin(
      #     torch.pi * param2) + v3(x) * torch.sin(torch.pi * param3)
      rhs2 = v1(x) * param1 + v2(x) * param2 + v3(x) * param3
      result = rhs1 + rhs2
      if torch.isnan(result).any() or torch.isinf(result).any():
        print(torch.nonzero(torch.isnan(result)))
        print(torch.nonzero(torch.isinf(result)))
        raise ValueError("NaN or Inf detected in the result.")
      return result.float()

    super().__init__(dim, param_dim, rhs)


# Factory
ODEFACTORY = Factory()
ODEFACTORY.register("Duffing", DuffingOscillator)
ODEFACTORY.register("vdpm", VanderPolMathieu)
ODEFACTORY.register("fhn", FitzHughNagumo)
ODEFACTORY.register("kdv", KortewegDeVries)


def registerODE(name, ode):
  ODEFACTORY.register(name, ode)
