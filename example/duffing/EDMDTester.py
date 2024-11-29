# This helper class helps to find the closest EDMDSolver/EDMDDLSolver

import torch
import PKoopmanDL as pkdl


class EDMDTester:

  def __init__(self,
               config_file,
               num_param,
               observable_func,
               param_min,
               param_max,
               seed_param,
               is_edmd=True):
    if is_edmd:
      TargetSolver = pkdl.EDMDRBFSolverWrapper
    else:
      TargetSolver = pkdl.EDMDDLSolverWrapper
    tmp_solver = TargetSolver(config_file)
    param_dim = len(tmp_solver.param)
    self._solvers = []
    self._params = []
    self._observable_func = observable_func
    torch.manual_seed(seed_param)
    for _ in range(num_param):
      param = torch.rand(1, param_dim)
      param = param * (param_max - param_min) + param_min
      self._params.append(param)
      solver = TargetSolver(config_file)
      solver.param = param
      self._solvers.append(solver)

  def _find_closest(self, param):
    min_distance = float('inf')
    min_index = -1

    for i, tensor in enumerate(self._params):
      distance = torch.norm(tensor - param, 2)
      if distance < min_distance:
        min_distance = distance
        min_index = i

    return min_index

  def solve(self, param):
    index = self._find_closest(param)
    self._solvers[index].setup(self._observable_func)
    return self._solvers[index].solve(), self._solvers[index].dictionary
