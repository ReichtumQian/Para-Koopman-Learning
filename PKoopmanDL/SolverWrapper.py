
import json
import torch
import PKoopmanDL as pkdl

class SolverWrapper:
  
  def __init__(self, nontrain_func, json_file):
    with open(json_file) as f:
      self._data = json.load(f)
    self.solver_type = self._data['solver_type']
    self.equ_type = self._data['equ_type']
    self._init_flowmap()
    self._init_dataset()
    self._init_dictionary(nontrain_func)
    self._init_solver()

  def _init_flowmap(self):
    self.dt = self._data['flowmap']['dt']
    self.t_step = self._data['flowmap']['t_step']
    self.flowmap = pkdl.ForwardEuler(self.t_step, self.dt)

  def _init_dataset(self):
    if self.equ_type == "Duffing":
      self.ode = pkdl.DuffingOscillator()
    elif self.equ_type == "vdp":
      self.ode = pkdl.VanDerPolOscillator()
    else:
      raise ValueError("Unknown equation type")
    self.n_traj = self._data['dataset']['n_traj']
    self.traj_len = self._data['dataset']['traj_len']
    self.x_min = self._data['dataset']['x_min']
    self.x_max = self._data['dataset']['x_max']
    self.seed_x = self._data['dataset']['seed_x']
    if isinstance(self.x_min, list):
      self.x_min = torch.tensor(self.x_min).unsqueeze(0)
    if isinstance(self.x_max, list):
      self.x_max = torch.tensor(self.x_max).unsqueeze(0)
    
    # deal with param
    if self.solver_type == 'paramkoopman':
      self.param_min = self._data['dataset']['param_min']
      self.param_max = self._data['dataset']['param_max']
      if isinstance(self.param_min, list):
        self.param_min = torch.tensor(self.param_min).unsqueeze(0)
      if isinstance(self.param_max, list):
        self.param_max = torch.tensor(self.param_max).unsqueeze(0)
    elif self.solver_type == 'EDMD' or 'EDMDDL':
      self.param = self._data['dataset']['param']
      self.param = torch.tensor(self.param).unsqueeze(0)
    
    # generate dataset
    if self.solver_type == 'paramkoopman':
      self.seed_param = self._data['dataset']['seed_param']
      self.n_traj_per_param = self._data['dataset']['n_traj_per_param']
      self.train_ratio = self._data['dataset']['train_ratio']
      self.dataset = pkdl.ParamODEDataSet(self.ode, self.flowmap)
      self.dataset.generate_data(self.n_traj, self.n_traj_per_param, self.traj_len, self.x_min, self.x_max, self.param_min, self.param_max, self.seed_x, self.seed_param)
      self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [int(self.train_ratio * len(self.dataset)), len(self.dataset) - int(self.train_ratio * len(self.dataset))])
    elif self.solver_type == "EDMDDL":
      self.train_ratio = self._data['dataset']['train_ratio']
      self.dataset = pkdl.ODEDataSet(self.ode, self.flowmap)
      self.dataset.generate_data(self.n_traj, self.traj_len, self.x_min, self.x_max, self.param, self.seed_x)
      self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [int(self.train_ratio * len(self.dataset)), len(self.dataset) - int(self.train_ratio * len(self.dataset))])
    elif self.solver_type == 'EDMD-RBF':
      self.dataset = pkdl.ODEDataSet(self.ode, self.flowmap)
      self.dataset.generate_data(self.n_traj, self.traj_len, self.x_min, self.x_max, self.param, self.seed_x)
    else:
      raise ValueError('solver_type must be paramkoopman, EDMD-RBF or EDMDDL')
  
  def _init_dictionary(self, nontrain_func):
    self.dim_output = self._data['dictionary']['dim_output']
    self.dim_nontrain = self._data['dictionary']['dim_nontrain']
    if self.solver_type == "EDMD-RBF":
      reg = self._data['dictionary']['reg']
      self.dictionary = pkdl.RBFDictionary(self.dataset.data_x, nontrain_func, self.ode.dim, self.dim_output, self.dim_nontrain, reg)
    elif self.solver_type == "EDMDDL":
      self.dic_layer_sizes = self._data['dictionary']['dic_layer_sizes']
      network = pkdl.FullConnResNet(self.ode.dim, self.dim_output - self.dim_nontrain, self.dic_layer_sizes)
      self.dictionary = pkdl.TrainableDictionary(network, nontrain_func, self.ode.dim, self.dim_output, self.dim_nontrain)
    elif self.solver_type == "paramkoopman":
      self.dic_layer_sizes = self._data['dictionary']['dic_layer_sizes']
      network = pkdl.FullConnResNet(self.ode.dim, self.dim_output - self.dim_nontrain, self.dic_layer_sizes)
      self.dictionary = pkdl.TrainableDictionary(network, nontrain_func, self.ode.dim, self.dim_output, self.dim_nontrain)
    else:
      raise ValueError('solver_type must be paramkoopman, EDMD-RBF or EDMDDL')
    
  def _init_solver(self):
    if self.solver_type == "EDMD-RBF":
      self.solver = pkdl.EDMDSolver(self.dictionary)
    elif self.solver_type == "EDMDDL":
      self.reg = self._data['solver']['reg']
      self.reg_final = self._data['solver']['reg_final']
      self.solver = pkdl.EDMDDLSolver(self.dictionary, self.reg, self.reg_final)
    elif self.solver_type == "paramkoopman":
      self.solver = pkdl.ParamKoopmanDLSolver(self.dictionary)
    else:
      raise ValueError('solver_type must be paramkoopman, EDMD-RBF or EDMDDL')
  
  def solve(self):
    if self.solver_type == "EDMD-RBF":
      return self.solver.solve(self.dataset)
    n_epochs = self._data['solver']['n_epochs']
    batch_size = self._data['solver']['batch_size']
    tol = self._data['solver']['tol']
    dic_lr = self._data['solver']['dic_lr']
    if self.solver_type == "EDMDDL":
      return self.solver.solve(self.train_dataset, self.val_dataset, n_epochs, batch_size, tol, dic_lr)
    elif self.solver_type == "paramkoopman":
      self.koopman_layer_sizes = self._data['solver']['koopman_layer_sizes']
      koopman_lr = self._data['solver']['koopman_lr']
      network = pkdl.FullConnNet(self.ode.param_dim, self.dim_output**2, self.koopman_layer_sizes)
      PK = pkdl.ParamKoopman(self.dim_output, network)
      return self.solver.solve(self.train_dataset, self.val_dataset, PK, n_epochs, batch_size, tol, dic_lr, koopman_lr)
    else:
      raise ValueError('solver_type must be paramkoopman, EDMD-RBF or EDMDDL')
  
      

      
