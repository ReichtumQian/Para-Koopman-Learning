import json
import torch
from .Factory import *
from .Dynamics import *
from .ODE import *
from .KoopmanDataSet import *
from .Dictionary import *
from .KoopmanSolver import *
from .Net import *
from .ODESolver import *


class SolverWrapper:

  def __init__(self, config_file):
    """ Initializes the SolverWrapper instance by loading configuration data from a JSON file.

    Args:
      config_file (str): The path to the configuration file in JSON format.

    This method performs the following actions:

    - Loads the configuration data from the specified JSON file.
    - Calls internal methods to read and process various configuration sections:
      - ODE configuration
      - ODE solver configuration
      - Dataset configuration
      - Dictionary configuration
      - Solver configuration
    """
    with open(config_file) as f:
      self._data = json.load(f)
    self._read_ode_config()
    self._read_ode_solver_config()
    self._read_dataset_config()
    self._read_dictionary_config()
    self._read_solver_config()

  def setup(self,
            observable_func,
            x_sample_func=torch.rand,
            param_sample_func=torch.rand):
    """
    Initializes the solver setup by configuring the ODE, dataset, and dictionary.

    Args:
      observable_func (torch.Tensor -> torch.Tensor): A function that defines the observables for the system.
      x_sample_func (callable, optional): A function to sample initial conditions. Defaults to torch.rand.
      param_sample_func (callable, optional): A function to sample parameters. Defaults to torch.rand.

    This method performs the following initializations:

    - Initializes the ordinary differential equation (ODE).
    - Sets up the ODE solver.
    - Initializes the dataset using the provided sampling functions.
    - Configures the dictionary with the given observable function.
    - Prepares the solver for execution.
    """
    self._init_ode()
    self._init_ode_solver()
    self._init_dataset(x_sample_func, param_sample_func)
    self._init_dictionary(observable_func)
    self._init_solver()

  def solve(self):
    """Placeholder method for solving a problem or executing a specific task.

    This method is intended to be overridden by subclasses to provide
    a concrete implementation of the solving logic. By default, it raises
    a NotImplementedError to indicate that the method has not been implemented.

    Returns:
      NotImplementedError: Indicates that the method needs to be implemented in a subclass.
    """
    return NotImplementedError

  def _read_ode_config(self):
    self.equ_type = self._data['equ_type']

  def _read_ode_solver_config(self):
    self.ode_solver_type = self._data['ode_solver']['type']
    self.dt = self._data['ode_solver']['dt']
    self.t_step = self._data['ode_solver']['t_step']

  def _read_dataset_config(self):
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
    if self._use_param_dataset:
      self.param_min = self._data['dataset']['param_min']
      self.param_max = self._data['dataset']['param_max']
      self.seed_param = self._data['dataset']['seed_param']
      self.n_traj_per_param = self._data['dataset']['n_traj_per_param']
      self.param_time_dependent = self._data['dataset'].get(
          'param_time_dependent', False)
      if isinstance(self.param_min, list):
        self.param_min = torch.tensor(self.param_min).unsqueeze(0)
      if isinstance(self.param_max, list):
        self.param_max = torch.tensor(self.param_max).unsqueeze(0)
    else:
      self.param = self._data['dataset']['param']
      self.param = torch.tensor(self.param).unsqueeze(0)

  def _read_dictionary_config(self):
    self.dim_output = self._data['dictionary']['dim_output']

  def _read_solver_config(self):
    # do nothing
    return

  def _init_ode(self):
    self.ode = ODEFACTORY.create(self.equ_type)

  def _init_ode_solver(self):
    self.ode_solver = ODESOLVERFACTORY.create(self.ode_solver_type, self.ode,
                                              self.t_step, self.dt)

  def _init_dataset(self, x_sample_func, param_sample_func):
    return NotImplementedError

  def _init_dictionary(self, observable_func):
    return NotImplementedError

  def _init_solver(self):
    return NotImplementedError


class EDMDRBFSolverWrapper(SolverWrapper):

  def __init__(self, config_file):
    """ Initializes the SolverWrapper instance.

    Args:
      config_file (str): The path to the configuration file.

    Attributes:
      _use_param_dataset (bool): A flag indicating whether to use a parameterized dataset.
    """
    self._use_param_dataset = False
    super().__init__(config_file)

  def _read_dictionary_config(self):
    super()._read_dictionary_config()
    self.reg = self._data['dictionary']['reg']

  def _init_dataset(self, x_sample_func, param_sample_func):
    self.dynamics = DiscreteDynamics(self.ode_solver, self.ode.dim)
    self.dataset = KoopmanDataSet(self.dynamics, x_sample_func)
    self.dataset.generate_data(self.n_traj, self.traj_len, self.x_min,
                               self.x_max, self.param, self.seed_x)

  def _init_dictionary(self, observable_func):
    self.dictionary = RBFDictionary(self.dataset.data_x, observable_func,
                                    self.ode.dim, self.dim_output, self.reg)

  def _init_solver(self):
    self.solver = EDMDSolver(self.dictionary)

  def solve(self):
    """Executes the EDMD-RBF method using the provided dataset.

    Returns:
      Koopman: The result of EDMD method.
    """
    return self.solver.solve(self.dataset)


class EDMDDLSolverWrapper(SolverWrapper):

  def __init__(self, config_file):
    """Initializes the SolverWrapper instance.

    Args:
      config_file (str): The path to the configuration file.

    Attributes:
      _use_param_dataset (bool): A flag indicating whether to use a parameterized dataset.
    """
    self._use_param_dataset = False
    super().__init__(config_file)

  def _read_dataset_config(self):
    super()._read_dataset_config()
    self.train_ratio = self._data['dataset']['train_ratio']

  def _read_dictionary_config(self):
    super()._read_dictionary_config()
    self.dic_layer_sizes = self._data['dictionary']['dic_layer_sizes']

  def _read_solver_config(self):
    super()._read_solver_config()
    self.reg = self._data['solver']['reg']
    self.reg_final = self._data['solver']['reg_final']
    self.n_epochs = self._data['solver']['n_epochs']
    self.batch_size = self._data['solver']['batch_size']
    self.tol = self._data['solver']['tol']
    self.dic_lr = self._data['solver']['dic_lr']

  def _init_dataset(self, x_sample_func, param_sample_func):
    self.dynamics = DiscreteDynamics(self.ode_solver, self.ode.dim)
    self.dataset = KoopmanDataSet(self.dynamics, x_sample_func)
    self.dataset.generate_data(self.n_traj, self.traj_len, self.x_min,
                               self.x_max, self.param, self.seed_x)
    self.train_dataset, self.val_dataset = torch.utils.data.random_split(
        self.dataset, [
            int(self.train_ratio * len(self.dataset)),
            len(self.dataset) - int(self.train_ratio * len(self.dataset))
        ])

  def _init_dictionary(self, observable_func):
    network = FullConnResNet(self.ode.dim,
                             self.dim_output - observable_func.dim - 1,
                             self.dic_layer_sizes)
    self.dictionary = TrainableDictionary(network, observable_func,
                                          self.ode.dim, self.dim_output)

  def _init_solver(self):
    self.solver = EDMDDLSolver(self.dictionary, self.reg, self.reg_final)

  def solve(self):
    """Solves the problem using EDMDDL algorithm with the provided datasets.

    Returns:
      Koopman: The result of the EDMDDL algorithm.
    """
    return self.solver.solve(self.train_dataset, self.val_dataset,
                             self.n_epochs, self.batch_size, self.tol,
                             self.dic_lr)


class ParamKoopmanDLSolverWrapper(SolverWrapper):

  def __init__(self, config_file):
    self._use_param_dataset = True
    super().__init__(config_file)

  def save_dataset(self, path):
    self.dataset.save(path)

  def load_dataset_setup(self,
                         path,
                         observable_func,
                         x_sample_func=torch.rand,
                         param_sample_func=torch.rand):
    self._init_ode()
    self._init_ode_solver()
    self.dynamics = DiscreteDynamics(self.ode_solver, self.ode.dim,
                                     self.ode.param_dim)
    self.dataset = ParamKoopmanDataSet(self.dynamics, x_sample_func,
                                       param_sample_func)
    self.dataset.load(path)
    self.train_dataset, self.val_dataset = torch.utils.data.random_split(
        self.dataset, [
            int(self.train_ratio * len(self.dataset)),
            len(self.dataset) - int(self.train_ratio * len(self.dataset))
        ])
    self._init_dictionary(observable_func)
    self._init_solver()

  def _read_dataset_config(self):
    super()._read_dataset_config()
    self.train_ratio = self._data['dataset']['train_ratio']

  def _read_dictionary_config(self):
    super()._read_dictionary_config()
    self.dic_layer_sizes = self._data['dictionary']['dic_layer_sizes']

  def _read_solver_config(self):
    super()._read_solver_config()
    self.n_epochs = self._data['solver']['n_epochs']
    self.batch_size = self._data['solver']['batch_size']
    self.tol = self._data['solver']['tol']
    self.dic_lr = self._data['solver']['dic_lr']
    self.koopman_layer_sizes = self._data['solver']['koopman_layer_sizes']
    self.koopman_lr = self._data['solver']['koopman_lr']

  def _init_dataset(self, x_sample_func, param_sample_func):
    self.dynamics = DiscreteDynamics(self.ode_solver, self.ode.dim,
                                     self.ode.param_dim)
    self.dataset = ParamKoopmanDataSet(self.dynamics, x_sample_func,
                                       param_sample_func)
    self.dataset.generate_data(self.n_traj, self.n_traj_per_param,
                               self.traj_len, self.x_min, self.x_max,
                               self.param_min, self.param_max, self.seed_x,
                               self.seed_param, self.param_time_dependent)
    self.train_dataset, self.val_dataset = torch.utils.data.random_split(
        self.dataset, [
            int(self.train_ratio * len(self.dataset)),
            len(self.dataset) - int(self.train_ratio * len(self.dataset))
        ])

  def _init_dictionary(self, observable_func):
    network = FullConnResNet(self.ode.dim,
                             self.dim_output - observable_func.dim - 1,
                             self.dic_layer_sizes)
    self.dictionary = TrainableDictionary(network, observable_func,
                                          self.ode.dim, self.dim_output)

  def _init_solver(self):
    self.solver = ParamKoopmanDLSolver(self.dictionary)
    network = FullConnNet(self.ode.param_dim, self.dim_output**2,
                          self.koopman_layer_sizes)
    self.K = ParamKoopman(self.dim_output, network)

  def solve(self, weight=None):
    """Applies the parametric Koopman learning algorithm.

    Returns:
      ParamKoopman: The result of the parametric Koopman learning algorithm.
    """
    return self.solver.solve(self.train_dataset, self.val_dataset, self.K,
                             self.n_epochs, self.batch_size, self.tol,
                             self.dic_lr, self.koopman_lr, weight)
