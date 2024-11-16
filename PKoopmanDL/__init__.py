import torch

from .ControlSolver import *
from .Dictionary import *
from .Dynamics import *
from .Koopman import *
from .Log import *
from .Net import *
from .ODE import *
from .DynamicsDataSet import *
from .KoopmanSolver import *
from .SolverWrapper import *

if torch.cuda.is_available():
  debug_message("CUDA is available. GPU is being used.")
else:
  debug_message("CUDA is not available. Using CPU for computation.")
