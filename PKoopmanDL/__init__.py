import torch

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
  info_message("CUDA is available. GPU is being used.")
else:
  info_message("CUDA is not available. Using CPU for computation.")
