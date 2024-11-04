import torch

from .Dictionary import *
from .FlowMap import *
from .Koopman import *
from .Net import *
from .ODE import *
from .ODEDataSet import *
from .Solver import *
from .SolverWrapper import *

if torch.cuda.is_available():
  print("CUDA is available. GPU is being used.")
else:
  print("CUDA is not available. Using CPU for computation.")
