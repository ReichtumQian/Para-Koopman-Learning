
import numpy as np
import torch
import scipy

class Dictionary:
  
  def __init__(self, function, dim_input, dim_output, dim_nontrain = 0):
    """ Initialize the Dictionary

    Args:
        function (tensor -> tensor): A batched vector function representing the basis functions.
        dim_input (int): The dimension of the input.
        dim_output (int): The dimension of the output.
    """
    self._function = function
    self._dim_input = dim_input
    self._dim_output = dim_output
    self._dim_nontrain = dim_nontrain
  
  def __call__(self, x):
    """ Apply the dictionary

    Args:
        x (tensor): The input $\mathbb{R}^{N \times dim_input}$.
    Returns:
        tensor: The output $\mathbb{R}^{N \times dim_output}$.
    """
    return self._function(x)
  
  @property
  def dim_input(self):
    return self._dim_input
  
  @property
  def dim_output(self):
    return self._dim_output

  @property
  def dim_nontrain(self):
    return self._dim_nontrain

    
class TrainableDictionary(Dictionary):

  def __init__(self, network, nontrain_func, dim_input, dim_output, dim_nontrain):
    """ Initialize the TrainableDictionary

    Args:
        network (torch.nn.Module): The trainable neural network,
            which is a mapping $\mathbb{R}^{N \times N_x} \rightarrow \mathbb{R}^{N \times (N_{\psi} - N_y)}$.
        nontrain_func (tensor -> tensor): The non-trainable neural network,
            which is a mapping $\mathbb{R}^{N \times N_x} \rightarrow \mathbb{R}^{N \times N_y}$.
        dim_input (int): The dimension of the input $N_x$.
        dim_output (int): The dimension of the output $N_{\psi}$.
        dim_nontrain (int): The number of non-trainable outputs $N_y$.
    """
    self.__network = network
    function = lambda x: torch.cat((nontrain_func(x), self.__network(x)), dim=1)
    super().__init__(function, dim_input, dim_output, dim_nontrain)
    
  def parameters(self):
    """Return the parameters of the trainable neural network.

    Returns:
        iterable: An iterable of parameters of the neural network `__network`.
    """
    return self.__network.parameters()

  def train(self):
    """Set the trainable neural network to training mode.
    """
    self.__network.train()


class RBFDictionary(Dictionary):
  
  def __init__(self, data_x, nontrain_func, dim_input, dim_output, dim_nontrain, regularizer):
    data_x = data_x.detach().numpy()
    dim_train = dim_output - dim_nontrain
    centers = scipy.cluster.vq.kmeans(data_x, dim_train)[0]
    def func(x):
      rbfs = []
      for n in range(dim_train):
        r = scipy.spatial.distance.cdist(x, np.matrix(centers[n, :]))
        rbf = scipy.special.xlogy(r**2, r + regularizer)
        rbfs.append(rbf)
      rbfs = np.array(rbfs)
      rbfs = rbfs.T.reshape(x.shape[0], -1)
      results = np.concatenate([nontrain_func(x).detach().numpy(), rbfs], axis=1)
      return torch.from_numpy(results)
    super().__init__(func, dim_input, dim_output, dim_nontrain)
