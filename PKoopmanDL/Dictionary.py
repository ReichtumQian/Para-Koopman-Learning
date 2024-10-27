
import numpy as np

class Dictionary:
  
  def __init__(self, function, dim_input, dim_output):
    """ Initialize the Dictionary

    Args:
        function (ndarray -> ndarray): A batched vector function representing the basis functions.
        dim_input (int): The dimension of the input.
        dim_output (int): The dimension of the output.
    """
    self._function = function
    self._dim_input = dim_input
    self._dim_output = dim_output
  
  def __call__(self, x):
    """ Apply the dictionary

    Args:
        x (ndarray): The input $\mathbb{R}^{N \times dim_input}$.
    Returns:
        ndarray: The output $\mathbb{R}^{N \times dim_output}$.
    """
    return self._function(x)
    
class TrainableDictionary(Dictionary):

  def __init__(self, network, nontrain_func, dim_input, dim_output, dim_nontrain):
    """ Initialize the TrainableDictionary

    Args:
        network (torch.nn.Module): The trainable neural network,
            which is a mapping $\mathbb{R}^{N \times N_x} \rightarrow \mathbb{R}^{N \times (N_{\psi} - N_y)}$.
        nontrain_func (ndarray -> ndarray): The non-trainable neural network,
            which is a mapping $\mathbb{R}^{N \times N_x} \rightarrow \mathbb{R}^{N \times N_y}$.
        dim_input (int): The dimension of the input $N_x$.
        dim_output (int): The dimension of the output $N_{\psi}$.
        dim_nontrain (int): The number of non-trainable outputs $N_y$.
    """
    self.__dim_nontrain = dim_nontrain
    self.__network = network
    function = lambda x: np.concatenate((nontrain_func(x), self.__network(x)), axis=0)
    super().__init__(function, dim_input, dim_output)
    
  def parameters(self):
    """Return the parameters of the trainable neural network.

    Returns:
        iterable: An iterable of parameters of the neural network `__network`.
    """
    return self.__network.parameters()

