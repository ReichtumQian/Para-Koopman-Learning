import numpy as np
import torch
import scipy


class ObservableFunction:

  def __init__(self, func, dim_output):
    """Observable functions of the dynamical system.

    Args:
        func (torch.Tensor -> torch.Tensor): Observable functions $(N, N_x) \\rightarrow (N, N_g)$
        dim_output (int): The dimension of the output $N_g$.
    """
    self._func = func
    self._dim = dim_output

  @property
  def dim(self):
    """The dimension of the output.

    Returns:
        int: The dimension of the output.
    """
    return self._dim

  def __call__(self, x):
    """Apply the observable functions to the input.

    Args:
        x (torch.Tensor): The input state of the size $(N, N_x)$

    Returns:
        torch.Tensor : The output of the observable functions.
    """
    return self._func(x)


class Dictionary:

  def __init__(self, function, dim_input, dim_output):
    """Dictionary $\\Psi$.

    Args:
        function (torch.Tensor -> torch.Tensor): Dictionary basis function $(N, N_x) \\rightarrow (N, N_{\\psi})$
        dim_input (int): Input dimension $N_x$.
        dim_output (int): Output dimension $N_\\psi$.
    """
    self._function = function
    self._dim_input = dim_input
    self._dim_output = dim_output

  def __call__(self, x):
    """Apply the dictionary to the input.

    Args:
        x (torch.Tensor): Input state of the size $(N, N_x)$

    Returns:
        torch.Tensor: The basis functions of the size $(N, N_\\psi)$
    """
    return self._function(x)

  @property
  def dim_input(self):
    """Input dimension.

    Returns:
        int: The input dimension.
    """
    return self._dim_input

  @property
  def dim_output(self):
    """Output dimension.

    Returns:
        int: The output dimension.
    """
    return self._dim_output


class TrainableDictionary(Dictionary):

  def __init__(self, network, observable_func, dim_input, dim_output):
    """A trainable dictionary

    Args:
      network (torch.nn.Module): The trainable network.
      observable_func (torch.Tensor -> torch.Tensor): Observable functions $(N, N_x) \\rightarrow (N, N_{\\psi})$
      dim_input (int): Input dimension $N_x$.
      dim_output (int): Output dimension $N_\\psi$.
      
    Notes:
      The constant observable function $\\mathbf{1}$ is included in the `TrainableDictionary` by default, so users don't need to define it explicitly.
    
    """
    self._network = network
    assert dim_output > observable_func.dim + 1, "dim_output must be greater than observable_func.dim + 1"
    function = lambda x: torch.cat(
        (observable_func(x).to(x.device), torch.ones(
            (x.size(0), 1)).to(x.device), self._network(x).to(x.device)),
        dim=1)
    super().__init__(function, dim_input, dim_output)

  def parameters(self):
    """Return the parameters of the network.

    Returns: 
        Iterator[torch.nn.Parameter]: The parameters of the network.
    """
    return self._network.parameters()

  def train(self):
    """Set the network to training mode.
    """
    self._network.train()

  def eval(self):
    """Set the network to evaluation mode.
    """
    self._network.eval()

  def save(self, path):
    """Save the network to a file.

    Args:
        path (str): The path to save the network.
    """
    torch.save(self._network.state_dict(), path)

  def load(self, path):
    """Load the network from a file.

    Args:
        path (str): The path to load the network.
    """
    self._network.load_state_dict(torch.load(path))


class RBFDictionary(Dictionary):

  def __init__(self, data_x, observable_func, dim_input, dim_output, reg):
    """Initialize the RBF dictionary.

    Args:
        data_x (torch.Tensor): The data to initialize the RBF dictionary.
        observable_func (ObservableFunction): The observable functions.
        dim_input (int): Input dimension $N_x$.
        dim_output (int): Output dimension $N_\\psi$.
        reg (float): The regularization parameter.
    """
    assert dim_output > observable_func.dim + 1, "dim_output must be greater than observable_func.dim + 1"
    dim_train = dim_output - observable_func.dim - 1
    # set the seed if you want to reproduce the same results
    # np.random.seed(0)
    centers = scipy.cluster.vq.kmeans(data_x, dim_train)[0]

    def func(x):
      rbfs = []
      for n in range(dim_train):
        r = scipy.spatial.distance.cdist(x, np.matrix(centers[n, :]))
        rbf = scipy.special.xlogy(r**2, r + reg)
        rbfs.append(rbf)
      rbfs = np.array(rbfs)
      rbfs = rbfs.T.reshape(x.shape[0], -1)
      results = np.concatenate(
          [observable_func(x).detach().numpy(),
           np.ones((x.shape[0], 1)), rbfs],
          axis=1)
      return torch.from_numpy(results)

    super().__init__(func, dim_input, dim_output)
