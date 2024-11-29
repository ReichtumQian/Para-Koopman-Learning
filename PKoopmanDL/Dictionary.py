import numpy as np
import torch
import scipy


class ObservableFunction:

  def __init__(self, func, dim_output):
    self._func = func
    self._dim = dim_output

  @property
  def dim(self):
    return self._dim

  def __call__(self, x):
    return self._func(x)


class Dictionary:

  def __init__(self, function, dim_input, dim_output):
    self._function = function
    self._dim_input = dim_input
    self._dim_output = dim_output

  def __call__(self, x):
    return self._function(x)

  @property
  def dim_input(self):
    return self._dim_input

  @property
  def dim_output(self):
    return self._dim_output


class TrainableDictionary(Dictionary):

  def __init__(self, network, observable_func, dim_input, dim_output):
    self._network = network
    assert dim_output > observable_func.dim + 1, "dim_output must be greater than observable_func.dim + 1"
    function = lambda x: torch.cat(
        (observable_func(x).to(x.device), torch.ones(
            (x.size(0), 1)).to(x.device), self._network(x).to(x.device)),
        dim=1)
    super().__init__(function, dim_input, dim_output)

  def parameters(self):
    return self._network.parameters()

  def train(self):
    self._network.train()

  def eval(self):
    self._network.eval()

  def save(self, path):
    torch.save(self._network.state_dict(), path)

  def load(self, path):
    self._network.load_state_dict(torch.load(path))


class RBFDictionary(Dictionary):

  def __init__(self, data_x, observable_func, dim_input, dim_output, reg):
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
