import numpy as np
import torch


class Koopman:

  def __init__(self, K):
    """Initialize the Koopman instance.

    Args:
        func (tensor -> tensor): A mapping function representing the Koopman operator.
        K (tensor): The Koopman matrix.
    """
    func = lambda x: (K @ x.t()).t()
    self._K = K
    self._func = func
    # self._compute_eig()

  def __call__(self, x):
    """Apply the Koopman operator on the input `x`.

    Args:
        x (tensor): The input to apply the Koopman operator, expected to be of shape (N, N_psi).

    Returns:
        tensor : The output of the Koopman operator.
    """
    return self._func(x)

  def predict(self, x0, dictionary, state_pos, traj_len):
    y = []
    x = x0
    psi = dictionary(x0)
    y.append(psi[:, state_pos])
    for _ in range(traj_len - 1):
      psi = self(psi)
      x = psi[:, state_pos]
      y.append(x)
      psi = dictionary(x)
    return torch.stack(y, dim=1)  # size: (N, traj_len, number of state)

  def predict_observable(self, x0, dictionary, observable_pos, traj_len):
    y = []
    psi = dictionary(x0)
    y.append(psi[:, observable_pos])
    for _ in range(traj_len - 1):
      psi = self(psi)
      y.append(psi[:, observable_pos])
    return torch.stack(y, dim=1)  # size: (N, traj_len, number of observable)

  def _compute_eig(self):
    eigenvalues, eigenvectors = np.linalg.eig(self._K.detach().numpy())
    idx = eigenvalues.real.argsort()[::-1]
    self.eigenvalues = torch.from_numpy(eigenvalues[idx])
    self.right_eigenvectors = torch.from_numpy(eigenvectors[:, idx])
    self.left_eigenvectors = torch.from_numpy(
        np.linalg.inv(self.right_eigenvectors))
    self.left_eigenvectors = torch.conj(self.left_eigenvectors.t())

  def save(self, path):
    torch.save(self._K, path)

  def load(self, path):
    K = torch.load(path)
    self.__init__(K)


class ParamKoopman:

  def __init__(self, size_K, network):
    """Initialize the ParamKoopman instance.

    Args:
        network ((tensor) -> tensor): Given the parameters `u`, returns the generated Koopman operator matrix.
    """
    self._size = size_K
    self._network = network

  def __call__(self, param, x):
    """Generate a Koopman operator based on the given parameters.

    Args:
        para (tensor): The parameter data, expected to be of shape (N, N_u).

    Returns:
        Koopman: The Koopman operator corresponding to the given parameters.
    """
    if param.size(0) == 1:
      net_param = param.expand(x.size(0), -1)
    else:
      net_param = param
    net_out = self._network(net_param)
    K = net_out.reshape(net_out.size(0), self._size, self._size)
    x = x.unsqueeze(2)
    result = torch.bmm(K, x).squeeze(2)
    return result

  def parameters(self):
    return self._network.parameters()

  def predict(self, param, x0, dictionary, state_pos, traj_len):
    y = []
    psi = dictionary(x0)
    y.append(psi[:, state_pos])
    for _ in range(traj_len - 1):
      psi = self(param, psi)
      x = psi[:, state_pos]
      y.append(x)
      psi = dictionary(x)
    return torch.stack(y, dim=1)  # size: (N, traj_len, number of state)

  def predict_observable(self, param, x0, dictionary, observable_pos, traj_len):
    y = []
    psi = dictionary(x0)
    y.append(psi[:, observable_pos])
    for _ in range(traj_len - 1):
      psi = self(param, psi)
      y.append(psi[:, observable_pos])
    return torch.stack(y, dim=1)  # size: (N, traj_len, number of observable)

  def train(self):
    self._network.train()

  def eval(self):
    self._network.eval()

  def save(self, path):
    data_to_save = {
        'state_dict': self._network.state_dict(),
        'size': self._size
    }
    torch.save(data_to_save, path)

  def load(self, path):
    data_loaded = torch.load(path)
    self._size = data_loaded['size']
    self._network.load_state_dict(data_loaded['state_dict'])
