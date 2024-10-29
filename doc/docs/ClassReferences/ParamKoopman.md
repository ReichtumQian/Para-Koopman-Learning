
It represents a mapping $K: \mathrm{span}(\Psi) \times U \rightarrow \mathrm{span}(\Psi)$,
which is a finite-dimensional approximation of the parametric Koopman operator.

## Attributes

- `_network` (torch.nn.Module): The neural network that approximates the parametric Koopman operator. Given the parameters `u`, returns the generated Koopman operator matrix.
- `_size` (int): The size of the Koopman operator matrix.

## Methods

- `__init__(self, size_K, network)`
- `__call__(self, para, x)`: Applies the parametric Koopman operator on `(x, para)`,
  which should satisfy $x \in \mathbb{R}^{N \times N_{\psi}}$,
  $\text{para} \in \mathbb{R}^{N \times N_u}$.
- `parameters(self)`: Returns the parameters of the neural network.
- `predict(self, para, x0, dictionary, dim_nontrain, traj_len)`: Predicts the trajectory of the system given the initial state `x0`, the dictionary `dictionary`, and the parameters `para`.
    - `para` (tensor): The parameters of the system.
    - `x0` (tensor): The initial state of the system.
    - `dictionary` (Dictionary): The dictionary used to represent the state of the system.
    - `dim_nontrain` (int): The dimension of the non-trainable part of the state.
    - `traj_len` (int): The length of the trajectory to predict.





