
The class `AbstractODE` is an abstract class for ordinary differential equations (ODEs)
of the form

$$ \dot{\mathbf{x}}(t) = \mathbf{f} (\mathbf{x}(t), \mathbf{u}). $$

## Attributes

- `_dim` (int): The dimension of the ODE.
- `_param_dim` (int): The dimension of the parameter $\mathbf{u}$.
- `_rhs` ((tensor, tensor) -> tensor): The right-hand side function of the ODE.

!!! info
    Since the `rhs` function must accommodate both fixed parameters and variable parameters,
    it should be capable of handling inputs of the form 
    $\mathbb{R}^{N \times N_x} \times \mathbb{R}^{1 \times N_u}$ (fixed parameters) and
    $\mathbb{R}^{N \times N_x} \times \mathbb{R}^{N \times N_u}$ (variable parameters).

## Methods

- `__init__(self, dim, param_dim, rhs)`:
    - `dim` (int): The dimension of the ODE.
    - `param_dim` (int): The dimension of the parameter $\mathbf{u}$.
    - `rhs` ((tensor, tensor) -> tensor): The right-hand side function of the ODE.
      It must be able to handle 
      $\mathbb{R}^{N \times N_x} \times \mathbb{R}^{N \times N_u} \rightarrow \mathbb{R}^{N \times N_x}$ and
      $\mathbb{R}^{N \times N_x} \times \mathbb{R}^{1 \times N_u} \rightarrow \mathbb{R}^{N \times N_x}$.
- `rhs(self, x, u)`: Computes the right-hand side function using the inputs `(x, u)`.
    - `x` (ndarray): The state of the system, in the form $\mathbb{R}^{N \times N_x}$.
    - `u` (ndarray): The parameter of the system, in the form $\mathbb{R}^{N \times N_u}$ or $\mathbb{R}^{1 \times N_u}$.
