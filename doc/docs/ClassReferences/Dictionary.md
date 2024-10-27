
A `Dictionary` is a vector function
$\Psi = (\psi_1, \psi_2, \cdots, \psi_{N_\psi})^T$,
where $\psi_i = g_i$ for $i = 1,\cdots,N_y$.
It supports batched operation
$\Psi: \mathbb{R}^{N \times N_x} \rightarrow \mathbb{R}^{N \times N_{\psi}}$.

## Attributes

- `_function` (ndarray -> ndarray): The batched vector function
  $\Psi: \mathbb{R}^{N \times N_x} \rightarrow \mathbb{R}^{N \times N_{\psi}}$.
- `_dim_input` (int): The input dimension $N_x$.
- `_dim_output` (int): The output dimension $N_{\psi}$.

## Methods

- `__init__(self, function, dim_input, dim_output)`
- `__call__(self, x)`: Applies the dictionary to input `x`, which must satisfy $x \in \mathbb{R}^{N \times N_x}$.

