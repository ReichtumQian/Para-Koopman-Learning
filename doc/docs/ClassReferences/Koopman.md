
The class `Koopman` is a mapping $K: \mathrm{span}(\Psi) \rightarrow \mathrm{span}(\Psi)$,
which acts as the finite-dimensional approximation of the Koopman operator $\mathcal{K}$,
i.e.,

$$ \mathcal{K} \Psi \approx K \Psi. $$

Given $\phi = \mathbf{a}^T \Psi$, where $\mathbf{a} \in \mathbb{R}^{N_{\psi}}$,
then the application of $\mathcal{K}$ satisfies

$$ \mathcal{K} \phi = \mathcal{K} \mathbf{a}^T \Psi \approx \mathbf{a}^T K \Psi. $$

## Attributes

- `_func` (tensor -> tensor): The mapping $K: \mathrm{span}(\Psi) \rightarrow \mathrm{span}(\Psi)$.

## Methods

- `__init__(self, K = None, func = None)`
    - `K` (tensor): The matrix representation of the Koopman operator.
    - `func` (tensor -> tensor): The mapping $K: \mathrm{span}(\Psi) \rightarrow \mathrm{span}(\Psi)$,
- `__call__(self, x)`: Applies the Koopman operator,
  `x` should satisfy $x \in \mathbb{R}^{N \times N_{\psi}}$.
- `predict(self, x0, dictionary, dim_nontrain, traj_len)`: Predicts the trajectory of the system.
    - `x0` (tensor): The initial state of the system.
    - `dictionary` (Dictionary): The dictionary used to map the state to the feature space.
    - `dim_nontrain` (int): The dimension of the non-trainable part of the state.
    - `traj_len` (int): The length of the trajectory to predict.

!!! info
    Understanding the `__call__` method of `Koopman`: Given the data set $\{x^{(n)}\}_{n = 1}^N$,
    it represents a mapping:

    $$ \left[
      \begin{array}{cccc}
        \psi_1(x^{(1)})&\psi_2(x^{(1)})&\cdots&\psi_{N_{\psi}}(x^{(1)})\\
        \psi_1(x^{(2)})&\psi_2(x^{(2)})&\cdots&\psi_{N_{\psi}}(x^{(2)})\\
        \vdots&\vdots&\ddots&\vdots\\
        \psi_1(x^{(N)})&\psi_2(x^{(N)})&\cdots&\psi_{N_{\psi}}(x^{(N)})\\
      \end{array}
    \right] \rightarrow \left[
      \begin{array}{cccc}
        \mathcal{K}\psi_1(x^{(1)})&\mathcal{K}\psi_2(x^{(1)})&\cdots&\mathcal{K}\psi_{N_{\psi}}(x^{(1)})\\
        \mathcal{K}\psi_1(x^{(2)})&\mathcal{K}\psi_2(x^{(2)})&\cdots&\mathcal{K}\psi_{N_{\psi}}(x^{(2)})\\
        \vdots&\vdots&\ddots&\vdots\\
        \mathcal{K}\psi_1(x^{(N)})&\mathcal{K}\psi_2(x^{(N)})&\cdots&\mathcal{K}\psi_{N_{\psi}}(x^{(N)})\\
      \end{array}
    \right]$$
