


Consider the autonomous (non-parametric) dynamical systems of the form

$$
\begin{align*}
\mathbf{x}_{n+1} &= \mathbf{f}(\mathbf{x}_n),\\
\mathbf{y}_n &= \mathbf{g}(\mathbf{x}_n),
\end{align*}
$$

where $\mathbf{g}$ is a length-$N_y$ vector of $L^2$ observable functions $\mathbf{y}$.
The goal is to find a finite-dimensional subspace $H \subset L^2$ that
contains the components of $\mathbf{g}$,
and moreover is *invariant* under the action of the Koopman operator, i.e.,

$$ \mathcal{K}(H) \subset H, $$

at least approximately.

## Dictionary

We build $H$ as the span of a set of *dictionary* functions
$\{\psi_1,\psi_2,\cdots,\psi_{N_{\psi}}\}$ where $\psi_i \in L^2(X, m)$,
with $\psi_i = g_i$ for $i = 1,2,\cdots,N_y$.
Let we write $\Psi := (\psi_1,\psi_2,\cdots,\psi_{N_{\psi}})^T$,
and consider the subspace $\mathrm{span}(\Psi) = \left\{ a^T\Psi: a \in \mathbb{R}^{N_{\psi}} \right\}$.
If we assume $\mathcal{K}(\mathrm{span}(\Psi)) \subset \mathrm{span}(\Psi)$,
then $\mathcal{K}$ can be considered as a linear transformation in
$\mathrm{span}(\Psi)$. Thus $\mathcal{K}$ can be represented
by a matrix $K \in \mathbb{R}^{N_{\psi} \times \mathbb{R}^{N_{\psi}}}$, satisfying

$$ \mathcal{K} \Psi  = K \Psi. $$

## EDMD Algorithm

From a data science perspective, we collect data pairs
$\{(\mathbf{x}_{n+1}^{(m)}, \mathbf{x}_n^{(m)})\}_{n,m=0}^{N-1,M-1}$,
where $\mathbf{x}_{n+1}^{(m)} = \mathbf{f} (\mathbf{x}_n^{(m)})$
and $\mathbf{x}_n^{(m)}$ is the state on the $m$th trajectory at time $n$.
Then, an approximation of the Koopman operator on this subspace is computed
via least squares

$$ \hat{K} =  \operatorname*{argmin}_{K \in \mathbb{R}^{N_{\psi} \times N_{\psi}}}
\sum\limits_{n,m = 0}^{N-1,M-1} \|\Psi(\mathbf{x}_{n+1}^{(m)}) - K \Psi(\mathbf{x}_n^{(m)})\|^2.$$

The solution is guaranteed to be unique when the number of data pairs is at
least equal to or larger than the dimension of the dictionary $\Psi$.

**Proposition**. The solution to the above problem is

$$ K = (YX^T)(XX^T)^+, $$

where $X = [\Psi(x_n^{(0)}), \cdots, \Psi(x_n^{(N-1)})]$,
$Y = [\Psi(x_{n+1}^{(0)}), \cdots \Psi(x_{n+1}^{(N-1)})]$,
and $+$ the pseudo inverse.

