


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

## Spectral Analysis

**Lemma**. Under appropriate scaling, the left eigenvectors $\xi_j$
and the corresponding right eigenvector $w_j$ of a matrix $K$ satisfy

$$ \xi_j^{\ast} w_j = 1, \xi_j^{\ast} w_i = 0, \quad \text{for} ~ i \neq j. $$

**Proof**. By the definition of the left eigenvector $\xi_j^{\ast}K = \lambda_j \xi_j^{\ast}$,
we have

$$ \xi_j^{\ast} K w_i = \xi_j^{\ast} (\lambda_i w_i) = \lambda_i(\xi_j^{\ast}w_i). $$

On the other hand, using the eigenvalue equation for $\xi_j^{\ast}$, we also have

$$ \xi_j^{\ast} K w_i = \lambda_j(\xi_j^{\ast}w_i). $$

By comparing these two expressions, we obtain

$$ \lambda_i(\xi_j^{\ast} w_i) = \lambda_j(\xi_j^{\ast}w_i). $$

If $\lambda_i \neq \lambda_j$, this equation implies that $\xi_j^{\ast}w_i = 0$.
If $\lambda_i = \lambda_j$, then $\xi_j^{\ast}w_i$ can be any number.
To satisfy the nomalization condition, we typically choose $\xi_j^{\ast}w_j =1$,
which completes the proof.

**Corollary**. Given a matrix $K \in \mathbb{R}^{N \times N}$,
denote $\Xi = [\xi_1,\cdots,\xi_N]$ and $W = [w_1,\cdots,w_M]$,
where $\xi_i, w_i$ are the left eigenectors and right eigenvectors
of $K$ with eigenvalue $\mu_i$, respectively.
Then under appropriate scaling, we have

$$ \Xi^{\ast} = W^{-1}. $$

**Proposition**. Suppose $\xi_j$ is a left eigenvector of matrix $K$ with
eigenvalue $\mu_j$. Then the function

$$ \varphi_j = \xi_j^\ast \Psi $$

is an eigenfunction of $\mathcal{K}$ with the same eigenvalue $\mu_j$.

**Proof**. Direct calculation yields

$$
\mathcal{K} \xi_j^{\ast} \Psi
= \xi_j^{\ast} \mathcal{K} \Psi
= \xi_j^\ast K \Psi
= \mu_j \xi_j^{\ast}\Psi.
$$

**Proposition**. Consider the observable $\mathbf{g}(\mathbf{x})$.
Assume that for all $g_i(\mathbf{x}) \in L^2(X, m)$,
there exists $V \in \mathbb{C}^{N_g \times N_{\psi}}$ such that

$$ \mathbf{g}(\mathbf{x}) = V \Phi(\mathbf{x}) $$

where $\Phi(\mathbf{x}) = [\varphi_1(x),\cdots,\varphi_{N_{\psi}}(x)]^T$.

**Proof**. Since all $g_i(\mathbf{x}) \in L^2(X, m)$,
we have $g_i(\mathbf{x}) = \sum\limits_{k = 1}^{N_{\psi}}
\psi_k(\mathbf{x})b_{k,i} = \mathbf{b}_i^T \Psi(\mathbf{x})$,
which yields

$$ \mathbf{g}(\mathbf{x}) = \left[
  \begin{array}{cccc}
    \mathbf{b}_1^T \Psi(x)&\mathbf{b}_2^T \Psi(x)&\cdots&\mathbf{b}_{N_{\psi}}^T \Psi(x)
  \end{array}
\right] = B \Psi(x). $$

Next we express $\psi_i$ in terms of $\varphi_i$, we have

$$
\Phi(x) = \left[
  \begin{array}{c}
    \xi_1^{\ast}\\
    \xi_2^{\ast}\\
    \vdots\\
    \xi_{N_{\psi}}^{\ast}
  \end{array}
\right] = \Xi^{\ast} \Psi(x).
$$

By the relation between left and right eigenvectors, we have $(\Xi^{\ast})^{-1} = W$.
Combining above equations yields

$$ \mathbf{g}(\mathbf{x}) = V \Phi(\mathbf{x}) = B W \Phi(\mathbf{x}). $$

!!! note
    Since the first $N_y$ functions of $\Psi$ are the observables $\mathbf{g}$,
    here we have

    $$ B =
    \left[
    \begin{array}{cc}
      I_{N_y}& O_{N_y \times (N_{\psi} - N_y)}
    \end{array}
    \right]_{N_y \times N_{\psi}}.
    $$



