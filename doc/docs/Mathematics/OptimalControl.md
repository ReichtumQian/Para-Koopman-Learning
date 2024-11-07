
Here we consider the Bolza control problem whose cost function depends on the
observable values

$$\min \limits_{\{\mathbf{u}_n\}_{n = 0,1,\cdots,N-1}}
J[\{\mathbf{u}_n\}] = \Phi(\mathbf{g}(\mathbf{x}_N)) +
\sum\limits_{n = 1}^N L_n(\mathbf{g}(\mathbf{x}_n), \mathbf{u}_{n-1}).$$

$$ \mathrm{s.t.} \quad \mathbf{x}_{n+1} = \mathbf{f}(\mathbf{x}_n, \mathbf{u}_n). $$

where the initial condition, the terminal cost $\Phi$
and the running cost $L_n$ are given,
but the dynamics $\mathbf{f}$ is unknown.
$\mathbf{g}(\mathbf{x})$ is the observable function,
and $\{\mathbf{u}_n\}_{n = 0}^{N-1}$ the controls.

## Optimal Control problems under PK-NN

PK-NN transforms the above optimal control problem into

$$
\min_{\{\mathbf{u}_{n}\}_{n=0,1,\ldots,N-1}} J [\{\mathbf{u}_{n}\}]
= \Phi(B\hat{\Psi}_{N}) +
\sum_{n=1}^{N} L_{n} (B\hat{\Psi}_{n},\mathbf{u}_{n-1})
$$

$$
\mathrm{s.t.} \quad \hat{\Psi}_{n+1} =
K(\mathbf{u}_{n})\hat{\Psi}_{n}, \quad
\hat{\Psi}_{0}=\Psi(\mathbf{x}_{0}),
$$



