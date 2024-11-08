
Here we apply the PK-NN to control the forced Korteweg-De Vries (KdV) equation

$$ \frac{\partial \eta(t, x)}{\partial t} + \eta(t, x) \frac{\partial \eta(t,
  x)}{\partial x} + \frac{\partial^3 \eta(t, x)}{\partial x^3} = w(t, x), $$

where $\eta(t, x)$ is the state,
$w(t,x) = \sum\limits_{i = 1}^3v_i(x) \sin(\pi u_i(t))$ is the forcing term.
Control parameters at $t_n$ are
$\mathbf{u}_n = (u_{1,n}, u_{2,n}, u_{3,n})^T \in [-1, 1]^3$.
The functions $v_i(x) = e^{-25(x - c_i)^2}$ are fixed spatial profiles with
$c_1 = - \frac{\pi}{2}, c_2 = 0, c_3 = \frac{\pi}{2}$.
We consider periodic boundary conditions on
the spatial variable $x \in [-\pi, \pi]$,
and we discretize with a spatial mesh of $128$ points.

We consider a tracking problem involving one of the following two observables:
the mass $\int_X \eta(t, x)\mathrm{d} x$ and the momentum $\int_X
\eta^2(t,x)\mathrm{d} x$.
Given a reference trajectory $\{r_n\}$,
the tracking problem refers to a Bolza problem with $\Phi \equiv 0$
and $L_n(m, \mathbf{u}) = |m - r_n|^2$.

## Solving via PK-NN

Training data are generated from $1000$ trajectories of length $200$ samples.
The initial conditions are a convex combination of three fixed spatial profiles
and written as

$$ \eta(0, x) = b_1 e^{-(x - \frac{\pi}{2})^2} + b_2(- \sin(\frac{x}{2})^2) +
b_3 e^{-(x + \frac{\pi}{2})^2}, $$

with $b_i > 0$ and $\sum\limits_{i = 1}^3 b_i = 1$,
$b_i$'s are randomly sampled in $(0,1)$ with uniform distribution.
The training controls $u_i(t)$ are uniformly randomly generated in $[-1, 1]$.

The dictionary designed for the two tracking problems is of the form

$$ \Psi(\eta) = (1, \int_X \eta(t, x)\mathrm{d} x, \int_X \eta^2(t, x)\mathrm{d}
x, \mathrm{NN}(\eta))^T $$

with $3$ trainable elements.

