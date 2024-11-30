
The class `FlowMap` implements the flow map $\varphi_t(x) := x(t), x(0) = x$.
This class serves as a base class for various numerical solvers
for initial value problems (IVPs).

## Attributes

- `_t_step` (float): The time step size of the trajectory.
- `_dt` (float): The time step size of the flow map.

## Methods

- `__init__(self, dt)`
- `step(self, ode, x, u)`: 
    - `ode` (AbstractODE): The ODE object.
    - `x` (tensor): The state vector.
    - `u` (tensor): The parameters input.
    - Returns (tensor): The next state vector.

