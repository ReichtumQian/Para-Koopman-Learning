
We use factory pattern to manage ODE solvers.
To customize an ODE solver, 
you can create a new class that inherits from the [[ODESolver.md | ODESolver]]
class and then register it.
Your class must override the `step` method.
Here's an example:

```python
import PKoopmanDL as pkdl

class MyODESolver(pkdl.ODESolver):
  def step(self, x, u):
    n_step = int(self._t_step / self._dt)
    for _ in range(n_step):
      x = x + self._dt * self._ode.rhs(x, u)
    return x

# Register the ODE
pkdl.register_ode_solver('MyODESolver', MyODESolver)
```

Once registered, you can use the custom ODE solver
in the same way as the built-in ODEs by
specifying `"MyODESolver"` in your configuration JSON file.