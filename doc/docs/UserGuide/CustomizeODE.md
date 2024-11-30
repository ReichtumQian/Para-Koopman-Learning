
We use factory pattern to manage ODEs.
To customize an ODE, you can create a new class that inherits from the [[AbstractODE.md | Abstract]] class and then register it.
Here's an example:

```python
import PKoopmanDL as pkdl

class MyODE(pkdl.AbstractODE):
  def rhs(self, x, u):
    dim = 2 # state dim
    param_dim = 2 # parameter dim

    def func():
      return x + u

    super().__init__(dim, param_dim, func)

# Register the ODE
pkdl.register_ode('MyODE', MyODE)
```

Once registered, you can use the custom ODE 
in the same way as the built-in ODEs by
specifying `"MyODE"` in your configuration JSON file.






