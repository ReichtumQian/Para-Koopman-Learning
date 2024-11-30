
We strongly encourage you to look through the examples in `example/simple examples` directory. 
We provide three simple examples using the EDMD algorithm, the EDMDDL algorithm and the parametric Koopman learning algorithm.

Each example consists of a JSON file and a jupyter notebook. 
Let's consider the parametric Koopman learning algorithm as an example.
The main train code includes the following:

``` python
import PKoopmanDL as pkdl

config_file = "ParamKoopman.json"
tmp_func = lambda x: x
observable_func = pkdl.ObservableFunction(tmp_func, 2)
solver = pkdl.ParamKoopmanDLSolverWrapper(config_file)
solver.setup(observable_func)
K = solver.solve()
```

- `config_file`: Specifies the JSON input file containing configuration parameters.
- `observable_func`: Defines the observable functions and their output dim.
  In this example, we use a full-state observable function.
- `ParamKoopmanSolverWrapper`: Takes the configuration file and setup the solver automatically.
- `K`: Represents the learned parametric Koopman operator.

More information about the input files and the solvers can be found at [[InputGuide.md | Input Guide]].
