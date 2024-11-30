
We strongly encourage you to look through the examples in `example/simple examples` directory. 
We provide three simple examples using the EDMD algorithm, the EDMDDL algorithm and the parametric Koopman learning algorithm.

Each example consists of a JSON file and a jupyter notebook. 
Let's consider the parametric Koopman learning algorithm as an example.

## Train the Parametric Koopman Operator

```python
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

## Generating the Trajectories

In our package, we treat the Koopman operator as a specialized
transition function.
Specifically, a `Dictionary` combined with a `Koopman` operator forms a discrete dynamical system:

$$ \Psi_{n+1} = \mathcal{K} \Psi_n. $$

A potentially tricky aspect is that when the full-state observable
functions are included in the dictionary,
the Koopman operator can be used to predict the state
through the sequence

$$ x_n \rightarrow \Psi_n \rightarrow \Psi_{n+1} \rightarrow x_{n+1}. $$

To facilitate this process,
we have designed a dedicated class [[KoopmanDynamics.md | KoopmanDynamics]].
This class is designed to predict state trajectories 
starting from an initial state `x0`.
To use it, 
simply create a `KoopmanDynamics` and call its `traj` method:

```python
state_pos = [0, 1] # the position of the state in the observable function
state_dim = 2
koopman_dynamics = pkdl.KoopmanDynamics(K, solver.dictionary, state_pos, state_dim)
# the output is of the form (N, traj_len, number of state),
# where N is the number of different initial states
p = koopman_dynamics.traj(x0, param, solver.traj_len)
```





