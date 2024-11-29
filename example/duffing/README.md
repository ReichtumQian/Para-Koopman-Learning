
## Testing the Parametric Duffing Equation

In this example, we consider the parametric Duffing equation:

$$ \begin{align*}
	&\dot{x}_{1} = x_{2},\\
	&\dot{x}_{2} = - \delta x_{2} - x_{1}(\beta + \alpha x_{1}^{2})
\end{align*} $$

Our goal is to compare `ParamKoopmanDLSolver`, `EDMDSolver`, and `EDMDDLSolver`.
Each trajectory is generated using time-independent parameters $\mathbf{u} = (\delta, \alpha, \beta)$.

## Parameters and Dataset Configuration

- $\nu_{1}$: This denotes the number of trajectories for each set of fixed parameters.
It is referred to as `n_traj_per_param` in the JSON configuration files.
- $\nu_{2}$: This represents the number of different parameter configurations in the dataset.
It is calculated as `n_traj/n_traj_per_param` in the JSON files.

For example, a file named `100-100` indicates $\nu_{1} = 100, \nu_{2} = 100$,
while `500-20` indicates $\nu_{1} = 500, \nu_{2} = 20$.

## Instructions for Running the Code

- Create a `data` folder in the current directory.
- Execute `train.py` to generate the dataset and train the model.
- Use the provided Jupyter notebooks (`ipynb` files) to plot the predictions.

