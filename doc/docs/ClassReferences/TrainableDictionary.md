
The `TrainableDictionary` class is a subclass of [[Dictionary.md|Dictionary]] class.
It contains a trainable neural network and a set of $N_y$ non-trainable observable functions 
(see [[ObservableFunction.md|ObservableFunction]]).
The dictionary can be represented by the mapping
$\Psi: \mathbb{R}^{N \times N_x} \rightarrow \mathbb{R}^{N \times N_{\psi}}$.

!!! Note
	The constant observable function $\mathbf{1}$ is included in the `TrainableDictionary` by default,
	so users don't need to define it explicitly.

## API Documentation

::: PKoopmanDL.TrainableDictionary