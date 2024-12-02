
A `Dictionary` is a vector-valued function denoted as

$$\Psi = (\psi_1, \psi_2, \cdots, \psi_{N_\psi})^T,$$

where each component $\psi_i = g_i$ for $i = 1,\cdots,N_y$.
The functions $g_{1},\cdots, g_{N_{y}}$ represent the target observable functions.
It supports batched operation
$\Psi: \mathbb{R}^{N \times N_x} \rightarrow \mathbb{R}^{N \times N_{\psi}}$.


## API Documentation

::: PKoopmanDL.Dictionary

