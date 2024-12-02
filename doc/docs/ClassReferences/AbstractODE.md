
The class `AbstractODE` is an abstract class for ordinary differential equations (ODEs)
of the form

$$ \dot{\mathbf{x}}(t) = \mathbf{f} (\mathbf{x}(t), \mathbf{u}). $$

!!! info
    Since the `rhs` function must accommodate both fixed parameters and variable parameters,
    it should be capable of handling inputs of the form 
    $\mathbb{R}^{N \times N_x} \times \mathbb{R}^{1 \times N_u}$ (fixed parameters) and
    $\mathbb{R}^{N \times N_x} \times \mathbb{R}^{N \times N_u}$ (variable parameters).

## API Documentation

::: PKoopmanDL.AbstractODE
