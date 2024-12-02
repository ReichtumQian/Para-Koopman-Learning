
The class `Koopman` is a mapping $K: \mathrm{span}(\Psi) \rightarrow \mathrm{span}(\Psi)$,
which acts as the finite-dimensional approximation of the Koopman operator $\mathcal{K}$,
i.e.,

$$ \mathcal{K} \Psi \approx K \Psi. $$

Given $\phi = \mathbf{a}^T \Psi$, where $\mathbf{a} \in \mathbb{R}^{N_{\psi}}$,
then the application of $\mathcal{K}$ satisfies

$$ \mathcal{K} \phi = \mathcal{K} \mathbf{a}^T \Psi \approx \mathbf{a}^T K \Psi. $$

!!! info
    Understanding the `__call__` method of `Koopman`: Given the data set $\{x^{(n)}\}_{n = 1}^N$,
    it represents a mapping:

    $$ \left[
      \begin{array}{cccc}
        \psi_1(x^{(1)})&\psi_2(x^{(1)})&\cdots&\psi_{N_{\psi}}(x^{(1)})\\
        \psi_1(x^{(2)})&\psi_2(x^{(2)})&\cdots&\psi_{N_{\psi}}(x^{(2)})\\
        \vdots&\vdots&\ddots&\vdots\\
        \psi_1(x^{(N)})&\psi_2(x^{(N)})&\cdots&\psi_{N_{\psi}}(x^{(N)})\\
      \end{array}
    \right] \rightarrow \left[
      \begin{array}{cccc}
        \mathcal{K}\psi_1(x^{(1)})&\mathcal{K}\psi_2(x^{(1)})&\cdots&\mathcal{K}\psi_{N_{\psi}}(x^{(1)})\\
        \mathcal{K}\psi_1(x^{(2)})&\mathcal{K}\psi_2(x^{(2)})&\cdots&\mathcal{K}\psi_{N_{\psi}}(x^{(2)})\\
        \vdots&\vdots&\ddots&\vdots\\
        \mathcal{K}\psi_1(x^{(N)})&\mathcal{K}\psi_2(x^{(N)})&\cdots&\mathcal{K}\psi_{N_{\psi}}(x^{(N)})\\
      \end{array}
    \right]$$

## API Documentation

::: PKoopmanDL.Koopman


