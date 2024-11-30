
Pytorch implementation of EDMDDL [1] and learning parametric Koopman decomposition [2]. Comprehensive documentation can be found [here](https://reichtumqian.github.io/ParamKoopmanDL/), examples can be found in the `example` folder.

## Quick Start

You can set up the environment and install this package by running the following commands:

``` bash
git clone https://github.com/ReichtumQian/ParamKoopmanDL.git
cd ParamKoopmanDL

# (Optional) Create a new environment
conda create -n KoopmanDL
conda activate KoopmanDL

# By default using CPU
pip install -r requirements.txt
pip install -e .
```

## References

[1] [Li, Q., Dietrich, F., Bollt, E. M., & Kevrekidis, I. G. (2017). Extended dynamic mode decomposition with dictionary learning: A data-driven adaptive spectral decomposition of the Koopman operator. Chaos: An Interdisciplinary Journal of Nonlinear Science, 27(10), 103111.](https://aip-scitation-org.libproxy1.nus.edu.sg/doi/full/10.1063/1.4993854)

[2] [Guo, Yue, Milan Korda, Ioannis G. Kevrekidis, and Qianxiao Li. "Learning Parametric Koopman Decompositions for Prediction and Control." arXiv preprint arXiv:2310.01124 (2023).](https://arxiv.org/abs/2310.01124)