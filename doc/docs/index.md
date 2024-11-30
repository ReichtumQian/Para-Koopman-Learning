
This repo implements the EDMD algorithm, EDMDDL algorithm [1] and the learning of parametric Koopman decomposition [2].

## Get Started

To set up the environment, please refer to [[SetUp.md | Set up the environment]].
This repository is designed to be user-friendly.
You only need a few lines of code to run a simple example.
For a quick start, please check out the [[QuickStart.md | Quick Start]].
A few examples are provided in the `example` folder of this repository.

## User Guide

- **Customizing Input Parameters and Solver**: [[InputGuide.md | Input Guide]].
- **Customizing Observable Functions**: [[CustomizeObservable.md | Custom Observable Guide]].
- **Customizing ODE**: [[CustomizeODE.md | Custom ODE Guide]].
- **Customizing ODE Solver**: [[CustomizeSolver.md | Custom ODE Solver Guide]].
- **Enhancing Performance with GPU and Multithreading**: [[Parallel.md | Parallel Computing Guide]].

## Class References

Most of the symbols in this documentation can be found at [2].

- Dictionary: [[Dictionary.md | Dictionary]], [[TrainableDictionary.md | TrainableDictionary]], [[RBFDictionary.md | RBFDictionary]].
- Koopman Operator: [[Koopman.md | Koopman]], [[ParamKoopman.md | ParamKoopman]].
- Solver: [[EDMDSolver.md | EDMDSolver]], [[EDMDDLSolver.md | EDMDDLSolver]], [[ParamKoopmanDLSolver.md | ParamKoopmanDLSolver]].
- ODEs: [[AbstractODE.md | AbstractODE]], [[DuffingOscillator.md | DuffingOscillator]], [[VanDerPolOscillator.md | VanDerPolOscillator]].
- ODE Solver: [[FlowMap.md | FlowMap]], [[ForwardEuler.md | ForwardEuler]].
- Data Management: [[ODEDataSet.md | ODEDataSet]], [[ParamODEDataSet.md | ParamODEDataSet]].
- Neural Network: [[FullConnBaseNet.md | FullConnBaseNet]], [[FullConnNet.md | FullConnNet]], [[FullConnResNet.md | FullConnResNet]].


## References

[1] [Li, Q., Dietrich, F., Bollt, E. M., & Kevrekidis, I. G. (2017). Extended dynamic mode decomposition with dictionary learning: A data-driven adaptive spectral decomposition of the Koopman operator. Chaos: An Interdisciplinary Journal of Nonlinear Science, 27(10), 103111.](https://aip-scitation-org.libproxy1.nus.edu.sg/doi/full/10.1063/1.4993854)

[2] [Guo, Yue, Milan Korda, Ioannis G. Kevrekidis, and Qianxiao Li. "Learning Parametric Koopman Decompositions for Prediction and Control." arXiv preprint arXiv:2310.01124 (2023).](https://arxiv.org/abs/2310.01124)
