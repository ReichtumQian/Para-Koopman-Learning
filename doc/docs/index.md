
This repo implements the EDMD algorithm, EDMDDL algorithm and the learning of parametric Koopman decomposition.

## Get Started

To set up the environment, please refer to [[SetUp.md | Set up the environment]].
This repository is designed to be user-friendly.
You only need a few lines of code to run a simple example.
For a quick start, please check out the [[QuickStart.md | Quick Start]].
A few examples are provided in the `example` folder of this repository.

## User Guide

We support several advanced features, including: 

- Customizing your ODE:
- Customizing your ODE Solver:
- Speedup through GPU and "multi-threading": [[Parallel.md | Parallel Computing]].

## Class References

Most of the symbols in this documentation can be found at [1].

- Dictionary: [[Dictionary.md | Dictionary]], [[TrainableDictionary.md | TrainableDictionary]], [[RBFDictionary.md | RBFDictionary]].
- Koopman Operator: [[Koopman.md | Koopman]], [[ParamKoopman.md | ParamKoopman]].
- Solver: [[EDMDSolver.md | EDMDSolver]], [[EDMDDLSolver.md | EDMDDLSolver]], [[ParamKoopmanDLSolver.md | ParamKoopmanDLSolver]].
- ODEs: [[AbstractODE.md | AbstractODE]], [[DuffingOscillator.md | DuffingOscillator]], [[VanDerPolOscillator.md | VanDerPolOscillator]].
- Flow Map: [[FlowMap.md | FlowMap]], [[ForwardEuler.md | ForwardEuler]].
- Data Management: [[ODEDataSet.md | ODEDataSet]], [[ParamODEDataSet.md | ParamODEDataSet]].
- Neural Network: [[FullConnBaseNet.md | FullConnBaseNet]], [[FullConnNet.md | FullConnNet]], [[FullConnResNet.md | FullConnResNet]].


## References

[1] [Guo, Yue, Milan Korda, Ioannis G. Kevrekidis, and Qianxiao Li. "Learning Parametric Koopman Decompositions for Prediction and Control." arXiv preprint arXiv:2310.01124 (2023).](https://arxiv.org/abs/2310.01124)
