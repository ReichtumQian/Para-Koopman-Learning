import PKoopmanDL as pkdl
import numpy as np
import matplotlib.pyplot as plt
import torch

nontrain_func = lambda x: torch.cat([torch.ones((x.shape[0], 1)), x], dim=1)

# ParamKoopman

config_file = "paramkoopman100-100.json"
param_solver1 = pkdl.ParamKoopmanDLSolverWrapper(config_file)
param_solver1.setup(nontrain_func)
param_K1 = param_solver1.solve()
param_K1.save("data/paramkoopman100-100_koopman.pt")
param_solver1.dictionary.save("data/paramkoopman100-100_dict.pt")

config_file = "paramkoopman500-20.json"
param_solver2 = pkdl.ParamKoopmanDLSolverWrapper(config_file)
param_solver2.setup(nontrain_func)
param_K2 = param_solver2.solve()
param_K2.save("data/paramkoopman500-20_koopman.pt")
param_solver2.dictionary.save("data/paramkoopman500-20_dict.pt")

config_file = "paramkoopman1000-10.json"
param_solver3 = pkdl.ParamKoopmanDLSolverWrapper(config_file)
param_solver3.setup(nontrain_func)
param_K3 = param_solver3.solve()
param_K3.save("data/paramkoopman1000-10_koopman.pt")
param_solver3.dictionary.save("data/paramkoopman1000-10_dict.pt")
