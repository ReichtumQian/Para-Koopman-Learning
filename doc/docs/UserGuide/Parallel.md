
This repository supports parallel computing through `CUDA` 
and "multi-threading" using the [joblib](https://joblib.readthedocs.io/en/stable/) library.

## Using Multi-threading

We utilize `joblib` to take advantage of multiple cores of the CPU,
primarily used to accelerate the process of solving ODEs with `Scipy`.
To set or get the number of threads, use the following code:

``` python
import PKoopmanDL

# get the number of threads
n_jobs = PKoopmanDL.get_n_jobs()

# set the number of threads
PKoopmanDL.set_n_jobs(n_jobs)

# use all available cores
PKoopmanDL.set_n_jobs(-1) 
```

## Using CUDA

CUDA is employed to accelerate neural network training,
supported by `pytorch`.
We use `torch.cuda` to manage the GPU resources,
and relevant information is displayed when the package is loaded.
