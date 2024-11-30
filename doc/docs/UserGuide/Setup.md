
We highly recommend using `Anaconda` to manage your Python environment for this project.

## Using CPU

You can follow these steps to set up an environment that uses the CPU:

``` bash
git clone https://github.com/ReichtumQian/ParamKoopmanDL.git
cd ParamKoopmanDL
# Create a new environment
conda create -n KoopmanDL python=3.8
conda activate KoopmanDL
# By default using CPU
pip install -r requirements.txt
```

## Using GPU

1. Install PyTorch with CUDA support:

``` bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

2. Install the remaining dependencies using either `pip` or `conda`:

``` bash
pip install matplotlib numpy scipy sqdm
```

