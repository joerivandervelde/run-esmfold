# Steps to install on Microsoft Windows x86_64

Running on native Windows is technically possible but using WSL is highly recommended to avoid many problems.

### Install Ubuntu on WSL/WSL2 for Windows

Here we used Ubuntu 22.04.5 LTS (GNU/Linux 4.4.0-19041-Microsoft x86_64). Every command in these steps is executed inside the WSL/WSL2 system.

### Download Miniforge3 and install in default location
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
chmod +x Miniforge3-Linux-x86_64.sh
./Miniforge3-Linux-x86_64.sh
```

### Check installation and activate base environment ('conda' is now a function)
```
~/miniforge3/bin/conda --version
source ~/miniforge3/bin/activate
which conda
conda --version
```

### Make sure we start clean ("Not a conda environment")
```
~/miniforge3/bin/conda env remove -n esmfold-env
```

### Switch to location of this repo, e.g.
```
cd /mnt/d/github/run-esmfold/
```

### Set up new environment and activate with the function (not binary)
```
~/miniforge3/bin/conda env create -f env/environment.yml
conda activate esmfold-env
```

### Check that the correct Python was installed (i.e. x86_64)
```
python -c "import platform; print(platform.platform()); print(platform.machine())"
```

### Check that pip is also part of this environment
```
which pip
```

### Upgrade Torch version manually within this environment
```
pip uninstall -y torch torchvision torchaudio
pip install "torch>=2.6.0" torchvision torchaudio
```

### Check if Torch, CUDA and/or MPS are installed and download ESMFold
```
python - << 'EOF'
import torch
from transformers import EsmForProteinFolding

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("mps:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
print("ESMFold loaded ok")
EOF
```
and check again, you should see something like this:
```
torch: 2.9.1+cu128
cuda: True
mps: False
ESMFold loaded ok
```

CUDA passthrough requires WSL2, so using WSL limits folding to CPU only.

The models are located at ` ~/.cache/huggingface/hub/models--facebook--esmfold_v1/blobs/` and the environment at `~/miniforge3/envs/esmfold-env/`.
