# Steps to install on Apple silicon (ARM64)

### Download Miniforge3 and install in default location
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
chmod +x Miniforge3-MacOSX-arm64.sh
./Miniforge3-MacOSX-arm64.sh
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

### Clone this repository and step into its directory
```
git clone https://github.com/joerivandervelde/run-esmfold.git
cd ~/git/run-esmfold
```

### Set up new environment and activate with the function (not binary)
```
~/miniforge3/bin/conda env create -f env/environment.yml
conda activate esmfold-env
```

### Check that the correct Python was installed (i.e. not arm64 but x86_64)
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
If you see `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.`, first run:
```
export KMP_DUPLICATE_LIB_OK=TRUE
```
and check again, you should see something like this:
```
torch: 2.9.1
cuda: False
mps: True
ESMFold loaded ok
```

The models are located at `~/.cache/huggingface/hub/models--facebook--esmfold_v1/blobs/` and the environment at `~/miniforge3/envs/esmfold-env/`.
