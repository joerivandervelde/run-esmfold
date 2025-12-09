# Steps to install on Nibbler HPC cluster

The [Nibbler High Performance Compute cluster](https://docs.gcc.rug.nl/nibbler/) is part of the league of robots - a collection of HPC clusters, which are named after robots. Deployment and functional administration of all clusters is a joined effort of the Genomics Coordination Center (GCC) and the Center for Information Technology (CIT) from the University Medical Center and University of Groningen, in collaboration with and as part of several research projects. Thanks to Joel for his [VTM Nibbler manual](https://github.com/joelkuiper/variable-taxon-mapper/blob/main/doc/nibbler_cluster.md) for reference to create this one.

### Login to Nibbler and check available disk space
```
ssh tunnel+nibbler
module load cluster-utils
quota
```

### Create personal working directory (change user and group accordingly)
```
mkdir -p /groups/umcg-gcc/tmp02/users/umcg-jvelde
```

### Add working directory as an environment variable in `~/.bashrc`:
```
export WORKDIR=/groups/umcg-gcc/tmp02/users/umcg-jvelde
```

### Reload bash and test workspace location
```
source ~/.bashrc
echo $WORKDIR
ls $WORKDIR
```

### Create ESMFold directory that will contain everything
```
mkdir -p "$WORKDIR/ESMFold"
cd $WORKDIR/ESMFold
```

### Create big tmp location
```
mkdir -p "$WORKDIR/ESMFold/tmp"
```

### Also tmp and pip cache locations in `~/.bashrc`:
```
export TMPDIR=$WORKDIR/ESMFold/tmp
export PIP_CACHE_DIR=$WORKDIR/ESMFold/pip_cache
```

### Reload bash and test locations
```
source ~/.bashrc
echo $TMPDIR
echo $PIP_CACHE_DIR
```


### Download Miniforge3 and install in specific location
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
chmod +x Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $WORKDIR/ESMFold/miniforge
```

### Check installation and activate base environment ('conda' is now a function)
```
$WORKDIR/ESMFold/miniforge/bin/conda --version
source $WORKDIR/ESMFold/miniforge/bin/activate
which conda
conda --version
```

### Make sure we start clean ("Not a conda environment")
```
$WORKDIR/ESMFold/miniforge/bin/conda env remove -n esmfold-env
```

### Clone this repository and step into its directory
```
git clone https://github.com/joerivandervelde/run-esmfold.git
cd run-esmfold
```

### Set up new environment and activate with the function (not binary)
```
$WORKDIR/ESMFold/miniforge/bin/conda env create -f env/environment.yml
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

### Add Hugging Face cache location as environment variables in your `~/.bashrc`:
```
export HF_HOME=$WORKDIR/ESMFold/hf_cache
export TRANSFORMERS_CACHE=$WORKDIR/ESMFold/hf_cache/models
```

### Reload bash and test workspace location
```
source ~/.bashrc
echo $HF_HOME
echo $TRANSFORMERS_CACHE
```

### Load CUDA
```
ml CUDA
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

### Fold 1EZG as a quick test
```
cd cd $WORKDIR/ESMFold/run-esmfold
source $WORKDIR/ESMFold/miniforge/bin/activate
conda activate esmfold-env 
python src/run_esmfold.py --sequence "QCTGGADCTSCTGACTGCGNCPNAVTCTNSQHCVKANTCTGSTDCNTAQTCTNSKDCFEANTCTDSTNCYKATACTNSSGCPGH" --output out/1EZG.pdb
```

### todo, sbatch job
```
#!/usr/bin/env bash
#SBATCH --job-name=esmfold
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:05:00
```
