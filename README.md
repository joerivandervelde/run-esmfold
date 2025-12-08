# run-esmfold
Procedures and scripts to run ESMFold on various environments

### How to install
* [Apple silicon](doc/install_apple.md)
* [Windows 64-bit](doc/install_win.md)
* Linux 64-bit: to do

### Test run


Fold the 1EZG antifreeze protein:
```
cd ~/git/run-esmfold
source ~/miniforge3/bin/activate
conda activate esmfold-env 
python src/run_esmfold.py --sequence "QCTGGADCTSCTGACTGCGNCPNAVTCTNSQHCVKANTCTGSTDCNTAQTCTNSKDCFEANTCTDSTNCYKATACTNSSGCPGH" --output out/1EZG.pdb
```

Folding is non-deterministic here, so fold another one and compare the two with [TM-align](https://github.com/joerivandervelde/TM-align_mass)
```
python src/run_esmfold.py --sequence "QCTGGADCTSCTGACTGCGNCPNAVTCTNSQHCVKANTCTGSTDCNTAQTCTNSKDCFEANTCTDSTNCYKATACTNSSGCPGH" --output out/1EZG-alt.pdb
../TM-align_mass/bin/TMalign out/1EZG.pdb out/1EZG-alt.pdb
```
You might get something like this
```
TM-score= 0.99972
```

Running the 1EZG example on a Windows PC with 16GB RAM and 8GB VRAM (RTX 3070) via WSL2 Ubuntu-22.04:
```
CPU mode
Model load time: 98.67 sec
Inference time: 130.93 sec
Total time: 229.60 sec

GPU mode (cuda)
Model load time: 165.74 sec
Inference time: 49.58 sec
Total time: 215.33 sec
```

Running the first 800AA of COL7A1 (UniProt Q02388) on a Intel i7-4790K 4 cores @ 4GHz Windows PC with 32GB RAM via WSL Ubuntu-22.04:
```
CPU mode
Model load time: 40.30 sec
Inference time: 11640.37 sec
Total time: 11680.67 sec
```
