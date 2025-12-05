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
