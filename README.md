# run-esmfold
Procedures and scripts to run ESMFold on various environments

### How to install
* [Apple silicon](doc/install_apple.md)
* [Windows 64-bit](doc/install_win.md)
* Linux 64-bit: to do
* [Nibbler HPC Cluster](doc/install_nibbler.md)

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

Folding the 1EZG example on a AMD Ryzen 5 5600X 6 cores @ 3.70GHz Windows PC with 16GB RAM and 8GB VRAM (RTX 3070) via WSL2 Ubuntu-22.04
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

### Batch run example

```
cd ~/git/run-esmfold
source ~/miniforge3/bin/activate
conda activate esmfold-env 
python src/run_esmfold.py esp-esp --data-root /mnt/c/Users/joeri/git/esp-esp/data/protein-folding/mane-select --bin 201-300
python src/run_esmfold.py esp-esp --data-root /mnt/c/Users/joeri/git/esp-esp/data/protein-folding/mane-plus-clinical --bin 201-300
```


### Benchmarks

Larger proteins require GPU to fold in reasonable time. However, if you have more RAM than VRAM, you can (slowly) fold proteins in CPU mode that are impossible in GPU mode. Case in point, folding the first 800AA of COL7A1 (UniProt Q02388) on a Intel i7-4790K 4 cores @ 4.00GHz Windows PC with 32GB RAM via WSL Ubuntu-22.04:
```
CPU mode
Model load time: 40.30 sec
Inference time: 11640.37 sec
Total time: 11680.67 sec
```

To get an impression of time scaling, we try different residue lengths on the same hardware. Folding the first n residues of COL7A1 (UniProt Q02388) on a AMD Ryzen 5 5600X 6 cores @ 3.70GHz Windows PC with 16GB RAM and 8GB VRAM (RTX 3070) via WSL2 Ubuntu-22.04:

|   n |    CPU   |   GPU   |
|----:|---------:|--------:|
| 100 |  115.22  |  36.33  |
| 200 |  213.63  | 217.22  |
| 300 |  420.63  |  fail   |
| 400 |  763.71  |  fail   |
| 500 | 1266.67  |  fail   |
| 600 | 1896.81  |  fail   |
| 700 | 3602.74  |  fail   |
| 800 |   fail   |  fail   |
