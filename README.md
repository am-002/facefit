# Implementation of regression-based face alignment techniques
This repository contains the implementation of my final individual project at Imperial College London.

It includes the implementation of the following face alignment methods:
* Explicit Shape Regression (ESR) by Cao et al.
* One millisecond Face Alignment by an Ensemble of Regression Trees (ERT)
* Face Alignment at 3000fps via regressing Local Binary Features (LBF)

# Installation
Install conda.
```bash
curl -s https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh | bash
```
Create and activate a new conda environment
```bash
conda create -n face-alignment python
source activate face-alignment
```

Install dependencies.
```bash
conda install -c menpo menpo, menpofit, menpodetect
conda install matplotlib numpy opencv python=2.7.9=1 pyzmq scikit-learn scipy zlib
pip install hickle
```
Installing IPython notebook.
```bash
conda update ipython ipython-notebook
```

