# Implementation of Cascaded-Regression based Methods for Facial Feature Points Detection
This repository contains the implementation of my final individual project at Imperial College London. Its objective was to study and implement the following methods for facial feature point detection ("face alignment"):
* Face Alignment by Explicit Shape Regression by Cao et al from CVPR 2012 (ESR)
* One millisecond face alignment with an ensemble of regression trees by Kazemi et al from CVPR 2014 (ERT)
* Face Alignment at 3000 FPS via Regressing Local Binary Features by Ren et al from CVPR 2014 (LBF)

All three methods (ESR, ERT and LBF) are implemented in in the *facefit* Python package.

## Installation
Install conda.
```bash
curl -s https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh | bash

# Create and activate a new conda environment
conda create -n face-alignment python -y
source activate face-alignment

# Install dependencies.
conda install -y -c menpo menpo menpofit menpodetect 
conda install -y matplotlib numpy opencv python=2.7.9=1 pyzmq scikit-learn scipy zlib
apt-get install libhdf5-dev -y
pip install hickle

# Build liblinear.
cd beng/facefit/external/liblinear/python
make
```

## Example usage
### Training and testing a model
It is possible to train and test models using the IPython notebooks included in the experiments/ directory. Note that for all three methods, training currently takes at least a couple of hours. 

Serialized (hickled) models can be found here:
https://www.dropbox.com/s/rsrq37bl4h254f4/models_lfpw_helen.zip?dl=0
All three models are trained on a dataset consiting of training images from a combination of the LFPW and Helen datasets. For more information on usage, consult the notebooks in experiments/.

### (Almost) Real-time webcam face fitting
```python
python clients/webcam_fitter.py path_to_hickled_model
```
e.g.
```python
python clients/webcam_fitter.py models/ert_lfpw_helen.hkl
```




