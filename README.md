[![Build Status](https://travis-ci.com/lsawade/GCMT3D.svg?branch=master)](https://travis-ci.com/lsawade/GCMT3D)

# GCMT3D

This is the initial commit for the revisited global cmt solution to allow for 3D 
velocity structures.


## Quick install

1. Create conda environment with python 3.7
2. Run the following lines:
```bash
conda activate <your_environent>
conda config --add channels conda-forge
git clone --branch devel https://github.com/lsawade/GCMT3D.git
cd GCMT3D
pip install numpy
conda install obspy
conda install basemap
pip install -r requirements.txt
pip install .
```

## Elaborate install

First, create a conda environment 
(https://docs.conda.io/projects/conda/en/latest/index.html) where you install 
numpy using either `$ pip install numpy` or `$ conda install numpy`. This is 
necessary for the obspy installation.

```bash
conda config --add channels conda-forgey
conda install obspy
```

You will also need to install `basemap` via `conda install basemap`. Sometimes,
this is done automatically when obspy is installed, but that is not always the
case.

As of now, you will need some requisites that aren't available through pip these 
requisites can be found in my repository:

* https://github.com/lsawade/spaceweight.git
* https://github.com/lsawade/pycmt3d.git

These can be installed using

```bash
git clone https://github.com/lsawade/spaceweight.git
cd spaceweight
pip install .
```

for space weight and

```bash
git clone --branch devel https://github.com/lsawade/pycmt3d.git
cd pycmt3d
pip install .
```

for pycmt3d. Note that these are different from Wenjie's repository as these are
fixed to accommodate Also make sure your obspy is up-to-date. I have encountered
some issues with basemap and an environment variable called `PROJ_LIB`, but I 
haven't been able to reproduce the error.

At last, install the development branch of this repository 
https://github.com/lsawade/GCMT3D.git:

```bash
git clone --branch devel https://github.com/lsawade/GCMT3D.git
cd GCMT3D
pip install .
```

## Travis automatic testing

The Travis automatic system is setup to load the latest anaconda release for 
Linux. Then, in a Python 3.7 environment, all required software is downloaded 
and installed. For the installation steps, please refer to the `.travis.yml` in 
the main directory. It mainly follows the steps in the quick install.

