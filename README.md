[![Build Status](https://travis-ci.com/lsawade/GCMT3D.svg?branch=devel)](https://travis-ci.com/lsawade/GCMT3D)

# GCMT3D

This is the initial commit for the revisited global cmt solution to allow for 3D 
velocity structures.


## Installation

In order to install GCMT3D we first need to create a new conda environment. Then we install the dependencies, and finally GCMT3D itself. The required steps are the following

#### 1. Create and activate a new conda environment with python 3.7

```bash
conda create -n <your_environment> python=3.7
conda activate <your_environment>
```

#### 2. Install dependencies:

```bash
# Install basemap using conda
conda config --add channels conda-forge
conda install basemap

# Install other requirements using pip
pip install -r requirements.txt
```

#### 3. Download and Install GCMT3D

```bash
# Download GCMT3D
git clone https://github.com/lsawade/GCMT3D.git

# Install GCMT3D in the current conda environment
cd GCMT3D/ && pip install .
```

Test that GCMT3D was successfully installed by running the following command

```bash
python -c "import gcmt3d"
```

## Travis automatic testing

The Travis automatic system is setup to load the latest anaconda release for 
Linux. Then, in a Python 3.7 environment, all required software is downloaded 
and installed. For the installation steps, please refer to the `.travis.yml` in 
the main directory. It mainly follows the steps in the quick install.

