[![Build Status](https://travis-ci.com/lsawade/GCMT3D.svg?branch=master)](https://travis-ci.com/lsawade/GCMT3D)

# GCMT3D

This is the initial commit for the revisited global cmt solution to allow for 3D 
velocity structures.

## Setup

As of now you know some requisites that aren't available through pip these 
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
fixed to accomodate Also make sure your obspy is up-to-date. I have encountered some
issues with basemap and an environment variable called `PROJ_LIB`, but I haven't 
been able to reproduce the Error.

At last, install the development branch of this repository 
https://github.com/lsawade/GCMT3D.git:

```bash
git clone --branch devel https://github.com/lsawade/GCMT3D.git
cd GCMT3D
pip install .
```
