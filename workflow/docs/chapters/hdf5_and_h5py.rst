
..
    Define roles below

.. role:: bash(code)
   :language: bash

.. role:: python(code)
    :language: python

Installation of hdf5 and h5py
=============================

To enable processing of the ASDF files, specifically windowing of the data,
hdf5 and and h5py need to be installed on your machine.


:bash:`hdf5` on the remote cluster
+++++++++++++++++++++++++++++++++++++

On remote clusters hdf5 is usually precompiled and you don't have to worry
about compilation. On the Princeton Tiger Cluster, the command to load a
parallel compiled version of hdf5 would be:

.. code:: bash

    module load hdf5/intel-17.0/openmpi-1.10.2/1.10.0

Usually they are easy to identify since they will have either been configured
with :bash:`openmpi` or :bash:`intel-mpi`. Hence, when one checks for available
modules using :bash:`module load`. Then, the exact location of the installation
can be found using the following line.

.. code :: bash

    module show hdf5/intel-17.0/openmpi-1.10.2/1.10.0

On the Princeton Tiger Cluster the output would look somewhat like this:

.. code :: bash

    -------------------------------------------------------------------
    /usr/local/share/Modules/modulefiles/hdf5/intel-17.0/openmpi-1.10.2/1.10.0:

    module-whatis	 Sets up hdf5 {version} for intel-17.0 openmpi-1.10.2 in your environment
    prepend-path	 PATH /usr/local/hdf5/intel-17.0/openmpi-1.10.2/1.10.0/bin
    prepend-path	 LD_LIBRARY_PATH /usr/local/hdf5/intel-17.0/openmpi-1.10.2/1.10.0/lib64
    prepend-path	 LIBRARY_PATH /usr/local/hdf5/intel-17.0/openmpi-1.10.2/1.10.0/lib64
    prepend-path	 MANPATH /usr/local/hdf5/intel-17.0/openmpi-1.10.2/1.10.0/share/man
    setenv		 HDF5DIR /usr/local/hdf5/intel-17.0/openmpi-1.10.2/1.10.0
    append-path	 -d   LDFLAGS -L/usr/local/hdf5/intel-17.0/openmpi-1.10.2/1.10.0/lib64
    append-path	 -d   INCLUDE -I/usr/local/hdf5/intel-17.0/openmpi-1.10.2/1.10.0/include
    append-path	 CPATH /usr/local/hdf5/intel-17.0/openmpi-1.10.2/1.10.0/include
    append-path	 -d   FFLAGS -I/usr/local/hdf5/intel-17.0/openmpi-1.10.2/1.10.0/include
    append-path	 -d   LOCAL_LDFLAGS -L/usr/local/hdf5/intel-17.0/openmpi-1.10.2/1.10.0/lib64
    append-path	 -d   LOCAL_INCLUDE -I/usr/local/hdf5/intel-17.0/openmpi-1.10.2/1.10.0/include
    append-path	 -d   LOCAL_CFLAGS -I/usr/local/hdf5/intel-17.0/openmpi-1.10.2/1.10.0/include
    append-path	 -d   LOCAL_FFLAGS -I/usr/local/hdf5/intel-17.0/openmpi-1.10.2/1.10.0/include
    append-path	 -d   LOCAL_CXXFLAGS -I/usr/local/hdf5/intel-17.0/openmpi-1.10.2/1.10.0/include
    -------------------------------------------------------------------

The first line shows the location of the installation:
:bash:`/usr/local/share/Modules/modulefiles/hdf5/intel-17.0/openmpi-1.10.2/1
.10.0`. This is important for the next step which is the installation of
:python:`h5py` which in turn is necessary to use parallel processing of the
:python:`ASDF` data sets.


Installation of :python:`h5py`
++++++++++++++++++++++++++++++


To install :python:`h5py` on the cluster, the precompiled version of mpi4py
from the server has to be loaded. Make sure it isn't installed beforehand!
Simply

.. code :: bash

    module load mpi4py

After this is load we can install h5py in the parallel version. For this, the
python source has to be downloaded because the :bash:`pip install <package>`
would install the package. Downloading to the current directory and unpacking
is done using:

.. code :: bash

    pip download --no-binary=h5py h5py
    tar -xvf h5py-?.*.tar.gz
    cd h5py-?.*

Then, make sure that you have your favorite :bash:`mpicc` compiler at hand.
Meaning, re-module load an existing mpi compiler because the conda environment
overwrites the system compiler.

.. code :: bash

    # E.g.
    module load openmpi

Afterwards, the package using following lines install using the following lines:

.. code :: bash

    export CC=mpicc
    python setup.py configure --mpi [--hdf5=/path/to/parallel/hdf5]
    python setup.py build

where the :bash:`/path/to/parallel/hdf5` is

.. code:: bash

    /usr/local/share/Modules/modulefiles/hdf5/intel-17.0/openmpi-1.10.2/1.10.0

which is the path we found in the above section.


