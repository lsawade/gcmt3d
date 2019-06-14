Specfem Setup and compilation
=============================

The package also includes a specfem configuration and parameter setup class.
It is not necessary to set it up, compile etc. if you already have setup
specfem, but it is necessary to set the correct flags in the SpecfemParameter
file.


Setting up the compilers
++++++++++++++++++++++++

It is important that specfem is both built and run with the same set of
compilers. The best thing is to ask your Sysadmin if you are unsure about the
available ones. For the Tiger Cluster at Princeton, the following setup of
compilers is used

.. code-block:: yaml

    # Modules
    modulelist: [intel, openmpi]
    gpu_module: cudatoolkit

    # C compiler
    cc: icc

    # C++ Comiler
    cpp: icpc

    # Fortran Compiler
    f90: ifort

    # MPI C Compiler
    mpicc: mpicc

    # MPI Fortran Compiler
    mpif90: mpifort

The modules are used to load the specific compilers which are then specified
by name.


Setting up ``specfem``
++++++++++++++++++++++

There are three main steps in getting specfem to run.

1. Set Values in the ``Par_file``

2. Configure the compilation setup.