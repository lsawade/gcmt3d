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
by name. These are set in the ``params/SpecfemParams/CompilersAndModules.yml``


Setting up ``specfem``
++++++++++++++++++++++

There are three main steps in getting specfem to run.

1. Set Values in the ``Par_file`` Configure the compilation setup.

2. Running ``specfem3d_globe``


1. The ``Par_file`` and Configuration
-------------------------------------

The ``Par_file`` can be set using the parameter file
``params/SpecfemParams/SpecfemParams.yml``. Here all parameters, such as the
length of the recording, the resolution, GPU or CPU runs etc. can be set.
Then, running the specfem can be configured and recompiled using the
``DATAFixer`` class or the script ``00_Fix_Specfem_And_Recompile.py`` which
directly takes in the ``YAML`` parameter file you have set, fixes the specfem
``Par_file``, configures, recompiles and creates the mesh.

2. Running ``specfem``
----------------------

Already now, specfem could be run using the ``run_solver()`` method of the
``DATAFixer`` class, but where is the fun in that! The next step is to create
 an earthquake database or an entry into an existing database so that
meaningful simulations can be run of which the simulated data can be inverted
 for new earthquake solutions.

