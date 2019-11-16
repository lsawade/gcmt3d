#!/bin/bash

# Change directory to the Specfem directory
RUNDIR=$1
cd $RUNDIR

# Define compilers
MPIF90=$2
F90=$3
CC=$4
CXX=$5
MPICC=$6

# Load necessary modules
for mod in $7
do
    echo "Module Loaded: $mod"
    module load $mod
done

# Load GPU toolkit
module load $8

echo "./configure MPIFC=$MPI90 FC=$F90 CC=$CC CXX=$CXX --with-cuda=$9"

./configure MPIFC=$MPI90 FC=$F90 CC=$CC CXX=$CXX --with-cuda=$9

make -j default

cd -
