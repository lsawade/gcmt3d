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

for mod in $7
do
    module load $mod
done

./configure MPIFC=$MPI90 FC=$F90 CC=$CC CXX=$CXX

make -j default

cd -