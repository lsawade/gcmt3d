#!/bin/bash

# Load necessary modules:
module load intel
module load openmpi
module load cudatoolkit

# Execute script
srun ./bin/xmeshfem3D
