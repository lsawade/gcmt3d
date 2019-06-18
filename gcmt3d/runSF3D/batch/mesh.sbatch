#!/bin/bash

# Load necessary modules:
module load intel
module load openmpi

# Execute script
srun ./bin/xmeshfem3D
