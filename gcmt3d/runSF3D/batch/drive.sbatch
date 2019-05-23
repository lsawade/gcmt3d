#!/bin/bash

#SBATCH --job-name=CMT_array_%a
# blub SBATCH --output=job_%a.out
# blub SBATCH --error=job_%a.err

# Load necessary modules:
module load intel/18.0
module load openmpi/intel-18.0
module load openmpi/intel-18.0

# Set FORTRAN COMPILER
export FORTRAN_COMPILER=INTEL

# Create String array as a function of wanted parameters
STRINGS=(CMT_rr CMT_tt CMT_pp CMT_rt CMT_rp CMT_tp CMT_depth CMT_lat CMT_lon)

# Check locaysh
if [ "$VERBOSE" -eq "1" ]
    then
      echo "JOB: ${STRINGS[${SLURM_ARRAY_TASK_ID}]}"
      echo "RUNDIR: $RUNDIR"
      echo "PARAMETERS: $NPAR"
      echo "NODES: $NODES"
      echo "TASKS: $TASKS"
      echo "PWD: $PWD"
      echo "DIR: ${STRINGS[${SLURM_ARRAY_TASK_ID}]}"
fi

# Change to earthquake simulation directory
cd "$RUNDIR/${STRINGS[${SLURM_ARRAY_TASK_ID}]}"

# Execute script
srun ./bin/xspecfem3D
