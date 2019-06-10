#!/bin/bash

#SBATCH --job-name=CMT_array_%a

# Load necessary modules:
module load intel/18.0
module load openmpi/intel-18.0

# Create String array as a function of wanted parameters
STRINGS=(CMT CMT_rr CMT_tt CMT_pp CMT_rt CMT_rp CMT_tp CMT_depth CMT_lat
CMT_lon)

# Check locaysh
if [ "$VERBOSE" -eq "1" ]
    then
      echo "JOB: $SLURM_ARRAY_TASK_ID"
      echo "RUNDIR: $RUNDIR"
      echo "PARAMETERS: $NPAR"
      echo "NODES: $NODES"
      echo "TASKS: $TASKS"
      echo "PWD: $PWD"
      echo "DIR: ${STRINGS[${SLURM_ARRAY_TASK_ID}]}"
fi

# Change to earthquake simulation directory
echo ${STRINGS[$SLURM_ARRAY_TASK_ID]}

cd $RUNDIR/${STRINGS[$SLURM_ARRAY_TASK_ID]}

echo $PWD

if [ "$VERBOSE" -eq "1" ]
    then
      echo "PWD: $PWD"
fi

# Execute script
srun ./bin/xspecfem3D
