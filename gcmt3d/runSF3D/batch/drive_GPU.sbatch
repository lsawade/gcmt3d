#!/bin/bash

#SBATCH --job-name=CMT_array_%a

# Load necessary modules:
for mod in $MODULES
do
    module load $mod
done

# Load GPU module
module load $GPU_MODULE

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
      echo "TASKS/NODE: $TASKS_PER_NODE"
      echo "MEMORY: $MEMORY"
      echo "PARAMETETERS: $NPAR"
      echo "DIR: $RUNDIR"
      echo "MODULES: $MODULES"
      echo "GPU MODULE: $GPU_MODULE"
      echo "TIME: $TIME"
      echo "SCRIPT: $SCRIPT"
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
