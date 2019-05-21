#!/bin/bash

#SBATCH -t 00:00:20
#SBATCH --job-name=CMT_array_%a
# blub SBATCH --output=job_%a.out
# blub SBATCH --error=job_%a.err


# Create String array as a function of wanted parameters
STRINGS= (CMT_rr CMT_tt CMT_pp CMT_rt CMT_rp CMT_tp CMT_depth CMT_lat CMT_lon)

# Check locaysh
if [ "$VERBOSE" -eq "1" ]
    then
      echo "JOB: ${STRINGS[${SLURM_ARRAY_TASK_ID}]}"
      echo "RUNDIR: $RUNDIR"
      echo "PARAMETERS: $NPAR"
      echo "NODES: $NODES"
      echo "TASKS: $TASKS"
      echo "PWD: $PWD"
      echo "FILE: input/param_${STRINGS[${SLURM_ARRAY_TASK_ID}]}.txt"
fi

# Change to earthquake simulation directory
cd "$RUNDIR/${STRINGS[${SLURM_ARRAY_TASK_ID}]}"

# Execute script
srun ./bin/xspecfem3D
