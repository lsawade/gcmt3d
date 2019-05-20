#!/bin/bash

#SBATCH --job-name=job_array_%a
#SBATCH --output=output/job_%a.out
#SBATCH --error=output/job_%a.err

echo "JOB: ${STRINGS[${SLURM_ARRAY_TASK_ID}]}"
echo "RUNDIR: $RUNDIR"
echo "PARAMETERS: $NPAR"
echo "NODES: $NODES"
echo "TASKS: $TASKS"

# Check locaysh
echo "PWD: $PWD"
echo "FILE: input/param_${STRINGS[${SLURM_ARRAY_TASK_ID}]}.txt"

# Create String array as a function of wanted parameters
#STRINGS= (CMT_rr CMT_tt CMT_pp CMT_rt CMT_rp CMT_tp CMT_depth CMT_lat CMT_lon)
STRINGS=(0 1)

# Change to earthquake simulation directory
cd "$RUNDIR/run_${STRINGS[${SLURM_ARRAY_TASK_ID}]}"

# Execute script
srun /bin/python demo.py input/param_${STRINGS[${SLURM_ARRAY_TASK_ID}]}.txt
