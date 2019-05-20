#!/bin/bash

# Get command line arguments
NODES=$1
TASKS=$2
NPAR=0-$(($3-1))
RUNDIR=$4
SCRIPT=$5

# Set max runtime for job
TIME=00:00:20

echo "SUBMITTING  ..."

# Run batch script
sbatch -N $NODES -n $TASKS --array=$NPAR -t $TIME --export=RUNDIR=$RUNDIR,
NODES=$NODES,TASKS=$TASKS,NPAR=$NPAR $SCRIPT

echo "SUBMITTED."
