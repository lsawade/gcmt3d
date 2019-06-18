#!/bin/bash

# Get command line arguments
export NODES=$1
export TASKS=$2
export TASKS_PER_NODE=$3
export MEMORY=$4
export NPAR=0-$5
export RUNDIR=$6
export TIME=$7
export MODULES=$8
export GPU_MODULE=$9
export VERBOSE="${10}"
export SCRIPT="${11}"


# Control output
if [ "$VERBOSE" -eq "1" ]
    then
      echo "Nodes: $NODES"
      echo "Tasks: $TASKS"
      echo "Tasks per Node: $TASKS_PER_NODE"
      echo "Memory: $MEMORY"
      echo "Parameters: $NPAR"
      echo "DIR: $RUNDIR"
      echo "Modules: $MODULES"
      echo "GPU module: $GPU_MODULE"
      echo "Walltime: $TIME"
      echo "Script: $SCRIPT"
fi

# Control output
if [ "$VERBOSE" -eq "1" ] 
    then
      echo "SUBMITTING ..."
fi

# Run batch script
sbatch -N $NODES -n $TASKS --ntasks-per-node $TASKS_PER_NODE \
--gres=gpu:$TASKS_PER_NODE -p pReserved --mem $MEMORY --array=$NPAR \
-t $TIME --export=ALL --output=$RUNDIR/job_%a.out --error=$RUNDIR/job_%a.err \
$SCRIPT


# Control output
if [ "$VERBOSE" -eq "1" ]
    then
      echo "SUBMITTED."

fi

