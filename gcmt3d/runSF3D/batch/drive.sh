#!/bin/bash

# Get command line arguments
NODES=$1
TASKS=$2
NPAR=0-$(($3-1))
RUNDIR=$4
SCRIPT=$5
VERBOSE=$6

if [ "$VERBOSE" -eq "1" ]
    then
      echo $NODES
      echo $TASKS
      echo $NPA        
      echo $RUNDIR
      echo $SCRIPT
fi
# Set max runtime for job
TIME=00:30:00

if [ "$VERBOSE" -eq "1" ] 
    then
      echo "SUBMITTING ..."
fi

# Run batch script
sbatch -N $NODES -n $TASKS --array=$NPAR -t $TIME --export=RUNDIR=$RUNDIR,NODES=$NODES,TASKS=$TASKS,NPAR=$NPAR,VERBOSE=$VERBOSE --output=$RUNDIR/job_%a.out --error=$RUNDIR/job_%a.err $SCRIPT

if [ "$VERBOSE" -eq "1" ]
    then
      echo "SUBMITTED."
fi
