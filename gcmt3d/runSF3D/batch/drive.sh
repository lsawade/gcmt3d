#!/bin/bash

# Get command line arguments
export NODES=$1
export TASKS=$2
export NPAR=0-$3
export RUNDIR=$4
export TIME=$5
export VERBOSE=$6
export SCRIPT=$7


# Control output
if [ "$VERBOSE" -eq "1" ]
    then
      echo $NODES
      echo $TASKS
      echo $NPAR
      echo $RUNDIR
      echo $TIME
      echo $SCRIPT
fi

# Control output
if [ "$VERBOSE" -eq "1" ] 
    then
      echo "SUBMITTING ..."
fi

# Run batch script
sbatch -N $NODES -n $TASKS --array=$NPAR -t $TIME --export=ALL --output=$RUNDIR/job_%a.out --error=$RUNDIR/job_%a.err $SCRIPT


# Control output
if [ "$VERBOSE" -eq "1" ]
    then
      echo "SUBMITTED."

fi
