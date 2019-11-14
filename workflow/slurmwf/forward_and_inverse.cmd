#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=50
#SBATCH --time 00:01:00

# GCMT3D Directory
GCMT3D=/home/lsawade/GCMT3D

# Changedirectory to slurmdir
cd $GCMT3D/workflow/slurm

# Multiconfiguration file
MULT=`realpath multi.conf`

# Change directory to testdir
cd /home/lsawade/GCMT3D/workflow/slurm/test_dir
TEST_OUT=`realpath test_output`

echo $EQ_IN_DB >> $TEST_OUT

CMTLIST=(CMT CMT_rr CMT_pp CMT_tt)

for CMT in "${CMTLIST[@]}"; do
    srun -n1 cat ./$CMT/CMT* >> $TEST_OUT &
done
wait
