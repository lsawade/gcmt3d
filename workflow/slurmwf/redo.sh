#!/bin/bash

# Get redo file from command line
redo_file=$1

if [ -z "$redo_file" ]
then
    echo "\$redo_file is empty. Quit"
    exit 0
fi

databasedir=/scratch/gpfs/database

for file in $(cat $redo_file)
do
    echo $databasedir/$file
done

