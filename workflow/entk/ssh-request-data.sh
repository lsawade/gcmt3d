#!/bin/bash
module load $4
conda activate $3
request-data -f $1 -p $2
