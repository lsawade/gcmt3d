#!/bin/bash

FILEDIR=$1
DONEDIR=$2
for file in $(ls $FILEDIR); 
    do ./invert_earthquake.start $FILEDIR/$file; 
    mv $file $DONEDIR/$file;
done;
