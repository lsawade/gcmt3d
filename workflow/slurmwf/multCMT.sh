#!/bin/bash

FILEDIR=$1

for file in $(ls $FILEDIR); 
    do ./invert_earthquake.start $FILEDIR/$file; 
done;
