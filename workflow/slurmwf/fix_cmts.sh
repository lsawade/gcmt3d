#!/bin/bash

# Directory to fix
CMTDIR=$1


for file in $(ls $CMTDIR)
do
    echo "Fixing $file"
    # This line does magic. 
    # it replace the line that starts with event name with a new line 
    # that takes the value from the file and strips the C from the id
    sed -i 's/^event name: .*$/event name:    '$(cat $CMTDIR/$file | head -2 | tail -1 | cut -d: -f2 | tr -d '[:space:]' | tr -d "C")'/' $CMTDIR/$file

done

echo "Done."


    
