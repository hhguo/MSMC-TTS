#!/bin/bash

dir_in=$1
dir_out=$2
sample_rate=$3

mkdir -p $dir_out

for filename in `ls $dir_in`; do
    if [ ${filename##*.} != 'wav' ]; then
        continue
    fi
    echo $filename
    sox $dir_in/$filename -c 1 -r $sample_rate --norm=-7 $dir_out/$filename
done
