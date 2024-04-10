#!/bin/bash

dir_in=$1
dir_out=$2

mkdir -p $dir_out

# Audio Processing
sample_rate=24000
dir_in_wav=${dir_in}/Wave
dir_wav=${dir_out}/wav_${sample_rate}
dir_mel=${dir_out}/mel

mkdir -p $dir_wav $dir_mel
bash audio/audio_normalization.sh $dir_in_wav $dir_wav $sample_rate
python audio/melspectrogram.py $dir_in $dir_mel

# Text Processing
dir_in_text=${dir_in}/PhoneLabeling
phone_file=${dir_out}/phone.txt
dur_file=${dir_out}/dur.txt
python text/parse_textgrid.py $dir_in_text $dir_wav ${phone_file}, ${dur_file}