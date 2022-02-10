#!/bin/bash

export SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source "$SCRIPTS_DIR"/soccernet_conf.sh

function sampling_aspect_ratio {
   while read match_path ; do
    sar=`ffprobe -v error -select_streams v:0 -show_entries stream=sample_aspect_ratio -of default=noprint_wrappers=1:nokey=1 "$SOCCERNET_PATH"/"$match_path"/1_HQ.mkv | head -1`
    if [ $sar == "N/A" ]; then
      sar="1:1"
    fi
    echo "$match_path","$sar"
   done
}
export -f sampling_aspect_ratio

cd  "$SOCCERNET_PATH"
printf "match_path,SAR\n" > sampling_aspect_ratio.csv

num_matches=`find . -mindepth 3 -maxdepth 3 -type d | wc -l`

echo "$num_matches"

find . -mindepth 3 \
       -maxdepth 3 \
       -type d | sed 's/^..//g' | sort | sampling_aspect_ratio | tqdm --total $num_matches >> sampling_aspect_ratio.csv
