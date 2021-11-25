#!/bin/bash

export SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source "$SCRIPTS_DIR"/soccernet_conf.sh

function num_optical_flow_frames {
   while read match_path ; do
    first_half="$(find "$SOCCERNET_PATH"/"$match_path"/1_HQ/u -mindepth 1 -maxdepth 1 -type f -name '*.jpg' | wc -l )"
    second_half="$(find "$SOCCERNET_PATH"/"$match_path"/2_HQ/u -mindepth 1 -maxdepth 1 -type f -name '*.jpg' | wc -l )"
    echo "$match_path","$first_half","$second_half"
   done
}
export -f num_optical_flow_frames

cd  "$SOCCERNET_PATH"
printf "match_path,num_frames_first_half,num_frames_second_half\n" > num_optical_flow_frames.csv

num_matches=`find . -mindepth 3 -maxdepth 3 -type d | wc -l`

find . -mindepth 3 \
       -maxdepth 3 \
       -type d | sed 's/^..//g' | sort | num_optical_flow_frames | tqdm --total $num_matches >> num_optical_flow_frames.csv
