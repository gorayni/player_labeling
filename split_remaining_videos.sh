#!/bin/bash

source soccernet_conf.sh

function was_not_segmented {
   while read line ; do
    fname=$(basename "$line");
    dname=$(dirname "$line");
    fbname=${fname%.*};

    segmentation_results_fpath="$dname"/segmentation_results_"$fbname".npy
    [ ! -f "$segmentation_results_fpath" ] && echo "$line"
   done
}
export -f was_not_segmented

# Remove temporal files
rm -f tmp_video_file_*

videos_file=`mktemp -t videos_XXXXXXXXXX.txt`
find "$SOCCERNET_PATH" \
     -mindepth 4 \
     -maxdepth 4 \
     -type l \
     -name '*_HQ.mkv' | was_not_segmented | sort | head -100 > "$videos_file"

find . -maxdepth 1 -type f -name 'tmp_video_file_*' | xargs -I{} rm "{}"
split -n l/"$NUM_PROCESSES" -e "$videos_file" tmp_video_file_
rm "$videos_file"

chmod a+rw tmp_video_file_*
