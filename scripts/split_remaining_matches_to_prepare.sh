#!/bin/bash

export SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source "$SCRIPTS_DIR"/soccernet_conf.sh

function was_not_prepared {
   while read match_path ; do
    labels_fpath="$match_path"/"preliminary_player_labels.pkl"
    [ ! -f "$labels_fpath" ] && echo "$match_path"
   done
}
export -f was_not_prepared

# Remove temporal files
rm -f tmp_matches_file_*

matches_file=`mktemp -t matches_XXXXXXXXXX.txt`

find "$SOCCERNET_PATH" \
     -mindepth 3 \
     -maxdepth 3 \
     -type d | was_not_prepared | sort > "$matches_file"


find . -maxdepth 1 -type f -name 'tmp_matches_file_*' | xargs -I{} rm "{}"
split -n l/"$NUM_LABELING_PROCESSES" -e "$matches_file" tmp_matches_file_
rm "$matches_file"
