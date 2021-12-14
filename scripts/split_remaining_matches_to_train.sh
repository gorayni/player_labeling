#!/bin/bash

export SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source "$SCRIPTS_DIR"/soccernet_conf.sh

export MODEL_NAME=$(jq -r .model.name "$CONFIG_DIR"/training.json)

function not_trained_yet {
   while read match_path ; do
     weights_fpath="$match_path"/player_labeling/weights/"$MODEL_NAME"/initial_model.pth.tar
     [ ! -f "$weights_fpath" ] && echo "$match_path"
   done
}
export -f not_trained_yet


# Remove temporal files
rm -f tmp_matches_to_train_file_*

matches_file=`mktemp -t matches_XXXXXXXXXX.txt`

find "$SOCCERNET_PATH" \
    -mindepth 3 \
    -maxdepth 3 \
    -type d | not_trained_yet | sort > "$matches_file"

find . -maxdepth 1 -type f -name 'tmp_matches_to_train_file_*' | xargs -I{} rm "{}"
split -n l/"$NUM_TRAINING_PROCESSES" -e "$matches_file" tmp_matches_to_train_file_
rm "$matches_file"
