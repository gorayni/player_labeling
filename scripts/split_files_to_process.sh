#!/bin/bash

export SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source "$SCRIPTS_DIR"/soccernet_conf.sh

model_name=""

case $1 in
  velocity)
    num_process=$NUM_VELOCITY_PROCESSES
    ;;
  preliminary_player_labels)
    num_process=$NUM_LABELING_PROCESSES
    ;;
  train)
    num_process=$NUM_TRAINING_PROCESSES
    model_name=$(jq -r .model.name "$CONFIG_DIR"/training.json)
    ;;
  segmentation)
    num_process=$NUM_SEGMENTATION_PROCESSES
    ;;
  team_classification)
    num_process=$NUM_TEAM_CLASSIFICATION_PROCESSES
    ;;
esac

tmp_file_prefix=tmp_"$1"_file_

# Remove temporal files
rm -f "$tmp_file_prefix"*

files_to_process=`mktemp -t files_to_processed_XXXXXXXXXX.txt`
python "$SCRIPTS_DIR"/split_files_to_process.py -d "$SOCCERNET_PATH" -m "$model_name" -f "$1" > "$files_to_process"

find . -maxdepth 1 -type f -name "$tmp_file_prefix"'*' | xargs -I{} rm "{}"
split -n l/"$num_process" -e "$files_to_process" "$tmp_file_prefix"

rm "$files_to_process"
