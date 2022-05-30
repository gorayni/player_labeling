#!/bin/bash

all_frames=""
scale="1.0"
iext="jpg"
while getopts :ae:s: opt; do
  case "${opt}" in 
    a)
      all_frames="True"
      ;;
    e)
      iext=${OPTARG}
      ;;
    s)
      scale=${OPTARG}
      ;;
    :) echo "Missing argument for option -$OPTARG"; exit 1;;
    \?) echo "Unknown option -$OPTARG"; exit 1;;
  esac
done

shift $((OPTIND-1))

fname=$(basename "$1");
dname=$(dirname "$1");
frames_dir="$2"

video_ini="$dname"/video.ini
if [ -f "$video_ini" ]; then
  start_sec=`grep -A2 '\['"$fname"'\]' "$video_ini" | tail -2 | head -1 | sed 's/.*= *//g'`
  start_time=`python3 -c 'import datetime; print(str(datetime.timedelta(seconds = '"$start_sec"')))'`

  duration_sec=`grep -A2 '\['"$fname"'\]' "$video_ini" | tail -1 | sed 's/.*= *//g'`  
else
  start_time="00:00:00.0"

  duration_sec=`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$1"`
fi
duration_time=`python3 -c 'import datetime; print(str(datetime.timedelta(seconds = '"$duration_sec"')))'`

scale_h="$scale"
sar=`ffprobe -v error -select_streams v:0 -show_entries stream=sample_aspect_ratio -of default=noprint_wrappers=1:nokey=1 "$1"`
if [ $sar != "N/A" ]; then
  sar=`echo "$sar" | tr ':' '/'`  
  scale_w=`python3 -c 'sar='"$sar"'; print(1 if sar <= 1 else '"$scale"'*sar)'`
else
  scale_w="$scale"
fi

filtergraph=scale=iw*"$scale_w":ih*"$scale_h"
if test -z "$all_frames"; then
  filtergraph="$filtergraph",fps=fps=2.:round=down
fi

if [ $iext == "png" ]; then
  ffmpeg -hide_banner \
         -loglevel error \
         -ss "$start_time" \
         -t "$duration_time" \
         -i "$1" \
         -vf "$filtergraph" \
         -vsync 1 \
         "$frames_dir"/%05d."$iext"
else
  ffmpeg -hide_banner \
         -loglevel error \
         -ss "$start_time" \
         -t "$duration_time" \
         -i "$1" \
         -vf "$filtergraph" \
         -vsync 1 \
         -q:v 1 \
         "$frames_dir"/%05d."$iext"
fi
