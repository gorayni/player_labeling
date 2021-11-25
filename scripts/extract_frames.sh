#!/bin/bash

fname=$(basename "$1");
dname=$(dirname "$1");
frames_dir="$2"

scale="1.0"
iext="jpg"

video_ini="$dname"/video.ini
if [ -f "$video_ini" ]; then    
  start_sec=`grep -A2 '\['"$fname"'\]' "$video_ini" | tail -2 | head -1 | sed 's/.*= *//g'`    
  start_time=`python3 -c 'import datetime; print(str(datetime.timedelta(seconds = '"$start_sec"')))'`

  duration_sec=`grep -A2 '\['"$fname"'\]' "$video_ini" | tail -1 | sed 's/.*= *//g'`
  duration_time=`python3 -c 'import datetime; print(str(datetime.timedelta(seconds = '"$duration_sec"')))'` 
else
  start_time="00:00:00.0"

  duration=`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$1"`
  duration_time=`python3 -c 'import datetime; print(str(datetime.timedelta(seconds = '"$duration"')))'`    
fi

if [ $iext == "png" ]; then
  ffmpeg -hide_banner -loglevel error -ss "$start_time" -t "$duration_time" -i "$1" -vf scale=iw*"$scale":ih*"$scale",fps=fps=2.:round=down -vsync 1 "$frames_dir"/%05d."$iext"
else
  ffmpeg -hide_banner -loglevel error -ss "$start_time" -t "$duration_time" -i "$1" -vf scale=iw*"$scale":ih*"$scale",fps=fps=2.:round=down -vsync 1 -q:v 1 "$frames_dir"/%05d."$iext"
fi
