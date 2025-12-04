#!/bin/bash

LOG_DIR="logs"
CONFIG_DIR="configs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

CONFIG_LIST=""

if [ "$#" -gt 0 ]; then
  for item in "$@"; do
    if [ -d "$item" ]; then
      for cfg in "$item"/*.json; do
        [ -e "$cfg" ] && CONFIG_LIST="$CONFIG_LIST $cfg"
      done
    else
      CONFIG_LIST="$CONFIG_LIST $item"
    fi
  done
else
  for cfg in "$CONFIG_DIR"/*.json; do
    [ -e "$cfg" ] && CONFIG_LIST="$CONFIG_LIST $cfg"
  done
fi

if [ -z "$CONFIG_LIST" ]; then
  echo "No config files found."
  exit 1
fi

mkdir -p "$LOG_DIR"

for cfg in $CONFIG_LIST; do
  name=$(basename "$cfg" .json)
  log_file="$LOG_DIR/${name}_${TIMESTAMP}.log"
  echo "Running $cfg ..."
  python -m surveillance --config "$cfg" | tee "$log_file"
done

#  ./run_exp.sh gpt-configs/home_gpt_fi30_native.json
#  ./run_exp.sh gemini-configs 
