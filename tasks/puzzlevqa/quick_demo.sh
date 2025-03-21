#!/bin/bash

############ [1] Single Run ############
PROJECT_DIR="./"

LABEL="quick_demo"

TASK="puzzlevqa"
DATA_FILE="$TASK/data/data.json"
LOG_DIR="$TASK/logs/$LABEL"
OUT_DIR="$TASK/results/$LABEL"
CACHE_DIR="$TASK/cache"

LLM="gpt-4o"

ENABLED_TOOLS="Text_Detector_Tool,Image_Captioner_Tool,Generalist_Solution_Generator_Tool,Advanced_Object_Detector_Tool"

INDEX=0

python solve.py \
--index $INDEX \
--task $TASK \
--data_file $DATA_FILE \
--llm_engine_name $LLM \
--output_json_dir $OUT_DIR \
--output_types direct \
--enabled_tools "$ENABLED_TOOLS" \
--max_time 300
