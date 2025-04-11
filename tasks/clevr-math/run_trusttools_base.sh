#!/bin/bash

############ [1] Batching Run ############
PROJECT_DIR="./"

############
# Updated label to reflect validation
LABEL="trusttools_validated"

THREADS=2
TASK="clevr-math"
DATA_FILE="$TASK/data/data.json"
LOG_DIR="$TASK/logs/$LABEL"
OUT_DIR="$TASK/results/$LABEL"
CACHE_DIR="$TASK/cache"

LLM="gpt-4o-mini"

# Keep tools the same, unless validation requires specific tools (unlikely)
ENABLED_TOOLS="Generalist_Solution_Generator_Tool"

# Set the desired alpha for validation
VALIDATION_ALPHA=0.1
############

cd $PROJECT_DIR
mkdir -p $LOG_DIR
mkdir -p $OUT_DIR # Ensure output directory is also created

# Define the array of specific indices
indices=($(seq 106 106))

# Skip indices if the output file already exists
new_indices=()
for i in "${indices[@]}"; do
    if [ ! -f "$OUT_DIR/output_$i.json" ]; then
        new_indices+=($i)
    else
        echo "Output file already exists: $OUT_DIR/output_$i.json"
    fi
done
indices=("${new_indices[@]}")
echo "Final indices: ${indices[@]}"

# Check if indices array is empty
if [ ${#indices[@]} -eq 0 ]; then
    echo "All tasks completed."
else
    # Function to run the task for a single index
    run_task() {
        local i=$1
        local alpha=$2 # Pass alpha as an argument
        echo "Running task for index $i with validation alpha $alpha"
        python solve.py \
        --index $i \
        --task $TASK \
        --data_file $DATA_FILE \
        --llm_engine_name $LLM \
        --root_cache_dir $CACHE_DIR \
        --output_json_dir $OUT_DIR \
        --output_types "direct,validated_direct" \
        --validation_alpha $alpha \
        --enabled_tools "$ENABLED_TOOLS" \
        --max_time 300 \
        --verbose False \
        2>&1 | tee $LOG_DIR/$i.log
        echo "Completed task for index $i"
        echo "------------------------"
    }

    # Export the function and variables so they can be used by parallel
    export -f run_task
    export TASK DATA_FILE LOG_DIR OUT_DIR CACHE_DIR LLM ENABLED_TOOLS VALIDATION_ALPHA

    # Run the tasks in parallel using GNU Parallel
    echo "Starting parallel execution..."
    # Pass VALIDATION_ALPHA to each task
    parallel -j $THREADS run_task {1} $VALIDATION_ALPHA ::: "${indices[@]}"
    echo "All tasks completed."
fi

############ [2] Calculate Scores ############
cd $PROJECT_DIR

# IMPORTANT: Decide which output to score.
# If you want to score the *original* direct output:
# RESPONSE_TYPE="direct"
# If you want to score the *validated* direct output:
RESPONSE_TYPE="validated_direct" # Change this based on what you want to evaluate

# The scoring script needs access to `outputs.validated_direct` in the JSON.
# Ensure calculate_score.py is updated accordingly if needed.
echo "Calculating score for response type: $RESPONSE_TYPE"

python $TASK/calculate_score.py \
--data_file $DATA_FILE \
--result_dir $OUT_DIR \
--response_type $RESPONSE_TYPE \
--output_file "$OUT_DIR/final_results_$RESPONSE_TYPE.json" \
| tee "$OUT_DIR/final_score_$RESPONSE_TYPE.log"

echo "Scoring complete. Results saved in $OUT_DIR"

