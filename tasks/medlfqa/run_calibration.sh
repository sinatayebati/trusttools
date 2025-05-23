#!/bin/bash

# Script to run the calibration process for the conformal factual method
# Author: User
# Date: $(date +%Y-%m-%d)

# Exit on error
set -e

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/calibrate_factual_threshold.py"

# Default values
CALIBRATION_JSON="./medlfqa/data_calibration/data_calibration.json"
ALPHA=0.2
REQUIRED_ACCURACY=0.80
OUTPUT_THRESHOLD="conformal_threshold.npz"
SCORE_FIELD="score"

# Function to print usage
function print_usage {
    echo "Usage: $0 [options]"
    echo ""
    echo "Conformal Prediction Calibration Script"
    echo "Computes threshold using method from 'Language Models with Conformal Factuality Guarantees'"
    echo ""
    echo "Options:"
    echo "  --calibration_json PATH       Path to calibration JSON dataset (required)"
    echo "  --alpha VALUE                 Significance level, e.g. 0.1 for 90% coverage (default: 0.1)"
    echo "  --required_accuracy VALUE     Required fraction correct (parameter 'a'), e.g. 0.9 for 90% accuracy (default: 0.9)"
    echo "  --output_threshold PATH       Output file name (default: conformal_threshold.npz)"
    echo "  --score_field FIELD          Field name for confidence scores in atomic_facts (default: score)"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --calibration_json path/to/calibration_data.json"
    echo "  $0 --calibration_json data.json --alpha 0.05 --required_accuracy 0.95"
    echo ""
    echo "Conformal Guarantee:"
    echo "  With probability >= $(echo "scale=1; (1-$ALPHA)*100" | bc)%, at least $(echo "scale=1; $REQUIRED_ACCURACY*100" | bc)% of accepted subclaims will be correct"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --calibration_json)
            CALIBRATION_JSON="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --required_accuracy)
            REQUIRED_ACCURACY="$2"
            shift 2
            ;;
        --output_threshold)
            OUTPUT_THRESHOLD="$2"
            shift 2
            ;;
        --score_field)
            SCORE_FIELD="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if calibration_json is provided
if [ -z "$CALIBRATION_JSON" ]; then
    echo "Error: --calibration_json is required"
    print_usage
    exit 1
fi

# Validate parameter ranges
if (( $(echo "$ALPHA <= 0" | bc -l) )) || (( $(echo "$ALPHA >= 1" | bc -l) )); then
    echo "Error: alpha must be in (0, 1), got $ALPHA"
    exit 1
fi

if (( $(echo "$REQUIRED_ACCURACY <= 0" | bc -l) )) || (( $(echo "$REQUIRED_ACCURACY > 1" | bc -l) )); then
    echo "Error: required_accuracy must be in (0, 1], got $REQUIRED_ACCURACY"
    exit 1
fi

# Check if calibration JSON file exists
if [ ! -f "$CALIBRATION_JSON" ]; then
    echo "Error: Calibration JSON file not found: $CALIBRATION_JSON"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Print configuration
echo "=== Conformal Prediction Calibration ==="
echo "Configuration:"
echo "  Calibration JSON: $CALIBRATION_JSON"
echo "  Alpha (significance level): $ALPHA"
echo "  Required accuracy (parameter 'a'): $REQUIRED_ACCURACY"
echo "  Output threshold file: $OUTPUT_THRESHOLD"
echo "  Score field: $SCORE_FIELD"
echo ""
echo "Conformal Guarantee:"
echo "  With probability >= $(echo "scale=1; (1-$ALPHA)*100" | bc)%, at least $(echo "scale=1; $REQUIRED_ACCURACY*100" | bc)% of accepted subclaims will be correct"
echo ""

# Run the Python script
echo "Running calibration..."
python "$PYTHON_SCRIPT" \
    --calibration_json "$CALIBRATION_JSON" \
    --alpha "$ALPHA" \
    --required_accuracy "$REQUIRED_ACCURACY" \
    --output_threshold "$OUTPUT_THRESHOLD" \
    --score_field "$SCORE_FIELD"

echo ""
echo "=== Calibration Completed Successfully! ==="
echo "Threshold saved to: $OUTPUT_THRESHOLD"
echo ""
echo "Next steps:"
echo "1. Use this threshold file in your ConformalValidator"
echo "2. The validator will apply: keep subclaims where score >= threshold"
echo "3. This provides the conformal guarantee shown above" 