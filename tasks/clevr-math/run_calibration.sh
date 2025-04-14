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
CALIBRATION_JSON="./clevr-math/data_calibration/data_calibration.json"
ALPHA=0.1
OUTPUT_THRESHOLD="conformal_threshold.npz"
MIN_INVERTED_SCORE=0.0

# Function to print usage
function print_usage {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --calibration_json PATH   Path to calibration JSON dataset (required)"
    echo "  --alpha VALUE             Significance level (default: 0.1)"
    echo "  --output_threshold PATH   Output file name (default: conformal_threshold.npz)"
    echo "  --min_inverted_score NUM  Default badness if no invalid chunk (default: 0.0)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --calibration_json path/to/calibration_data.json --alpha 0.05"
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
        --output_threshold)
            OUTPUT_THRESHOLD="$2"
            shift 2
            ;;
        --min_inverted_score)
            MIN_INVERTED_SCORE="$2"
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

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Run the Python script
echo "Running calibration with:"
echo "  Calibration JSON: $CALIBRATION_JSON"
echo "  Alpha: $ALPHA"
echo "  Output threshold file: $OUTPUT_THRESHOLD"
echo "  Min inverted score: $MIN_INVERTED_SCORE"
echo ""

python "$PYTHON_SCRIPT" \
    --calibration_json "$CALIBRATION_JSON" \
    --alpha "$ALPHA" \
    --output_threshold "$OUTPUT_THRESHOLD" \
    --min_inverted_score "$MIN_INVERTED_SCORE"

echo ""
echo "Calibration completed successfully!" 