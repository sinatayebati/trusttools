#!/bin/bash

# Help function
function show_help {
    echo "Usage: $0 [options] file1.json file2.json ..."
    echo "       $0 [options] --dir directory_with_jsons --pattern 'output_*.json'"
    echo
    echo "Options:"
    echo "  -h, --help             Show this help message"
    echo "  -m, --model MODEL      Specify the LLM model (default: gpt-4.1-mini)"
    echo "  -d, --domain DOMAIN    Specify the domain (medical, math, etc., default: general)"
    echo "  -p, --pattern PATTERN  Process files matching pattern (e.g., 'output_*.json')"
    echo "  -l, --limit N          Limit processing to N files (default: process all files)"
    echo "  --dir DIRECTORY        Specify a directory containing the JSON files"
    echo
    echo "Examples:"
    echo "  $0 --domain medical file1.json file2.json"
    echo "  $0 --model gpt-4o --domain medical --dir /path/to/jsons --pattern 'output_*.json'"
    echo "  $0 --domain medical --dir /path/to/jsons --pattern 'output_*.json' --limit 5"
}

# Default values
MODEL="gpt-4.1-mini"
DOMAIN="general"
PATTERN=""
LIMIT=0
DIRECTORY=""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command line arguments
FILES=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -p|--pattern)
            PATTERN="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        --dir)
            DIRECTORY="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

# Process files based on directory and pattern
FULL_PATHS=()
if [ -n "$DIRECTORY" ]; then
    if [ ! -d "$DIRECTORY" ]; then
        echo "Error: Directory $DIRECTORY does not exist"
        exit 1
    fi
    
    # If pattern is provided, find matching files in the directory
    if [ -n "$PATTERN" ]; then
        # Use find instead of ls for better handling of filenames with spaces
        MATCHED_FILES=($(find "$DIRECTORY" -maxdepth 1 -name "$PATTERN" -type f 2>/dev/null))
        if [ ${#MATCHED_FILES[@]} -eq 0 ]; then
            echo "No files matched pattern: $PATTERN in directory: $DIRECTORY"
            exit 1
        fi
        
        # Apply limit if specified
        if [ $LIMIT -gt 0 ] && [ $LIMIT -lt ${#MATCHED_FILES[@]} ]; then
            MATCHED_FILES=("${MATCHED_FILES[@]:0:$LIMIT}")
        fi
        
        FULL_PATHS=("${MATCHED_FILES[@]}")
    elif [ ${#FILES[@]} -gt 0 ]; then
        # If individual files were specified, prepend the directory
        for file in "${FILES[@]}"; do
            FULL_PATHS+=("$DIRECTORY/$file")
        done
    else
        echo "Error: Either --pattern or explicit file names must be specified with --dir"
        show_help
        exit 1
    fi
else
    # No directory specified, use files as is
    if [ ${#FILES[@]} -eq 0 ]; then
        echo "Error: No files specified"
        show_help
        exit 1
    fi
    FULL_PATHS=("${FILES[@]}")
fi

# Ensure we have files to process
if [ ${#FULL_PATHS[@]} -eq 0 ]; then
    echo "No files to process"
    exit 1
fi

# Build the command with absolute paths
CMD="python3 $SCRIPT_DIR/llm_judge.py --model $MODEL --domain $DOMAIN"

# Add all files with their full paths
for file in "${FULL_PATHS[@]}"; do
    CMD="$CMD \"$file\""
done

echo "Running judge with model: $MODEL, domain: $DOMAIN"
echo "Processing ${#FULL_PATHS[@]} files"
echo "Command: $CMD"

# Execute the command
eval $CMD

echo "Completed processing ${#FULL_PATHS[@]} files" 