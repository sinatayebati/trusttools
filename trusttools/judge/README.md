# LLM Judge

This module provides tools to use LLMs as judges to evaluate the validity of text segments in response to queries.

## Overview

The LLM Judge evaluates text segments from JSON files, determining if each segment is valid based on:
1. Factual accuracy
2. Relevance to the query
3. Usefulness in addressing the query

## Files

- `llm_judge.py`: The main Python script that uses OpenAI models to judge text segments
- `run_judge.sh`: A shell script to run the judge on multiple JSON files with various options

## Requirements

- Python 3.6+
- OpenAI API key set in your environment variables
- Required Python packages (trusttools and its dependencies)

## Usage

### Running with the Shell Script

```bash
# Basic usage with specific files
./run_judge.sh file1.json file2.json

# Specify domain and model
./run_judge.sh --domain medical --model gpt-4.1-mini file1.json file2.json

# Process files matching a pattern
./run_judge.sh --domain medical --pattern "output_*.json"

# Limit to processing only 5 files
./run_judge.sh --domain medical --pattern "output_*.json" --limit 5

# Get help
./run_judge.sh --help
```

### Options

- `-h, --help`: Show help message
- `-m, --model MODEL`: Specify the LLM model (default: gpt-4.1-mini)
- `-d, --domain DOMAIN`: Specify the domain (medical, math, etc., default: general)
- `-p, --pattern PATTERN`: Process files matching pattern (e.g., 'output_*.json')
- `-l, --limit N`: Limit processing to N files
- `--dir DIRECTORY`: Specify a directory containing the JSON files

### Running the Python Script Directly

```bash
python3 llm_judge.py --model gpt-4.1-mini --domain medical file1.json file2.json
```

## JSON File Format

The script expects JSON files with the following structure:

```json
{
    "query": "Can acanthosis nigricans go away?",
    "validation_results": {
        "direct_output_atoms": [
            {
                "text": "Text segment 1",
                "score": -0.089,
                "valid": true
            },
            {
                "text": "Text segment 2",
                "score": -0.161,
                "valid": true
            },
            // More segments...
        ]
    }
}
```

After processing, the `valid` field in each segment will be updated based on the LLM's judgment, and a new field `judge_reason` will be added with a brief explanation.

## Example

```bash
# Process all medical-related JSON files in a specific directory
./run_judge.sh --model gpt-4o-mini --domain medical --dir /home/sina/projects/octotools/tasks/medlfqa/results/trusttools_validated --pattern "output_*.json"
``` 