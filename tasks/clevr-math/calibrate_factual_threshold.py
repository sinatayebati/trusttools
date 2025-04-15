"""
calibrate_factual_threshold.py

Usage (example):
  python calibrate_factual_threshold.py \
      --calibration_json path/to/calibration_data.json \
      --alpha 0.1 \
      --output_threshold path/to/conformal_threshold.npz

Description:
  Reads a calibration dataset in JSON format where each sample has:
   - "pid", "query", "image", "answer", "response"
   - "atomic_facts": a list of { "text", "score", "valid" }

  We assume "score" is the (mean) logprob of each chunk/atom,
  where more negative means "less confident" (worse).
  and "valid" is a manual annotation of correctness (True/False).

  We then compute a single "badness" value per sample:
    badness_i = max( -score_of_all_invalid_chunks )
    if at least one invalid chunk; otherwise use a default min_inverted_score.

  Because we invert the logprob scores for invalid chunks,
  a chunk with an extremely negative logprob ends up with
  a large positive "badness" (indicating it's highly suspect).

  Then we collect these sample-level badness values and compute
  the (1-alpha) quantile. This threshold is stored in a .npz file
  for later usage:
     'threshold' => <computed numeric threshold>
"""

import argparse
import json
import numpy as np
import os
import sys

def calibrate_factual_threshold(
    calibration_json_path: str,
    alpha: float,
    output_threshold_file: str,
    min_inverted_score: float = 0.0
):
    """
    Reads the calibration dataset and computes a single numeric threshold
    for 'factual conformal' trimming, inverting negative logprob scores.

    :param calibration_json_path: Path to the JSON file containing calibration samples
    :param alpha: Significance level (e.g. 0.1 means we keep ~90% coverage)
    :param output_threshold_file: Path to .npz file where threshold will be saved
    :param min_inverted_score: 'badness' score assigned if a sample has no invalid chunks
        (default 0.0 means 'perfectly safe' if no invalid chunk)
    :return: None (results are written to disk)
    """
    if not os.path.exists(calibration_json_path):
        raise FileNotFoundError(f"Calibration JSON file not found: {calibration_json_path}")

    # 1) Load the calibration data
    with open(calibration_json_path, "r") as f:
        calibration_data = json.load(f)

    # 2) Build array of "badness" for each sample
    #    Because your chunk "score" is presumably negative logprob,
    #    we invert it to ensure more negative => larger badness
    #    Then for an invalid chunk, we track these "inverted" scores,
    #    and set sample-level badness to max(inverted_invalid_scores).
    #    If no invalid chunk => set to min_inverted_score (e.g. 0.0).
    badness_scores = []
    for sample in calibration_data:
        # The sample must contain "atomic_facts"
        if "atomic_facts" not in sample or not isinstance(sample["atomic_facts"], list):
            print(f"Warning: 'atomic_facts' missing or not a list for sample pid={sample.get('pid')}")
            continue

        # Collect the *inverted* scores of all invalid chunks
        inverted_scores_for_invalid = []
        for atom in sample["atomic_facts"]:
            if atom.get("valid") is False and isinstance(atom.get("score"), (int, float)):
                # invert the chunk's logprob
                # e.g., if chunk score = -4.3, then badness = +4.3
                inverted_score = -atom["score"]
                inverted_scores_for_invalid.append(inverted_score)

        if inverted_scores_for_invalid:
            # If there's at least one invalid chunk, define sample's badness as the maximum
            # i.e. the worst chunk among the invalid ones
            badness_value = max(inverted_scores_for_invalid)
        else:
            # No invalid chunk => set to a default min_inverted_score
            badness_value = min_inverted_score

        badness_scores.append(badness_value)

    if not badness_scores:
        raise ValueError("No valid data points found in calibration set—cannot compute threshold.")

    badness_array = np.array(badness_scores, dtype=float)

    # 3) Compute threshold as the (1 - alpha)-quantile
    #    Because "badness" is bigger for more severely incorrect responses,
    #    we pick the top (1-alpha) as the cutoff => threshold = quantile(badness, 1 - alpha)
    coverage = 1.0 - alpha
    threshold = np.quantile(badness_array, coverage)

    print(f"[INFO] Computed threshold at {coverage*100:.1f}% coverage: {threshold}")

    # 4) Save the threshold to file
    output_dir = os.path.dirname(output_threshold_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    np.savez_compressed(output_threshold_file, threshold=threshold)
    print(f"[INFO] Threshold saved to '{output_threshold_file}'")


if __name__ == "__main__":
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="Calibrate a factual conformal threshold based on labeled atomic_facts, inverting negative logprob scores.")
    parser.add_argument("--calibration_json", required=True, help="Path to the calibration JSON dataset.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Significance level (alpha). e.g., 0.1 => 90% coverage.")
    parser.add_argument("--output_threshold", help="File to store the threshold. Defaults to 'conformal_threshold.npz' in script directory.")
    parser.add_argument("--min_inverted_score", type=float, default=0.0,
                        help="Default 'badness' if no invalid chunk is found. Typically 0 => 'perfectly safe'.")
    args = parser.parse_args()
    
    # Set default output path in script directory if not specified
    if args.output_threshold is None:
        args.output_threshold = os.path.join(script_dir, "conformal_threshold.npz")
    # If relative path is given but doesn't specify directory, put it in script directory
    elif not os.path.isabs(args.output_threshold) and not os.path.dirname(args.output_threshold):
        args.output_threshold = os.path.join(script_dir, args.output_threshold)

    calibrate_factual_threshold(
        calibration_json_path=args.calibration_json,
        alpha=args.alpha,
        output_threshold_file=args.output_threshold,
        min_inverted_score=args.min_inverted_score
    )