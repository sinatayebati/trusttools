"""
calibrate_factual_threshold.py

Usage (example):
  python calibrate_factual_threshold.py \
      --calibration_json path/to/calibration_data.json \
      --alpha 0.1 \
      --required_accuracy 0.9 \
      --output_threshold path/to/conformal_threshold.npz

Description:
  Implements the exact conformal prediction method from "Language Models with Conformal Factuality Guarantees".
  
  Calibration phase:
  1. For each sample, compute r_score: minimum threshold needed to achieve required_accuracy fraction correct
  2. Take (1-alpha) quantile of r_scores as the conformal threshold
  
  Test phase (in validator):
  1. Apply threshold directly: keep subclaims where score >= threshold
  2. Conformal guarantee: with probability (1-alpha), ≥ required_accuracy fraction will be correct
"""

import argparse
import json
import numpy as np
import os
import sys
from math import ceil

def get_r_score(sample, required_accuracy: float, score_field: str = "score"):
    """
    Compute the r_score for a sample: minimum threshold needed to achieve 
    required_accuracy fraction of correct subclaims.
    
    Exactly follows the paper's get_r_score function.
    
    Args:
        sample: Sample with "atomic_facts" list containing score and valid fields
        required_accuracy: Required fraction correct (parameter 'a' in paper)
        score_field: Field name for confidence scores
    
    Returns:
        r_score: minimum threshold, or -100000 if impossible
    """
    atomic_facts = sample.get("atomic_facts", [])
    if not atomic_facts:
        return -100000
    
    # Get all confidence scores with noise (as in paper)
    scores_with_noise = []
    for atom in atomic_facts:
        if score_field in atom and isinstance(atom[score_field], (int, float)):
            # Add small noise to break ties (exactly as in paper)
            noise = np.random.normal(0, 0.001)
            score_with_noise = atom[score_field] + noise
            # CRITICAL FIX: Clip to valid range [0,1] to prevent threshold > 1.0
            score_with_noise = np.clip(score_with_noise, 0.0, 1.0)
            scores_with_noise.append(score_with_noise)
        else:
            return -100000  # Invalid data
    
    # Create threshold candidates sorted in descending order (as in paper)
    threshold_set = sorted(set(scores_with_noise), reverse=True)
    
    # CRITICAL FIX: Check if sample is "easy" (all subclaims are valid)
    all_valid = all(atomic_facts[i].get("valid") is True for i in range(len(atomic_facts)))
    if all_valid:
        # If all subclaims are valid, any threshold achieves 100% accuracy
        # Return lowest possible threshold to accept all subclaims
        return min(scores_with_noise)
    
    curr_threshold = threshold_set[0]
    for threshold in threshold_set:
        curr_threshold = threshold
        
        # Apply threshold: accept subclaims with score >= threshold
        accepted_indices = [
            i for i, score in enumerate(scores_with_noise) 
            if score >= threshold
        ]
        
        # CRITICAL FIX: Handle empty acceptance set like the paper
        if not accepted_indices:
            # No subclaims accepted = 100% accuracy (vacuous truth)
            entailed_fraction = 1.0
        else:
            # Compute fraction of accepted subclaims that are correct
            correct_count = sum(
                1 for i in accepted_indices 
                if atomic_facts[i].get("valid") is True
            )
            entailed_fraction = correct_count / len(accepted_indices)
        
        # If fraction drops below required_accuracy, return current threshold
        if entailed_fraction < required_accuracy:
            return curr_threshold
    
    # If we never dropped below required_accuracy, return lowest threshold
    return min(scores_with_noise)


def compute_threshold(alpha: float, calibration_data: list, required_accuracy: float, score_field: str = "score"):
    """
    Computes the conformal prediction threshold exactly as in the paper.
    
    Args:
        alpha: Significance level (e.g., 0.1 for 90% coverage)
        calibration_data: List of calibration samples
        required_accuracy: Required fraction correct (parameter 'a')
        score_field: Field name for confidence scores
    
    Returns:
        threshold: Conformal prediction threshold
        r_scores: List of computed r_scores for analysis
    """
    # Compute r_score for each sample (exactly as in paper)
    r_scores = [get_r_score(sample, required_accuracy, score_field) for sample in calibration_data]
    
    # Remove invalid r_scores
    valid_r_scores = [r for r in r_scores if r != -100000]
    
    if not valid_r_scores:
        raise ValueError("No valid r_scores computed from calibration data")
    
    # Compute threshold using exact formula from paper:
    # quantile_target_index = ceil((n+1)*(1-alpha))
    # Map to array index by subtracting 1 for zero-indexing
    quantile_target_index = ceil((len(valid_r_scores) + 1) * (1 - alpha))
    threshold = sorted(valid_r_scores)[quantile_target_index - 1]
    
    return threshold, valid_r_scores


def calibrate_factual_threshold(
    calibration_json_path: str,
    alpha: float,
    output_threshold_file: str,
    required_accuracy: float = 0.9,
    score_field: str = "score"
):
    """
    Calibrates conformal prediction threshold using exact method from the paper.
    
    Args:
        calibration_json_path: Path to calibration JSON with atomic_facts
        alpha: Significance level (e.g. 0.1 means 90% coverage)
        output_threshold_file: Output .npz file path
        required_accuracy: Required fraction correct (parameter 'a')
        score_field: Field name for confidence scores in atomic_facts
    """
    if not os.path.exists(calibration_json_path):
        raise FileNotFoundError(f"Calibration JSON file not found: {calibration_json_path}")

    print(f"[INFO] Loading calibration data from {calibration_json_path}")
    print(f"[INFO] Parameters: alpha={alpha}, required_accuracy={required_accuracy}")
    
    # Load calibration data
    with open(calibration_json_path, "r") as f:
        calibration_data = json.load(f)

    # Validate data format and collect stats
    total_samples = len(calibration_data)
    valid_samples = 0
    total_atoms = 0
    valid_atoms = 0
    invalid_atoms = 0
    
    for sample in calibration_data:
        if "atomic_facts" not in sample or not isinstance(sample["atomic_facts"], list):
            continue
        
        has_valid_data = True
        for atom in sample["atomic_facts"]:
            if score_field not in atom or "valid" not in atom:
                has_valid_data = False
                break
        
        if has_valid_data:
            valid_samples += 1
            for atom in sample["atomic_facts"]:
                total_atoms += 1
                if atom.get("valid") is True:
                    valid_atoms += 1
                elif atom.get("valid") is False:
                    invalid_atoms += 1
    
    print(f"[INFO] Data statistics:")
    print(f"  - Total samples: {total_samples}")
    print(f"  - Valid samples: {valid_samples}")
    print(f"  - Total atoms: {total_atoms}")
    print(f"  - Valid atoms: {valid_atoms} ({100*valid_atoms/total_atoms:.1f}%)")
    print(f"  - Invalid atoms: {invalid_atoms} ({100*invalid_atoms/total_atoms:.1f}%)")
    
    if valid_samples == 0:
        raise ValueError("No valid samples found in calibration data")

    # Compute conformal threshold using paper's method
    threshold, r_scores = compute_threshold(alpha, calibration_data, required_accuracy, score_field)
    
    print(f"[INFO] R-score statistics:")
    print(f"  - Number of valid r_scores: {len(r_scores)}")
    print(f"  - Mean: {np.mean(r_scores):.4f}")
    print(f"  - Std: {np.std(r_scores):.4f}")
    print(f"  - Min: {np.min(r_scores):.4f}")
    print(f"  - Max: {np.max(r_scores):.4f}")
    print(f"  - 25th percentile: {np.percentile(r_scores, 25):.4f}")
    print(f"  - 50th percentile (median): {np.percentile(r_scores, 50):.4f}")
    print(f"  - 75th percentile: {np.percentile(r_scores, 75):.4f}")
    print(f"  - 95th percentile: {np.percentile(r_scores, 95):.4f}")
    
    # Check for potentially problematic thresholds
    scores_above_one = sum(1 for r in r_scores if r > 1.0)
    if scores_above_one > 0:
        print(f"[WARNING] {scores_above_one}/{len(r_scores)} r_scores are > 1.0 (this shouldn't happen after fixes)")
    
    print(f"[INFO] Conformal threshold (at {100*(1-alpha):.1f}% coverage): {threshold:.6f}")
    
    # Additional diagnostic info
    if threshold > 1.0:
        print(f"[WARNING] Threshold > 1.0 detected! This will reject all subclaims.")
        print(f"[DEBUG] Consider using lower required_accuracy or checking calibration data quality.")
    elif threshold > 0.95:
        print(f"[INFO] High threshold detected ({threshold:.4f}). This will be quite conservative.")
    
    # Count "easy" samples (all valid subclaims)
    easy_samples = 0
    for sample in calibration_data:
        if "atomic_facts" in sample:
            all_valid = all(atom.get("valid") is True for atom in sample["atomic_facts"])
            if all_valid:
                easy_samples += 1
    
    if easy_samples > len(calibration_data) * 0.8:
        print(f"[WARNING] {easy_samples}/{len(calibration_data)} ({100*easy_samples/len(calibration_data):.1f}%) samples have all valid subclaims.")
        print(f"[WARNING] High-quality calibration data may lead to overly conservative thresholds.")

    # Save threshold and metadata
    output_dir = os.path.dirname(output_threshold_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    np.savez_compressed(output_threshold_file, 
                       threshold=threshold,
                       alpha=alpha,
                       required_accuracy=required_accuracy,
                       num_samples=len(r_scores),
                       r_scores=r_scores)
    print(f"[INFO] Threshold saved to '{output_threshold_file}'")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="Calibrate conformal prediction threshold using exact method from the paper.")
    parser.add_argument("--calibration_json", required=True, help="Path to calibration JSON dataset.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Significance level (e.g., 0.1 => 90% coverage).")
    parser.add_argument("--required_accuracy", type=float, default=0.9, help="Required fraction correct (parameter 'a' in paper).")
    parser.add_argument("--output_threshold", help="Output threshold file (default: conformal_threshold.npz in script dir).")
    parser.add_argument("--score_field", default="score", help="Field name for confidence scores in atomic_facts.")
    args = parser.parse_args()
    
    # Validate parameters
    if not (0 < args.required_accuracy <= 1):
        raise ValueError(f"required_accuracy must be in (0, 1], got {args.required_accuracy}")
    if not (0 < args.alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {args.alpha}")
    
    # Set default output path
    if args.output_threshold is None:
        args.output_threshold = os.path.join(script_dir, "conformal_threshold.npz")
    elif not os.path.isabs(args.output_threshold) and not os.path.dirname(args.output_threshold):
        args.output_threshold = os.path.join(script_dir, args.output_threshold)

    calibrate_factual_threshold(
        calibration_json_path=args.calibration_json,
        alpha=args.alpha,
        output_threshold_file=args.output_threshold,
        required_accuracy=args.required_accuracy,
        score_field=args.score_field
    )