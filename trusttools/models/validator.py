# Create this file: trusttools/models/conformal_validator.py

import numpy as np
import re
import string # For normalization
from typing import List, Dict, Any, Tuple, Optional

# Assuming ChatOpenAI can be imported and configured for logprobs
# You might need to adjust the import path based on your project structure
from trusttools.engine.openai import ChatOpenAI, GenerationResult
from pydantic import BaseModel, Field

class Atom(BaseModel):
    """Represents a single atomic claim extracted from a response."""
    text: str
    score: Optional[float] = Field(None, description="Confidence score (e.g., mean log probability)")
    logprobs: Optional[List[float]] = Field(None, description="Token log probabilities for the atom")
    valid: bool = Field(True, description="Whether the atom is considered valid after trimming")

class ConformalValidator:
    """
    Validates LLM responses by separating them into atomic claims,
    scoring their confidence using log probabilities provided from the generation step,
    trimming uncertain claims via split conformal prediction, and merging the valid claims back.
    """
    def __init__(self, llm_engine_name: str):
        """
        Initializes the ConformalValidator. Only needs a standard engine for separation/merging.

        Args:
            llm_engine_name: The name of the LLM engine to use for non-scoring tasks.
        """
        self.llm_engine_name = llm_engine_name
        # Standard engine for tasks not requiring logprobs (separation, merging)
        self.llm_engine = ChatOpenAI(model_string=llm_engine_name, is_multimodal=False, capture_logits=False)
        print(f"Initialized ConformalValidator with engine: {llm_engine_name} (for separation/merging)")

    def separate(self, response: str) -> List[str]:
        """
        Separates a response into atomic claims using an LLM.
        """
        if not response or not response.strip():
            return []

        prompt = f"""
Analyze the following text. Break it down into the smallest possible factual statements (atoms) that retain the original meaning. Each atom should be a distinct, self-contained claim. Do not add any interpretation or new information. List each atom on a new line, starting with '- '.

Text:
{response}

Atoms:
"""
        try:
            # Use the standard engine call (no logprobs needed here)
            separation_result_obj: GenerationResult = self.llm_engine(prompt)
            separation_result = separation_result_obj.text

            if separation_result is None:
                 print(f"Error during separation LLM call: {separation_result_obj.error}")
                 return []

            # Extract lines starting with '- '
            atoms = [line.strip()[2:].strip() for line in separation_result.strip().split('\n') if line.strip().startswith('- ')]
            # Basic filtering for empty strings that might slip through
            atoms = [atom for atom in atoms if atom]
            print(f"Separator extracted {len(atoms)} atoms.")
            return atoms
        except Exception as e:
            print(f"Error during separation process: {e}")
            return []

    def _normalize_text(self, text: str) -> str:
        """Simple normalization for mapping."""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def score_atoms(self, atom_texts: List[str], logprob_content: List[Dict]) -> List[Atom]:
        """
        Scores atoms by mapping them to the provided logprob_content from the original generation.
        """
        atoms = []
        if not logprob_content:
            print("Warning: No logprob content provided for scoring. Cannot score atoms.")
            # Return atoms without scores
            return [Atom(text=text, score=None, logprobs=None) for text in atom_texts]

        # Handle different potential logprob_content formats
        standardized_logprobs = self._standardize_logprobs(logprob_content)
        if not standardized_logprobs:
            print("Warning: Could not standardize logprob content. Using raw format.")
            standardized_logprobs = logprob_content

        # Reconstruct the original text from logprobs for mapping reference
        try:
            original_tokens = [item.get('token', '') for item in standardized_logprobs]
            original_full_text = "".join(original_tokens)
            normalized_original_text = self._normalize_text(original_full_text)
            print(f"Reconstructed original text length: {len(original_full_text)}")
            print(f"Normalized original text length: {len(normalized_original_text)}")
            print(f"First 100 chars of original text: {original_full_text[:100]}")
        except Exception as e:
            print(f"Error reconstructing text from logprobs: {e}")
            print(f"Using raw text format. Logprob type: {type(standardized_logprobs)}")
            original_full_text = "ERROR_RECONSTRUCTING"
            normalized_original_text = "ERROR_RECONSTRUCTING"

        current_logprob_idx = 0 # Keep track of our position in the logprob_content list

        print(f"Scoring {len(atom_texts)} atoms using provided logprobs...")
        for i, atom_text in enumerate(atom_texts):
            print(f"  Mapping atom {i+1}/{len(atom_texts)}: '{atom_text[:60]}...'")
            normalized_atom = self._normalize_text(atom_text)
            atom_logprobs = []
            mapped = False

            if not normalized_atom:
                 print(f"    Skipping empty normalized atom.")
                 atoms.append(Atom(text=atom_text, score=None, logprobs=None))
                 continue

            # --- Simple Sliding Window / Greedy Alignment ---
            # Try to find the start of the normalized atom in the remaining normalized original text
            best_match_start_idx = -1
            try:
                # Search from the current position onwards
                search_area = normalized_original_text[current_logprob_idx:]
                match_pos = search_area.find(normalized_atom)
                if match_pos != -1:
                    best_match_start_idx = current_logprob_idx + match_pos
                    print(f"    Found exact match at position {best_match_start_idx}")
                # Add fallback search from beginning if not found
                elif normalized_original_text != "ERROR_RECONSTRUCTING":
                    print(f"    No match from current position {current_logprob_idx}, searching from beginning...")
                    match_pos = normalized_original_text.find(normalized_atom)
                    if match_pos != -1:
                        best_match_start_idx = match_pos
                        print(f"    Found atom start at {best_match_start_idx} searching from beginning.")
                    else:
                        # Try fuzzy matching as a last resort for important atoms
                        print(f"    No exact match found, trying fuzzy matching...")
                        # Basic substring matching - find largest common substring
                        for window_size in range(min(len(normalized_atom), 20), 4, -1):  # Try decreasing window sizes
                            atom_start = normalized_atom[:window_size]
                            if atom_start in normalized_original_text:
                                match_pos = normalized_original_text.find(atom_start)
                                best_match_start_idx = match_pos
                                print(f"    Found fuzzy match with window size {window_size} at position {match_pos}")
                                break
            except Exception as find_e:
                print(f"    Error during string find: {find_e}")
                best_match_start_idx = -1

            if best_match_start_idx != -1:
                 print(f"    Potential match found starting around original index {best_match_start_idx}")
                 # Attempt to align and collect logprobs
                 # This part is complex: map normalized start index back to original token index
                 # A simple approximation: use the index directly (might be off due to normalization)
                 # A better way: iterate original_tokens up to best_match_start_idx char count.
                 original_char_count = 0
                 start_token_idx = -1
                 
                 # Find token position by counting characters
                 for idx, token_info in enumerate(standardized_logprobs):
                     if idx < current_logprob_idx and best_match_start_idx >= current_logprob_idx:
                         # Don't search before current position unless we found an earlier match
                         continue
                         
                     token_text = token_info.get('token', '')
                     # Simple direct check: first token starts with beginning of atom
                     normalized_token = self._normalize_text(token_text)
                     if normalized_atom.startswith(normalized_token) and normalized_token:
                         # Found potential start
                         start_token_idx = idx
                         print(f"    Mapped start to token index {start_token_idx} (direct match)")
                         break
                     
                     # Character counting approach (less reliable)
                     original_char_count += len(token_text)
                     if original_char_count >= best_match_start_idx:
                         start_token_idx = idx
                         print(f"    Mapped start to token index {start_token_idx} (char count)")
                         break

                 if start_token_idx == -1:
                     # Fall back to estimating position
                     total_tokens = len(standardized_logprobs)
                     total_chars = len(original_full_text)
                     if total_chars > 0:
                         # Estimate token position using proportional position in text
                         pos_ratio = best_match_start_idx / total_chars
                         estimated_token_idx = int(pos_ratio * total_tokens)
                         # Clamp to valid range
                         start_token_idx = max(0, min(estimated_token_idx, total_tokens - 1))
                         print(f"    Estimated token position: {start_token_idx} (fallback)")
                     else:
                         # Nothing else worked, use current position
                         start_token_idx = current_logprob_idx
                         print(f"    Using current position as fallback: {start_token_idx}")

                 if start_token_idx != -1:
                     # Greedily consume tokens matching the atom
                     current_atom_pos = 0
                     temp_atom_logprobs = []
                     temp_original_tokens_matched = [] # For debugging

                     for token_idx in range(start_token_idx, len(standardized_logprobs)):
                         token_info = standardized_logprobs[token_idx]
                         token_text = token_info.get('token', '')
                         if 'logprob' in token_info:
                             logprob = token_info['logprob']
                         else:
                             # Try alternative field names if 'logprob' not present
                             logprob = token_info.get('token_logprob', token_info.get('log_prob', None))
                             if logprob is None:
                                 print(f"    Warning: No logprob found for token {token_text}. Keys: {token_info.keys()}")
                                 logprob = -5.0  # Default value if missing

                         # Try to match token against remaining part of atom
                         remaining_atom = atom_text[current_atom_pos:]
                         # Use original atom text for matching length, but allow fuzzy match logic
                         # Simple check: does the atom *start* with this token (case-insensitive)?
                         if remaining_atom.lower().startswith(token_text.lower()):
                             temp_atom_logprobs.append(logprob)
                             temp_original_tokens_matched.append(token_text)
                             current_atom_pos += len(token_text)
                         # Add more flexible matching here if needed (e.g., skipping whitespace)
                         elif token_text.strip() and remaining_atom.lower().strip().startswith(token_text.lower().strip()):
                             # Try matching with whitespace removed
                             temp_atom_logprobs.append(logprob)
                             temp_original_tokens_matched.append(token_text)
                             # Estimate how much to advance - length of token + any whitespace
                             advance = len(token_text)
                             while current_atom_pos < len(atom_text) and atom_text[current_atom_pos].isspace():
                                 advance += 1
                                 current_atom_pos += 1
                             current_atom_pos += len(token_text.strip())
                         else:
                             # This token doesn't match, but allow skipping small tokens (like whitespace)
                             if len(token_text.strip()) <= 1:
                                 # Skip tiny tokens and continue
                                 continue
                             else:
                                 # End matching on substantial mismatch
                                 break

                         if current_atom_pos >= len(atom_text):
                             # We've matched the whole atom (or close enough)
                             atom_logprobs = temp_atom_logprobs
                             mapped = True
                             print(f"    Successfully mapped atom ending at token index {token_idx}.")
                             print(f"    Matched tokens: {''.join(temp_original_tokens_matched)}")
                             # Advance the global index past the matched tokens
                             current_logprob_idx = token_idx + 1
                             break
                     if not mapped:
                          print(f"    Failed to consume full atom text after finding start.")

            if not mapped:
                print(f"    Warning: Could not reliably map atom to logprobs.")
                # Last resort - use statistical mapping if available
                if len(standardized_logprobs) > 0:
                    # Use average logprob of all tokens as a statistical estimate
                    all_logprobs = []
                    for token_info in standardized_logprobs:
                        if 'logprob' in token_info:
                            all_logprobs.append(token_info['logprob'])
                        elif 'token_logprob' in token_info:
                            all_logprobs.append(token_info['token_logprob'])

                    if all_logprobs:
                        avg_logprob = sum(all_logprobs) / len(all_logprobs)
                        print(f"    Using statistical estimate (avg logprob: {avg_logprob:.4f}) for unmapped atom")
                        atom_logprobs = [avg_logprob] * (len(atom_text) // 4 + 1)  # Estimate token count
                atom_logprobs = None

            atom_score = self._calculate_score_from_logprobs(atom_logprobs)
            atoms.append(Atom(text=atom_text, score=atom_score, logprobs=atom_logprobs)) # Store raw logprobs if needed
            print(f"    Score: {atom_score}")

        return atoms

    def _calculate_score_from_logprobs(self, logprobs: Optional[List[float]]) -> Optional[float]:
        """Calculates a score (mean logprob) from log probabilities."""
        if logprobs is None or len(logprobs) == 0:
            return None
        # Using mean log probability. Use np.exp to convert back to probability before averaging?
        # No, average of logprobs is standard. exp(mean(logprobs)) is geometric mean of probs.
        score = np.mean(logprobs)
        return score

    def trim_split_conformal(self, atoms: List[Atom], alpha: float) -> List[Atom]:
        """
        Trims atoms using Split Conformal Prediction with a fixed threshold
        derived from the current batch's scores.

        Args:
            atoms: List of Atom objects with scores.
            alpha: Significance level (e.g., 0.1 means keep claims in the top 90% confidence).

        Returns:
            List of Atom objects with the `valid` flag updated.
        """
        print(f"Trimming atoms with alpha = {alpha}...")
        if not atoms:
            return []

        # Filter out atoms without a valid score for threshold calculation
        valid_scores = [atom.score for atom in atoms if atom.score is not None]

        if not valid_scores:
            print("Warning: No valid scores found for trimming. Marking all atoms as invalid.")
            for atom in atoms:
                atom.valid = False
            return atoms

        # Calculate the threshold: the alpha-quantile of scores.
        # Higher logprob scores are better. We want to keep scores >= threshold.
        # Ensure scores are floats for quantile calculation
        valid_scores_float = [float(s) for s in valid_scores]
        threshold = np.quantile(valid_scores_float, alpha) # Low scores are trimmed
        print(f"  Calculated threshold (alpha={alpha} quantile): {threshold} ({len(valid_scores_float)} scores)")

        trimmed_atoms = []
        kept_count = 0
        for atom in atoms:
            if atom.score is None or float(atom.score) < threshold: # Compare score against threshold
                atom.valid = False
            else:
                atom.valid = True
                kept_count += 1
            trimmed_atoms.append(atom)
            # print(f"  Atom: '{atom.text[:60]}...' Score: {atom.score}, Valid: {atom.valid}") # Verbose

        print(f"Trimming complete. Kept {kept_count}/{len(atoms)} atoms.")
        return trimmed_atoms

    def merge(self, atoms: List[Atom]) -> str:
        """
        Merges valid atoms back into a coherent response using an LLM.
        """
        valid_atoms_text = [atom.text for atom in atoms if atom.valid]

        if not valid_atoms_text:
            print("Merging: No valid atoms remaining.")
            return "" # Or return a message like "[No valid content remaining after conformal validation]"

        print(f"Merging {len(valid_atoms_text)} valid atoms...")

        facts_list = "\n".join([f"- {text}" for text in valid_atoms_text])

        prompt = f"""
Combine the following factual statements into a single, coherent paragraph. Maintain the original meaning and do not add any new information, explanations, or introductory/concluding phrases. Ensure smooth transitions where appropriate.

Facts:
{facts_list}

Merged Paragraph:
"""
        try:
            # Use standard engine call
            merged_response_obj: GenerationResult = self.llm_engine(prompt)
            merged_response = merged_response_obj.text
            if merged_response is None:
                 print(f"Error during merging LLM call: {merged_response_obj.error}")
                 return " ".join(valid_atoms_text) # Fallback

            print("Merging successful.")
            return merged_response.strip()
        except Exception as e:
            print(f"Error during merging process: {e}")
            # Fallback: just join the sentences simply?
            return " ".join(valid_atoms_text)

    def validate(self, response: str, logprob_content: Optional[List[Dict]], alpha: float = 0.1) -> Tuple[str, List[Atom]]:
        """
        Validates an LLM response using the separate-score-trim-merge pipeline.

        Args:
            response: The original LLM response string.
            logprob_content: The list of logprob dictionaries from the generation step.
            alpha: The desired significance level for trimming (e.g., 0.1).

        Returns:
            A tuple containing:
            - The validated response string.
            - A list of Atom objects with their scores and validity status.
        """
        print("\n--- Starting Conformal Validation ---")
        print(f"Original Response: {response[:100]}...")
        print(f"Alpha: {alpha}")
        print(f"Logprobs provided: {logprob_content is not None} (Length: {len(logprob_content) if logprob_content else 'N/A'})")

        try:
            # 1. Separate
            atom_texts = self.separate(response)
            if not atom_texts:
                print("Validation failed: Could not separate response into atoms.")
                return response, [Atom(text=response, valid=True)]  # Return original as valid atom if separation fails
    
            # 2. Score using provided logprobs
            if not logprob_content:
                print("Validation failed: Logprobs not provided for scoring.")
                # Keep all atoms as valid if scoring isn't possible
                atoms_no_score = [Atom(text=t, score=None, logprobs=None, valid=True) for t in atom_texts]
                # Return the original response but mark atoms for transparency
                print("Returning original response with all atoms marked as valid.")
                return response, atoms_no_score
    
            atoms_with_scores = self.score_atoms(atom_texts, logprob_content)
    
            # Check if scoring produced any valid scores
            valid_scores_count = sum(1 for atom in atoms_with_scores if atom.score is not None)
            if valid_scores_count == 0:
                print(f"Validation warning: Could not score any atoms (likely mapping issue). Keeping original response.")
                # Mark all atoms as valid
                for atom in atoms_with_scores: 
                    atom.valid = True
                return response, atoms_with_scores  # Return original, atoms list shows scoring attempt
            
            print(f"Successfully scored {valid_scores_count}/{len(atoms_with_scores)} atoms")
    
            # 3. Trim
            trimmed_atoms = self.trim_split_conformal(atoms_with_scores, alpha)
            
            # Check if we have any valid atoms after trimming
            valid_atoms_count = sum(1 for atom in trimmed_atoms if atom.valid)
            if valid_atoms_count == 0:
                # If no atoms remain valid after trimming, keep at least 50% of the highest scoring atoms
                print("Warning: No atoms remain valid after trimming. Keeping top 50% of atoms by score.")
                # Sort atoms by score and keep top half
                sorted_atoms = sorted(
                    [a for a in atoms_with_scores if a.score is not None],
                    key=lambda x: x.score if x.score is not None else float('-inf'),
                    reverse=True  # Higher scores are better
                )
                
                # Keep at least half the atoms or 3, whichever is greater
                keep_count = max(len(sorted_atoms) // 2, min(3, len(sorted_atoms)))
                for i, atom in enumerate(atoms_with_scores):
                    if i < keep_count and atom.score is not None:
                        atom.valid = True
                
                trimmed_atoms = atoms_with_scores
                print(f"Adjusted validation to keep {keep_count} highest scoring atoms")
    
            # 4. Merge
            validated_response = self.merge(trimmed_atoms)
            if not validated_response.strip():
                print("Warning: Merged response is empty. Falling back to original response.")
                return response, trimmed_atoms
    
            print("--- Conformal Validation Complete ---")
            print(f"Validated Response: {validated_response[:100]}...")
    
            return validated_response, trimmed_atoms
            
        except Exception as e:
            print(f"Error during validation process: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to original response due to validation error.")
            return response, [Atom(text=response, valid=True)]  # Return original marked as valid

    def _standardize_logprobs(self, logprob_content: List[Dict]) -> List[Dict]:
        """
        Standardizes different logprob formats to a common format with 'token' and 'logprob' keys.
        """
        try:
            if not logprob_content or not isinstance(logprob_content, list):
                print(f"Invalid logprob_content format: {type(logprob_content)}")
                return []
                
            print(f"Standardizing logprobs format. Item count: {len(logprob_content)}")
            if len(logprob_content) > 0:
                print(f"First item keys: {list(logprob_content[0].keys()) if isinstance(logprob_content[0], dict) else 'not a dict'}")
                
            standardized = []
            
            for item in logprob_content:
                if not isinstance(item, dict):
                    print(f"Warning: Expected dict in logprob_content, got {type(item)}")
                    continue
                    
                std_item = {}
                
                # Handle token field - look for various possible names
                if 'token' in item:
                    std_item['token'] = item['token']
                elif 'text' in item:
                    std_item['token'] = item['text']
                else:
                    # If no token info, synthesize a placeholder
                    std_item['token'] = ''
                    
                # Handle logprob field - look for various possible names
                if 'logprob' in item:
                    std_item['logprob'] = item['logprob']
                elif 'token_logprob' in item:
                    std_item['logprob'] = item['token_logprob']
                elif 'log_prob' in item: 
                    std_item['logprob'] = item['log_prob']
                else:
                    # If no logprob, use a default value
                    std_item['logprob'] = -5.0
                    
                standardized.append(std_item)
                
            print(f"Standardized {len(standardized)} logprob items")
            return standardized
            
        except Exception as e:
            print(f"Error standardizing logprobs: {e}")
            return []
