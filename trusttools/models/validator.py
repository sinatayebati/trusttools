import numpy as np
import re
import string # For normalization
from typing import List, Dict, Any, Tuple, Optional

from trusttools.engine.openai import ChatOpenAI, GenerationResult
from pydantic import BaseModel, Field

class Atom(BaseModel):
    """Represents a single atomic claim extracted from a response."""
    text: str
    score_logprobs: Optional[float] = Field(None, description="Mean log probability score for this atom")
    score_selfeval: Optional[float] = Field(None, description="Self-evaluation score (0-1) for factuality")
    valid: Optional[bool] = Field(None, description="Whether the atom is considered valid after evaluation")
    
    def dict_without_logprobs(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the Atom without internal logprobs."""
        return {
            "sub_claim": self.text,
            "score_logprobs": self.score_logprobs,
            "score_selfeval": self.score_selfeval,
            "valid": self.valid
        }

class ConformalValidator:
    def __init__(self, llm_engine_name: str, threshold_file: Optional[str] = None):
        """
        Initializes the ConformalValidator.

        Args:
            llm_engine_name: The name of the LLM engine to use for LLM-based validation tasks.
            threshold_file: Optional path to a .npz file containing a pre-calibrated conformal threshold.
        """
        self.llm_engine_name = llm_engine_name
        # Standard engine for validation tasks (NOT multimodal)
        self.llm_engine = ChatOpenAI(model_string=llm_engine_name, is_multimodal=False, capture_logits=True)
        print(f"Initialized ConformalValidator with engine: {llm_engine_name}")
        
        # Load pre-calibrated threshold if provided
        self.calibrated_threshold = None
        if threshold_file:
            try:
                threshold_data = np.load(threshold_file)
                self.calibrated_threshold = float(threshold_data['threshold'])
                print(f"Loaded pre-calibrated threshold: {self.calibrated_threshold} from {threshold_file}")
            except Exception as e:
                print(f"Warning: Failed to load threshold from {threshold_file}: {e}")

    def extract_subclaims(self, query: str, query_analysis: str, response: str, enable_annotation: bool = False) -> List[Atom]:
        """
        Extracts sub-claims from the response using an LLM, scores them, and optionally validates them.
        Uses a structured generation approach with markers to accurately compute logprobs.
        
        Args:
            query: The original user query
            query_analysis: The query analysis from the agent
            response: The agent's response to validate
            enable_annotation: Whether to enable LLM-based annotation/validation
            
        Returns:
            List of Atom objects with text and scores
        """
        print("Extracting sub-claims from response using LLM with marker-based scoring...")
        
        try:    
            # Create prompt for LLM to extract sub-claims with structured markers
            # Prepare the valid tag section separately to avoid f-string backslash issues
            valid_tag_section = ""
            if enable_annotation:
                valid_tag_section = """<valid>
[true/false based on factuality, relevance, and necessity for answering the query]
</valid>"""
            
            extraction_prompt = f"""
You are an expert factual claim analyzer. Given a user query, query analysis, and an agent response, break down the response into a set of small, independent, and self-contained sub-claims.

IMPORTANT: You must follow this exact format to output each sub-claim:

<claim>
[The atomic statement extracted from the response]
</claim>
<score>
[A self-evaluation score between 0.0 and 1.0 for factuality and relevance. Be critical and use the full range from 0.0 to 1.0, where 0.0 means completely incorrect or irrelevant, and 1.0 means perfectly factual and relevant. Most claims should fall somewhere in between.]
</score>
{valid_tag_section}

USER QUERY:
{query}

QUERY ANALYSIS:
{query_analysis}

AGENT RESPONSE:
{response}

Instructions:
- Break down the response into the smallest meaningful sub-claims
- Ensure each sub-claim is a standalone, factual statement
- Evaluate each sub-claim independently and critically
- Use the full scoring range (0.0-1.0), not just high scores
- Be thorough in identifying all factual claims in the response
- Place each sub-claim within the exact marker tags as shown above

Begin extracting sub-claims now:

"""

            # Generate claims with LLM - IMPORTANT: Pass string, not a list to avoid multimodal processing
            try:
                # Use direct string input instead of list to avoid multimodal processing
                extraction_result: GenerationResult = self.llm_engine.generate(
                    extraction_prompt,  # Direct string, not wrapped in list [extraction_prompt]
                    capture_logits=True
                )
            except Exception as e:
                print(f"Error in LLM generation: {str(e)}")
                # Try fallback with __call__ method which handles strings differently
                try:
                    extraction_result = self.llm_engine(extraction_prompt)
                except Exception as inner_e:
                    print(f"Fallback also failed: {str(inner_e)}")
                    # Return a default atom if everything fails
                    return [Atom(text=response, score_logprobs=None, score_selfeval=0.5, valid=True)]
            
            if extraction_result.text is None:
                print(f"Error during sub-claim extraction: {extraction_result.error}")
                # Ensure we return a valid atom with default values
                return [Atom(text=response, score_logprobs=None, score_selfeval=0.5, valid=True)]
            
            generated_text = extraction_result.text
            logprobs_data = extraction_result.logprob_content
            
            if not logprobs_data:
                print("Warning: No logprobs captured during generation")
            
            # Extract claims and scores using regex
            # Create pattern for the exact structure we specified in the prompt
            claim_pattern = r'<claim>\s*(.*?)\s*</claim>\s*<score>\s*([\d\.]+)\s*</score>(?:\s*<valid>\s*(true|false)\s*</valid>)?'
            claims_matches = re.findall(claim_pattern, generated_text, re.DOTALL)
            
            if not claims_matches:
                print("Failed to extract structured claims from LLM response. Using original response as single claim.")
                return [Atom(text=response, score_logprobs=None, score_selfeval=None, valid=True)]
            
            # Calculate token positions for each claim to compute mean logprobs
            atoms = []
            token_index = 0
            
            # Extract token logprobs from the logprobs data
            token_logprobs = []
            if logprobs_data:
                # Modern format: each item has 'token' and 'logprob' fields
                if isinstance(logprobs_data, list) and logprobs_data and isinstance(logprobs_data[0], dict):
                    token_logprobs = [item.get('logprob') for item in logprobs_data if 'logprob' in item]
                # Legacy format: flat list of logprobs
                elif isinstance(logprobs_data, dict) and 'token_logprobs' in logprobs_data:
                    token_logprobs = logprobs_data['token_logprobs']
            
            # Process claims and compute per-claim logprobs
            for claim_text, score_text, valid_text in claims_matches:
                # Parse self-evaluation score
                try:
                    score_selfeval = float(score_text.strip())
                except ValueError:
                    score_selfeval = None  # Default to None if we can't parse
                
                # Only set valid field if annotation is enabled
                valid = None
                if enable_annotation and valid_text:
                    valid = valid_text.strip().lower() == 'true'
                
                # Find the position of this claim in the generated text to calculate claim-specific logprobs
                claim_with_tags = f"<claim>\n{claim_text}\n</claim>"
                claim_start_idx = generated_text.find(claim_with_tags)
                claim_end_idx = claim_start_idx + len(claim_with_tags) if claim_start_idx >= 0 else -1
                
                # Calculate claim-specific logprob if we can find the claim in the text
                claim_logprob = None
                if claim_start_idx >= 0 and logprobs_data and isinstance(logprobs_data, list):
                    # Get tokens that correspond to this claim
                    claim_tokens = []
                    token_text = ""
                    for token_info in logprobs_data:
                        if 'token' in token_info and 'logprob' in token_info:
                            token_text += token_info['token']
                            # If we're within the claim text bounds
                            if token_text.find(claim_text) >= 0:
                                claim_tokens.append(token_info['logprob'])
                    
                    # Calculate mean logprob for this specific claim
                    if claim_tokens:
                        mean_logprob = float(np.mean([lp for lp in claim_tokens if lp is not None]))
                        # map mean log-prob  (−∞,0]  →  (0,1] with a strictly-monotone exp ----------
                        prob_score = float(np.clip(np.exp(mean_logprob), 1e-12, 1.0))
                    else:
                        prob_score = None
                
                atoms.append(Atom(
                    text=claim_text.strip(),
                    score_logprobs=prob_score,
                    score_selfeval=score_selfeval,
                    valid=valid
                ))
            
            print(f"Extracted {len(atoms)} sub-claims from response")
            
            # Only use global stats as fallback if per-claim logprobs weren't calculated
            if token_logprobs:
                claims_missing_logprobs = [atom for atom in atoms if atom.score_logprobs is None]
                if claims_missing_logprobs:
                    global_mean_logprob = float(np.mean([lp for lp in token_logprobs if lp is not None]))
                    global_prob_score   = float(np.clip(np.exp(global_mean_logprob), 1e-12, 1.0))
                    for atom in claims_missing_logprobs:
                        atom.score_logprobs = global_prob_score
            
            return atoms
            
        except Exception as e:
            print(f"Error during sub-claim extraction: {e}")
            import traceback
            traceback.print_exc()
            # Return a fallback atom with the original response
            return [Atom(text=response, score_logprobs=None, score_selfeval=0.5, valid=True)]

    def trim_split_conformal(self, atoms: List[Atom], alpha: float) -> List[Atom]:
        """
        Trims atoms using Split Conformal Prediction based on either:
        1. A pre-calibrated threshold loaded from file (preferred)
        2. A batch-computed threshold if no pre-calibrated threshold is available

        Args:
            atoms: List of Atom objects with scores.
            alpha: Significance level (e.g., 0.1 means keep claims in the top 90% confidence).

        Returns:
            List of Atom objects with the `valid` flag updated.
        """
        print(f"Trimming atoms with alpha = {alpha}...")
        if not atoms:
            return []

        # Check if validation is already done
        if all(atom.valid is not None for atom in atoms):
            print("Atoms already validated, skipping conformal inference")
            return atoms

        # Filter out atoms without a valid score for threshold calculation
        valid_atoms = [atom for atom in atoms if atom.score_logprobs is not None]

        if not valid_atoms:
            print("Warning: No valid scores found for trimming. Marking all atoms as invalid.")
            for atom in atoms:
                atom.valid = False
            return atoms

        # Use pre-calibrated threshold if available
        if self.calibrated_threshold is not None:
            print(f"Using pre-calibrated threshold: {self.calibrated_threshold}")
            threshold = self.calibrated_threshold
            
            # When using the pre-calibrated threshold, we need to invert the scores
            # because the threshold was calibrated on inverted scores (higher = worse)
            trimmed_atoms = []
            kept_count = 0
            for atom in atoms:
                if atom.score_logprobs is None:
                    atom.valid = False
                else:
                    # scores are now positive and “higher = better”
                    atom.valid = atom.score_logprobs >= threshold
                    if atom.valid:
                        kept_count += 1
                trimmed_atoms.append(atom)
        else:
            # Fallback: Use self-evaluation scores if available
            print("No calibrated threshold available. Using self-evaluation scores for filtering.")
            kept_count = 0
            for atom in atoms:
                if atom.score_selfeval is not None:
                    atom.valid = atom.score_selfeval >= 0.5  # Simple threshold on self-evaluation
                else:
                    atom.valid = True  # Keep by default if no scores
                
                if atom.valid:
                    kept_count += 1
            
            trimmed_atoms = atoms
        
        print(f"Trimming complete. Kept {kept_count}/{len(atoms)} atoms.")
        return trimmed_atoms

    def merge(self, atoms: List[Atom]) -> str:
        """
        Merges valid atoms back into a coherent response using an LLM.
        """
        valid_atoms_text = [atom.text for atom in atoms if atom.valid]

        if not valid_atoms_text:
            print("Merging: No valid atoms remaining.")
            return ""

        print(f"Merging {len(valid_atoms_text)} valid atoms...")

        facts_list = "\n".join([f"- {text}" for text in valid_atoms_text])

        prompt = f"""
Combine the following factual statements into a single, coherent paragraph. Maintain the original meaning and do not add any new information, explanations, or introductory/concluding phrases. Ensure smooth transitions where appropriate.

Facts:
{facts_list}

Merged Paragraph:
"""
        try:
            merged_response_obj: GenerationResult = self.llm_engine(prompt)
            merged_response = merged_response_obj.text
            if merged_response is None:
                 print(f"Error during merging LLM call: {merged_response_obj.error}")
                 return " ".join(valid_atoms_text) # Fallback

            print("Merging successful.")
            return merged_response.strip()
        except Exception as e:
            print(f"Error during merging process: {e}")
            return " ".join(valid_atoms_text)

    def validate(self, response: str, query: str, query_analysis: str, 
                 enable_annotation: bool = False, alpha: float = 0.1) -> Tuple[str, List[Atom]]:
        """
        Validates an agent response using LLM-based claim extraction and evaluation.
        
        Args:
            response: The original agent response.
            query: The original user query.
            query_analysis: The query analysis from the agent.
            enable_annotation: Whether to enable LLM-based validation.
            alpha: The significance level for conformal trimming.
            
        Returns:
            A tuple containing:
            - The validated response string.
            - A list of Atom objects with their validation status.
        """
        print("\n--- Starting LLM-based Validation ---")
        print(f"Original Response: {response[:100]}...")
        print(f"Alpha: {alpha}")
        print(f"Annotation enabled: {enable_annotation}")
        
        try:
            # 1. Extract and score sub-claims using LLM
            atoms = self.extract_subclaims(query, query_analysis, response, enable_annotation)
            
            # 2. Apply trimming if annotation was not enabled
            if not enable_annotation:
                atoms = self.trim_split_conformal(atoms, alpha)
            
            # 3. Merge valid atoms
            valid_atoms_count = sum(1 for atom in atoms if atom.valid)
            if valid_atoms_count == 0:
                print("Warning: No atoms remain valid after evaluation. Keeping original response.")
                # Even if no atoms are valid, return the atoms with the original response
                # Also ensure every atom has the required fields
                for atom in atoms:
                    if atom.valid is None:
                        atom.valid = False
                    if atom.score_selfeval is None:
                        atom.score_selfeval = 0.0
                return response, atoms
            
            validated_response = self.merge(atoms)
            if not validated_response.strip():
                print("Warning: Merged response is empty. Falling back to original response.")
                return response, atoms
            
            print("--- LLM-based Validation Complete ---")
            print(f"Validated Response: {validated_response[:100]}...")
            
            return validated_response, atoms
            
        except Exception as e:
            print(f"Error during validation process: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to original response due to validation error.")
            # Create a valid atom with default values instead of returning None
            return response, [Atom(
                text=response, 
                score_logprobs=None, 
                score_selfeval=0.5, 
                valid=True
            )]
