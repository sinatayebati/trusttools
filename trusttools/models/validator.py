# Create this file: trusttools/models/conformal_validator.py

import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional

# Assuming ChatOpenAI can be imported and configured for logprobs
# You might need to adjust the import path based on your project structure
from trusttools.engine.openai import ChatOpenAI
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
    scoring their confidence using log probabilities, trimming uncertain claims
    via split conformal prediction, and merging the valid claims back.
    """
    def __init__(self, llm_engine_name: str, logprob_engine_kwargs: Optional[Dict] = None):
        """
        Initializes the ConformalValidator.

        Args:
            llm_engine_name: The name of the LLM engine to use.
            logprob_engine_kwargs: Dictionary of keyword arguments to pass to the
                                   ChatOpenAI engine specifically for requesting logprobs.
                                   Example: {'logprobs': True, 'top_logprobs': 1}
                                   Defaults to an empty dict if None.
        """
        self.llm_engine_name = llm_engine_name
        # Standard engine for tasks not requiring logprobs (separation, merging)
        self.llm_engine = ChatOpenAI(model_string=llm_engine_name, is_multimodal=False)

        # Engine specifically for getting logprobs (scoring)
        # IMPORTANT: This assumes ChatOpenAI accepts kwargs to enable logprob fetching.
        # You MUST ensure your ChatOpenAI implementation supports this.
        _logprob_kwargs = logprob_engine_kwargs if logprob_engine_kwargs is not None else {}
        self.llm_engine_logprobs = ChatOpenAI(
            model_string=llm_engine_name,
            is_multimodal=False,
            **_logprob_kwargs
        )
        print(f"Initialized ConformalValidator with engine: {llm_engine_name}")
        if _logprob_kwargs:
             print(f"Logprob engine configured with: {_logprob_kwargs}")
        else:
             print("Warning: Logprob engine not explicitly configured. Scoring might fail or use placeholders.")


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
            separation_result = self.llm_engine(prompt)
            # Extract lines starting with '- '
            atoms = [line.strip()[2:].strip() for line in separation_result.strip().split('\n') if line.strip().startswith('- ')]
            # Basic filtering for empty strings that might slip through
            atoms = [atom for atom in atoms if atom]
            print(f"Separator extracted {len(atoms)} atoms.")
            return atoms
        except Exception as e:
            print(f"Error during separation LLM call: {e}")
            return []

    def _get_logprobs_for_atom(self, atom_text: str) -> Optional[List[float]]:
        """
        Retrieves token log probabilities for a given atom text.

        *** Placeholder Implementation ***
        This is the most complex part. You need to replace this with a robust
        method based on your LLM engine's capabilities.

        Option 1 (Requires Engine Support & Mapping): Get logprobs for the *original* response,
                   then map the atom_text span back to those tokens/logprobs.
        Option 2 (Simpler but Slow & Context-Lossy): Regenerate the atom asking for logprobs.
                   This is implemented below as a placeholder.

        Returns:
            A list of log probabilities for the tokens in the atom, or None if unavailable.
        """
        if not hasattr(self.llm_engine_logprobs, 'get_logprobs'):
             print(f"Warning: Logprob engine {self.llm_engine_name} does not appear to support logprob fetching via a 'get_logprobs' method. Cannot score accurately.")
             # Return a placeholder score (e.g., random) if you want the pipeline to run
             # return [np.random.rand()] # Example placeholder
             return None # Indicate failure

        try:
            # Assuming llm_engine_logprobs can take text and return logprobs directly
            # The exact method call (`get_logprobs`) and its return format depend HEAVILY
            # on how you implement logprob support in ChatOpenAI.
            # This might involve returning a richer object than just a string.
            logprob_data = self.llm_engine_logprobs.get_logprobs(atom_text) # This method is hypothetical!

            if logprob_data and isinstance(logprob_data, list):
                 # Assuming logprob_data is directly the list of token logprobs
                 return logprob_data
            else:
                 # Handle cases where logprobs aren't returned as expected
                 print(f"Warning: Could not retrieve valid logprobs for atom: {atom_text[:50]}...")
                 return None

        except Exception as e:
            print(f"Error retrieving logprobs for atom '{atom_text[:50]}...': {e}")
            return None


    def _calculate_score_from_logprobs(self, logprobs: Optional[List[float]]) -> Optional[float]:
        """Calculates a score (mean logprob) from log probabilities."""
        if logprobs is None or len(logprobs) == 0:
            return None
        # Using mean log probability. Ensure it's robust against empty lists.
        score = np.mean(logprobs)
        # Clamp score to avoid -inf issues if needed, though mean should handle empty lists returning None.
        # score = max(score, -1e9) # Optional: Set a floor if needed
        return score


    def score_atoms(self, atom_texts: List[str]) -> List[Atom]:
        """
        Scores each atom text based on its estimated log probabilities.
        """
        atoms = []
        print(f"Scoring {len(atom_texts)} atoms...")
        for i, text in enumerate(atom_texts):
            print(f"  Scoring atom {i+1}/{len(atom_texts)}: '{text[:60]}...'")
            atom_logprobs = self._get_logprobs_for_atom(text)
            atom_score = self._calculate_score_from_logprobs(atom_logprobs)
            atoms.append(Atom(text=text, score=atom_score, logprobs=atom_logprobs))
            print(f"    Score: {atom_score}")
        return atoms

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
        # Lower scores are less confident, so we keep scores >= threshold.
        threshold = np.quantile(valid_scores, alpha)
        print(f"  Calculated threshold: {threshold} ({len(valid_scores)} scores)")

        trimmed_atoms = []
        kept_count = 0
        for atom in atoms:
            if atom.score is None or atom.score < threshold:
                atom.valid = False
            else:
                atom.valid = True
                kept_count += 1
            trimmed_atoms.append(atom)
            print(f"  Atom: '{atom.text[:60]}...' Score: {atom.score}, Valid: {atom.valid}")

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
            merged_response = self.llm_engine(prompt)
            print("Merging successful.")
            return merged_response.strip()
        except Exception as e:
            print(f"Error during merging LLM call: {e}")
            # Fallback: just join the sentences simply?
            return " ".join(valid_atoms_text)


    def validate(self, response: str, alpha: float = 0.1) -> Tuple[str, List[Atom]]:
        """
        Validates an LLM response using the separate-score-trim-merge pipeline.

        Args:
            response: The original LLM response string.
            alpha: The desired significance level for trimming (e.g., 0.1).

        Returns:
            A tuple containing:
            - The validated response string.
            - A list of Atom objects with their scores and validity status.
        """
        print("\n--- Starting Conformal Validation ---")
        print(f"Original Response: {response[:100]}...")
        print(f"Alpha: {alpha}")

        # 1. Separate
        atom_texts = self.separate(response)
        if not atom_texts:
            print("Validation failed: Could not separate response into atoms.")
            return response, [] # Return original response if separation fails

        # 2. Score
        atoms_with_scores = self.score_atoms(atom_texts)
        # Check if scoring produced any valid scores
        if all(atom.score is None for atom in atoms_with_scores):
             print("Validation failed: Could not score any atoms (likely logprob issue).")
             # Decide fallback: return original, empty string, or original with all atoms marked invalid?
             for atom in atoms_with_scores: atom.valid = False
             return response, atoms_with_scores # Return original, but atoms list shows invalidity


        # 3. Trim
        trimmed_atoms = self.trim_split_conformal(atoms_with_scores, alpha)

        # 4. Merge
        validated_response = self.merge(trimmed_atoms)

        print("--- Conformal Validation Complete ---")
        print(f"Validated Response: {validated_response[:100]}...")

        return validated_response, trimmed_atoms
