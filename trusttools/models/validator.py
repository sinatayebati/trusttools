import numpy as np
import re
import string # For normalization
from typing import List, Dict, Any, Tuple, Optional

from trusttools.engine.openai import ChatOpenAI, GenerationResult
from pydantic import BaseModel, Field

class Atom(BaseModel):
    """Represents a single atomic claim extracted from a response."""
    text: str
    score: Optional[float] = Field(None, description="Confidence score (e.g., mean log probability)")
    logprobs: Optional[List[float]] = Field(None, description="Token log probabilities for the atom")
    valid: bool = Field(True, description="Whether the atom is considered valid after trimming")
    
    def dict_without_logprobs(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the Atom without the logprobs field."""
        return {
            "text": self.text,
            "score": self.score,
            "valid": self.valid
        }

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
            atoms = [atom for atom in atoms if atom]
            print(f"Separator extracted {len(atoms)} atoms.")
            return atoms
        except Exception as e:
            print(f"Error during separation process: {e}")
            return []


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
        valid_scores_float = [float(s) for s in valid_scores]
        threshold = np.quantile(valid_scores_float, alpha) # Low scores are trimmed
        print(f"  Calculated threshold (alpha={alpha} quantile): {threshold} ({len(valid_scores_float)} scores)")

        trimmed_atoms = []
        kept_count = 0
        for atom in atoms:
            if atom.score is None or float(atom.score) < threshold:
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

    def _standardize_logprobs(self, logprob_content: List[Dict], keep_top_logprobs: bool = True) -> List[Dict]:
        """
        Standardizes different logprob formats to a common format with 'token' and 'logprob' keys.
        
        Args:
            logprob_content: The list of logprob dictionaries to standardize
            keep_top_logprobs: Whether to keep the top_logprobs field (defaults to True)
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
                
                if 'token' in item:
                    std_item['token'] = item['token']
                elif 'text' in item:
                    std_item['token'] = item['text']
                else:
                    std_item['token'] = ''
                    
                if 'logprob' in item:
                    std_item['logprob'] = item['logprob']
                elif 'token_logprob' in item:
                    std_item['logprob'] = item['token_logprob']
                elif 'log_prob' in item: 
                    std_item['logprob'] = item['log_prob']
                else:
                    std_item['logprob'] = -5.0
                  
                if keep_top_logprobs and 'top_logprobs' in item:
                    std_item['top_logprobs'] = item['top_logprobs']
                
                standardized.append(std_item)
                
            print(f"Standardized {len(standardized)} logprob items" + 
                  (", excluding top_logprobs and bytes" if not keep_top_logprobs else ""))
            return standardized
            
        except Exception as e:
            print(f"Error standardizing logprobs: {e}")
            return []

    def chunk_and_score_direct(self, response: str, logprob_content: List[Dict]) -> List[Atom]:
        """
        Directly chunks and scores the original response using the logprob content.
        Uses structure-based chunking rather than LLM-based atom extraction for reliable mapping.
        
        Args:
            response: The original response text
            logprob_content: List of token-level logprob dictionaries
            
        Returns:
            List of Atom objects with text and scores
        """
        if not logprob_content or not response.strip():
            print("Warning: No logprob content provided or empty response.")
            return [Atom(text=response, score=None, logprobs=None)]
        
        # Step 1: Standardize logprobs format and remove top_logprobs to save memory
        standardized_logprobs = self._standardize_logprobs(logprob_content, keep_top_logprobs=False)
        
        if not standardized_logprobs:
            print("Warning: Could not standardize logprobs.")
            return [Atom(text=response, score=None, logprobs=None)]
        
        # Step 2: Reconstruct the full text from tokens and create token index
        reconstructed_text = ""
        token_positions = []
        
        for i, token_info in enumerate(standardized_logprobs):
            token_text = token_info['token']
            start_pos = len(reconstructed_text)
            end_pos = start_pos + len(token_text)
            
            token_positions.append({
                'index': i,
                'text': token_text,
                'start': start_pos,
                'end': end_pos,
                'logprob': token_info['logprob']
            })
            
            reconstructed_text += token_text
        
        print(f"Reconstructed text length: {len(reconstructed_text)}")
        print(f"First 100 chars: {reconstructed_text[:100]}...")
        
        # Step 3: Chunk the text using only structure-based chunking for consistency
        import re
        
        chunks = []
        
        # Start by splitting the text into paragraphs
        print("Using structure-based chunking")
        paragraphs = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Process each paragraph to extract structural elements
        for paragraph in paragraphs:
            # Check if the paragraph is a standalone header or title
            if re.match(r'^#+\s+.+$', paragraph) or re.match(r'^.+\n[=-]+$', paragraph):
                chunks.append(paragraph)
                continue
            
            # Extract numbered steps (like "1. Step description")
            # This pattern will match full numbered items with all their content
            numbered_steps = re.findall(r'^\d+\.\s+.+?(?=^\d+\.|\Z)', paragraph, re.MULTILINE | re.DOTALL)
            if numbered_steps:
                chunks.extend([step.strip() for step in numbered_steps])
                continue
            
            # Extract bulleted lists
            bulleted_items = re.findall(r'^\s*[\*\-•]\s+.+?(?=^\s*[\*\-•]|\Z)', paragraph, re.MULTILINE | re.DOTALL)
            if bulleted_items:
                chunks.extend([item.strip() for item in bulleted_items])
                continue
            
            # Extract any bold or emphasized sections (e.g., "**Section title**:")
            sections = re.split(r'(\*\*[^*]+\*\*:)', paragraph)
            if len(sections) > 1:
                current_section = ""
                for i, section in enumerate(sections):
                    if i % 2 == 1:  # It's a section header
                        if current_section:
                            chunks.append(current_section.strip())
                        current_section = section
                    else:  # It's content
                        current_section += section
                if current_section:
                    chunks.append(current_section.strip())
                continue
            
            # If no specific structure was found, use the whole paragraph
            chunks.append(paragraph)
        
        # If we still don't have any chunks, just split into sentences as fallback
        if not chunks:
            print("Falling back to simple sentence splitting")
            sentence_pattern = r'(?<=[.!?])\s+|(?<=[.!?])$'
            chunks = [s.strip() for s in re.split(sentence_pattern, response) if s.strip()]
        
        # If still no chunks, use the whole response
        if not chunks:
            print("All chunking methods failed, using whole response")
            chunks = [response]
        
        print(f"Chunked text into {len(chunks)} segments")
        
        # Step 4: Map chunks to token positions and calculate scores
        final_atoms = []
        
        for chunk in chunks:
            if not chunk.strip():
                continue
            
            # Find this chunk in the original reconstructed text
            chunk_start = reconstructed_text.find(chunk)
            
            if chunk_start != -1:
                # Direct match found
                chunk_end = chunk_start + len(chunk)
                
                # Get all tokens that overlap with this chunk
                chunk_tokens = [t for t in token_positions 
                               if (t['start'] < chunk_end and t['end'] > chunk_start)]
                
                if chunk_tokens:
                    # Calculate score as mean logprob
                    logprobs = [t['logprob'] for t in chunk_tokens]
                    score = np.mean(logprobs)
                    final_atoms.append(Atom(
                        text=chunk,
                        score=score,
                        logprobs=logprobs,
                        valid=True
                    ))
                    print(f"Mapped chunk: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}', "
                         f"Token count: {len(chunk_tokens)}, Score: {score:.4f}")
                else:
                    final_atoms.append(Atom(text=chunk, score=None, logprobs=None, valid=True))
                    print(f"No tokens mapped for chunk: '{chunk[:50]}...'")
            else:
                print(f"Could not map chunk directly: '{chunk[:50]}...'")
                final_atoms.append(Atom(text=chunk, score=None, logprobs=None, valid=True))
        
        if not final_atoms:
            print("Warning: All chunking methods failed. Treating entire response as one chunk.")
            
            # Score the entire response
            all_logprobs = [t['logprob'] for t in token_positions]
            if all_logprobs:
                overall_score = np.mean(all_logprobs)
                return [Atom(text=response, score=overall_score, logprobs=all_logprobs, valid=True)]
            else:
                return [Atom(text=response, score=None, logprobs=None, valid=True)]
        
        return final_atoms

    def _calculate_score_from_logprobs(self, logprobs: Optional[List[float]]) -> Optional[float]:
        """Calculates a score (mean logprob) from log probabilities."""
        if logprobs is None or len(logprobs) == 0:
            return None
        # Using mean log probability as the score measure
        score = np.mean(logprobs)
        return score

    def validate(self, response: str, logprob_content: Optional[List[Dict]], alpha: float = 0.1) -> Tuple[str, List[Atom]]:
        """
        Validates an LLM response using direct sentence chunking and scoring.
        
        Args:
            response: The original LLM response string.
            logprob_content: The list of logprob dictionaries from the generation step.
            alpha: The desired significance level for trimming (e.g., 0.1).
            
        Returns:
            A tuple containing:
            - The validated response string.
            - A list of Atom objects with their scores and validity status.
        """
        print("\n--- Starting Direct Conformal Validation ---")
        print(f"Original Response: {response[:100]}...")
        print(f"Alpha: {alpha}")
        print(f"Logprobs provided: {logprob_content is not None} (Length: {len(logprob_content) if logprob_content else 'N/A'})")
        
        try:
            # Check if we have logprobs to work with
            if not logprob_content:
                print("Validation not possible: No logprobs provided.")
                return response, [Atom(text=response, valid=True)]
            
            # 1. Directly chunk and score the response
            atoms_with_scores = self.chunk_and_score_direct(response, logprob_content)
            
            # Check if scoring produced any valid scores
            valid_scores_count = sum(1 for atom in atoms_with_scores if atom.score is not None)
            if valid_scores_count == 0:
                print("Validation warning: Could not score any chunks. Keeping original response.")
                for atom in atoms_with_scores:
                    atom.valid = True
                return response, atoms_with_scores
            
            print(f"Successfully scored {valid_scores_count}/{len(atoms_with_scores)} chunks")
            
            # 2. Trim low-confidence chunks
            trimmed_atoms = self.trim_split_conformal(atoms_with_scores, alpha)
            
            # Check if we have any valid atoms after trimming
            valid_atoms_count = sum(1 for atom in trimmed_atoms if atom.valid)
            if valid_atoms_count == 0:
                print("Warning: No chunks remain valid after trimming. Keeping top 50% of chunks by score.")
                # Sort atoms by score and keep top half
                sorted_atoms = sorted(
                    [a for a in atoms_with_scores if a.score is not None],
                    key=lambda x: x.score if x.score is not None else float('-inf'),
                    reverse=True  # Higher scores are better
                )
                
                # Keep at least half the atoms or 3, whichever is greater
                keep_count = max(len(sorted_atoms) // 2, min(3, len(sorted_atoms)))
                for i, atom in enumerate(sorted_atoms):
                    if i < keep_count:
                        atom.valid = True
                
                trimmed_atoms = atoms_with_scores
                print(f"Adjusted validation to keep {keep_count} highest scoring chunks")
            
            # 3. Merge valid chunks using the existing LLM-based merge method for better coherence
            validated_response = self.merge(trimmed_atoms)
            if not validated_response.strip():
                print("Warning: Merged response is empty. Falling back to original response.")
                return response, trimmed_atoms
            
            print("--- Direct Conformal Validation Complete ---")
            print(f"Validated Response: {validated_response[:100]}...")
            
            return validated_response, trimmed_atoms
            
        except Exception as e:
            print(f"Error during validation process: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to original response due to validation error.")
            return response, [Atom(text=response, valid=True)]
