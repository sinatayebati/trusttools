# Create this file: trusttools/models/conformal_validator.py

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
                
                # Conditionally include top_logprobs based on parameter    
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
        Uses sentence-based chunking rather than LLM-based atom extraction for reliable mapping.
        
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
        
        # Step 3: Chunk the text using multiple strategies
        import re
        
        # Try several chunking approaches
        chunks_from_all_methods = []
        
        # Approach 1: Structure-based chunking (headers, lists, paragraphs)
        try:
            print("Using structure-based chunking")
            # This pattern handles markdown headers and list items
            structural_pattern = r'(\n#+\s+.*?(?=\n#+|\Z))|\n\d+\.\s+.*?(?=\n\d+\.|\Z)|\n\*\s+.*?(?=\n\*|\Z)'
            structural_chunks = re.findall(structural_pattern, '\n' + response, re.DOTALL)
            structural_chunks = [chunk.strip() for chunk in structural_chunks if chunk.strip()]
            
            # If we got some, add them to our chunks
            if structural_chunks:
                print(f"Found {len(structural_chunks)} structural chunks (headers, lists)")
                chunks_from_all_methods.extend(structural_chunks)
        except Exception as e:
            print(f"Structure-based chunking failed: {e}")
        
        # Approach 2: Paragraph-based chunking
        try:
            print("Using paragraph-based chunking")
            paragraphs = re.split(r'\n\s*\n', response)
            paragraph_chunks = [p.strip() for p in paragraphs if p.strip()]
            
            if paragraph_chunks:
                print(f"Found {len(paragraph_chunks)} paragraphs")
                chunks_from_all_methods.extend(paragraph_chunks)
        except Exception as e:
            print(f"Paragraph-based chunking failed: {e}")
        
        # Approach 3: Sentence-based chunking
        try:
            print("Using sentence-based chunking")
            # This pattern handles common sentence endings (.!?)
            sentence_pattern = r'(?<=[.!?])\s+|(?<=[.!?])$'
            
            # Apply sentence splitting to the paragraphs
            sentence_chunks = []
            for paragraph in paragraph_chunks:
                sentences = re.split(sentence_pattern, paragraph)
                sentence_chunks.extend([s.strip() for s in sentences if s.strip()])
            
            if sentence_chunks:
                print(f"Found {len(sentence_chunks)} sentences")
                chunks_from_all_methods.extend(sentence_chunks)
        except Exception as e:
            print(f"Sentence-based chunking failed: {e}")
        
        # Deduplicate and filter chunks
        unique_chunks = []
        seen = set()
        for chunk in chunks_from_all_methods:
            if chunk not in seen and len(chunk.strip()) > 0:
                seen.add(chunk)
                unique_chunks.append(chunk)
        
        # If we still don't have any chunks, use the whole response
        if not unique_chunks:
            print("All chunking methods failed, using whole response")
            unique_chunks = [response]
        
        print(f"Final chunked text: {len(unique_chunks)} segments")
        
        # Step 4: Map chunks to token positions and calculate scores
        final_atoms = []
        
        for chunk in unique_chunks:
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
                        valid=True  # Default to valid, will be updated during trimming
                    ))
                    print(f"Mapped chunk: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}', "
                         f"Token count: {len(chunk_tokens)}, Score: {score:.4f}")
                else:
                    final_atoms.append(Atom(text=chunk, score=None, logprobs=None, valid=True))
                    print(f"No tokens mapped for chunk: '{chunk[:50]}...'")
            else:
                # Chunk not found directly, use fuzzy matching
                # Clean both texts for better matching
                clean_chunk = re.sub(r'\s+', ' ', chunk).strip()
                clean_reconstructed = re.sub(r'\s+', ' ', reconstructed_text).strip()
                
                chunk_start = clean_reconstructed.find(clean_chunk)
                if chunk_start != -1:
                    chunk_end = chunk_start + len(clean_chunk)
                    
                    # Get overlapping tokens
                    chunk_tokens = [t for t in token_positions 
                                   if (t['start'] < chunk_end and t['end'] > chunk_start)]
                    
                    if chunk_tokens:
                        logprobs = [t['logprob'] for t in chunk_tokens]
                        score = np.mean(logprobs)
                        final_atoms.append(Atom(
                            text=chunk, 
                            score=score,
                            logprobs=logprobs,
                            valid=True
                        ))
                        print(f"Mapped chunk (fuzzy): '{chunk[:50]}...' Score: {score:.4f}")
                    else:
                        final_atoms.append(Atom(text=chunk, score=None, logprobs=None, valid=True))
                else:
                    # Last resort: approximate mapping based on position in text
                    print(f"Could not directly map: '{chunk[:50]}...'")
                    final_atoms.append(Atom(text=chunk, score=None, logprobs=None, valid=True))
        
        if not final_atoms:
            # Fallback if chunking fails completely
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
