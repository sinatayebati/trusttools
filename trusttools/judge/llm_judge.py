#!/usr/bin/env python3

import os
import json
import argparse
import sys
from typing import List, Dict, Any, Optional

# Add parent directory to path to import the OpenAI engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from trusttools.engine.openai import ChatOpenAI

class LLMJudge:
    def __init__(self, model: str = "gpt-4.1-mini", domain: str = "general"):
        """
        Initialize the LLM judge.
        
        Args:
            model: The LLM model to use for judging
            domain: The domain of evaluation (medical, math, etc.)
        """
        self.model = model
        self.domain = domain
        self.llm = ChatOpenAI(model_string=model)
        
    def create_judge_prompt(self, query: str, texts: List[Dict]) -> str:
        """
        Create a prompt for the LLM to judge whether the text is valid.
        
        Args:
            query: The original query
            texts: List of dictionaries containing text to judge
            
        Returns:
            The formatted prompt
        """
        prompt = f"""You are an expert judge in the {self.domain} domain. 
Your task is to evaluate if each of the following text segments is a valid response to the query.

QUERY: "{query}"

For each text segment, evaluate if it is:
1. Factually correct and scientifically sound based on established knowledge in the {self.domain} field
2. Directly relevant to the query
3. Necessary to answer the query

TEXT SEGMENTS:
"""

        for i, item in enumerate(texts, 1):
            prompt += f"{i}. {item['text']}\n\n"
            
        prompt += """
For each text segment, respond in the following JSON format:
```
{
  "judgments": [
    {"segment": 1, "valid": true/false, "reason": "brief explanation"},
    {"segment": 2, "valid": true/false, "reason": "brief explanation"},
    ...
  ]
}
```

Only include true or false for the 'valid' field, without quotes. Provide brief, concise explanations.
"""
        return prompt
    
    def judge_text_segments(self, query: str, texts: List[Dict]) -> List[Dict]:
        """
        Judge whether each text segment is valid.
        
        Args:
            query: The original query
            texts: List of dictionaries containing text to judge
            
        Returns:
            Updated list of dictionaries with judgment results
        """
        prompt = self.create_judge_prompt(query, texts)
        
        # Get response from LLM
        response = self.llm.generate(prompt)
        
        # Extract JSON from response
        response_text = response.text if hasattr(response, 'text') else response
        try:
            # Find JSON block in response
            json_start = response_text.find('```')
            if json_start != -1:
                json_start = response_text.find('{', json_start)
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
            else:
                json_str = response_text
                
            result = json.loads(json_str)
            
            # Update original texts with judgments
            for judgment in result.get('judgments', []):
                segment_idx = judgment.get('segment', 0) - 1
                if 0 <= segment_idx < len(texts):
                    texts[segment_idx]['valid'] = judgment.get('valid', False)
                    # Optionally store the reason if needed
                    texts[segment_idx]['judge_reason'] = judgment.get('reason', '')
            
            return texts
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response as JSON: {e}")
            print(f"Response: {response_text}")
            # If parsing fails, keep the original valid values
            return texts


def process_file(file_path: str, model: str, domain: str) -> None:
    """
    Process a single JSON file.
    
    Args:
        file_path: Path to the JSON file
        model: LLM model to use
        domain: Domain of evaluation
    """
    try:
        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract query and validation results
        query = data.get('query', '')
        validation_results = data.get('validation_results', {})
        direct_output_atoms = validation_results.get('direct_output_atoms', [])
        
        if not query or not direct_output_atoms:
            print(f"Missing required data in {file_path}")
            return
            
        # Initialize the LLM judge
        judge = LLMJudge(model=model, domain=domain)
        
        # Judge text segments
        updated_atoms = judge.judge_text_segments(query, direct_output_atoms)
        
        # Update the original data
        data['validation_results']['direct_output_atoms'] = updated_atoms
        
        # Save updated JSON
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Successfully processed {file_path}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='LLM Judge for evaluating text segments')
    parser.add_argument('files', nargs='+', help='JSON files to process')
    parser.add_argument('--model', type=str, default='gpt-4.1-mini', help='LLM model to use')
    parser.add_argument('--domain', type=str, default='general', help='Domain of evaluation (medical, math, etc.)')
    
    args = parser.parse_args()
    
    for file_path in args.files:
        process_file(file_path, args.model, args.domain)


if __name__ == '__main__':
    main() 