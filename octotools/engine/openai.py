try:
    from openai import OpenAI
except ImportError:
    raise ImportError("If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, and add 'OPENAI_API_KEY' to your environment variables.")

import os
import json
import base64
import platformdirs
import numpy as np
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union, Dict, Any, Optional

from .base import EngineLM, CachedEngine

import openai

from dotenv import load_dotenv
load_dotenv()


from pydantic import BaseModel

class DefaultFormat(BaseModel):
    response: str

# Define global constants for structured models
OPENAI_STRUCTURED_MODELS = ['gpt-4o', 'gpt-4o-2024-08-06','gpt-4o-mini',  'gpt-4o-mini-2024-07-18']

# Add constant for local models
LOCAL_MODELS = ['meta-llama/Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.2-11B-Vision-Instruct']

# Default models for each environment
DEFAULT_API_MODEL = "gpt-4o-mini"
DEFAULT_LOCAL_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


class ChatOpenAI(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string=None,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False,
        enable_cache: bool=True,
        use_local_model: bool=False,
        local_model_endpoint: str=None,
        capture_logits: bool=False,
        logits_dir: str=None,
        **kwargs):
        """
        :param model_string: The model to use
        :param system_prompt: The system prompt to use
        :param is_multimodal: Whether the model can process images
        :param enable_cache: Whether to cache responses
        :param use_local_model: Whether to use a locally hosted model instead of OpenAI
        :param local_model_endpoint: The endpoint for the local model, defaults to env variable or http://localhost:8000/v1
        :param capture_logits: Whether to capture and save logits from the local model
        :param logits_dir: Directory to save captured logits
        """
        if model_string is None:
            if use_local_model:
                model_string = DEFAULT_LOCAL_MODEL
            else:
                model_string = DEFAULT_API_MODEL
                
        # Validate model selection based on local vs API
        if use_local_model and model_string not in LOCAL_MODELS:
            print(f"Warning: {model_string} not recognized as a local model. Available local models: {', '.join(LOCAL_MODELS)}")
            print(f"Falling back to default local model: {DEFAULT_LOCAL_MODEL}")
            model_string = DEFAULT_LOCAL_MODEL
            
        if enable_cache:
            root = platformdirs.user_cache_dir("octotools")
            cache_path = os.path.join(root, f"cache_openai_{model_string}.db")
            
            self.image_cache_dir = os.path.join(root, "image_cache")
            os.makedirs(self.image_cache_dir, exist_ok=True)

            super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        self.model_string = model_string
        self.is_multimodal = is_multimodal
        self.enable_cache = enable_cache
        self.use_local_model = use_local_model
        self.capture_logits = capture_logits
        
        # Set up logits directory if needed
        if capture_logits:
            if logits_dir is None:
                root = platformdirs.user_cache_dir("octotools")
                self.logits_dir = os.path.join(root, "logits")
            else:
                self.logits_dir = logits_dir
            os.makedirs(self.logits_dir, exist_ok=True)
        
        # Configure client based on local vs OpenAI
        if use_local_model:
            if local_model_endpoint is None:
                local_model_endpoint = os.getenv("LOCAL_LLM_ENDPOINT", "http://localhost:8000/v1")
            
            local_model_endpoint = local_model_endpoint.rstrip('/')
            if local_model_endpoint.endswith('/v1'):
                base_endpoint = local_model_endpoint[:-3]
            else:
                base_endpoint = local_model_endpoint
            
            self.base_endpoint_str = base_endpoint
            
            # Verify server is running
            try:
                health_check = requests.get(f"{base_endpoint}/v1/models")
                if health_check.status_code == 200:
                    print("✓ Successfully connected to local LLM server")
                    models = health_check.json()
                    available_models = [model["id"] for model in models.get("data", [])]
                    print(f"Available models: {available_models}")
                    
                    # Ensure we're using a model name that exists on the server
                    if self.model_string not in available_models:
                        if len(available_models) > 0:
                            self.model_string = available_models[0]
                            print(f"Switching to available model: {self.model_string}")
                else:
                    print(f"⚠️  Warning: Server health check failed with status {health_check.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Warning: Could not connect to server: {str(e)}")
            
            # Create client with the correct base URL
            self.client = OpenAI(
                base_url=f"{base_endpoint}/v1",
                api_key="dummy-key"
            )
        else:
            if os.getenv("OPENAI_API_KEY") is None:
                raise ValueError("Please set the OPENAI_API_KEY environment variable if you'd like to use OpenAI models.")
            
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            print(f"Using OpenAI API model: {self.model_string}")
        
        if enable_cache:
            print(f"!! Cache enabled for model: {self.model_string}")
        else:
            print(f"!! Cache disabled for model: {self.model_string}")

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        try:
            attempt_number = self.generate.retry.statistics.get('attempt_number', 0) + 1
            if attempt_number > 1:
                print(f"Attempt {attempt_number} of 5")

            if isinstance(content, str):
                return self._generate_text(content, system_prompt=system_prompt, **kwargs)
            
            elif isinstance(content, list):
                if (not self.is_multimodal) and (not self.use_local_model):
                    raise NotImplementedError("Multimodal generation is only supported for GPT-4 models or local models with multimodal capability.")
                
                return self._generate_multimodal(content, system_prompt=system_prompt, **kwargs)

        except openai.LengthFinishReasonError as e:
            print(f"Token limit exceeded: {str(e)}")
            print(f"Tokens used - Completion: {e.completion.usage.completion_tokens}, Prompt: {e.completion.usage.prompt_tokens}, Total: {e.completion.usage.total_tokens}")
            return {
                "error": "token_limit_exceeded",
                "message": str(e),
                "details": {
                    "completion_tokens": e.completion.usage.completion_tokens,
                    "prompt_tokens": e.completion.usage.prompt_tokens,
                    "total_tokens": e.completion.usage.total_tokens
                }
            }
        except openai.RateLimitError as e:
            print(f"Rate limit error encountered: {str(e)}")
            return {
                "error": "rate_limit",
                "message": str(e),
                "details": getattr(e, 'args', None)
            }
        except Exception as e:
            print(f"Error in generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            return {
                "error": type(e).__name__,
                "message": str(e),
                "details": getattr(e, 'args', None)
            }
        
    def _generate_text(
        self, prompt, system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, response_format=None
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        response_text = None

        if self.enable_cache:
            cache_key = sys_prompt_arg + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user", "content": prompt},
        ]

        try:
            if self.use_local_model:
                if self.capture_logits:
                    # Use logits capture method
                    response_text = self._call_local_model_with_logits(
                        base_url=self.base_endpoint_str,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )
                else:
                    # Use standard OpenAI client
                    response = self.client.chat.completions.create(
                        model=self.model_string,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stream=False
                    )
                    response_text = response.choices[0].message.content
                    
            elif self.model_string in ['o1', 'o1-mini']:
                response = self.client.beta.chat.completions.parse(
                    model=self.model_string,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens
                )
                response_text = "Token limit exceeded" if response.choices[0].finishreason == "length" else response.choices[0].message.parsed
                
            elif self.model_string in OPENAI_STRUCTURED_MODELS and response_format is not None:
                response = self.client.beta.chat.completions.parse(
                    model=self.model_string,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    response_format=response_format
                )
                response_text = response.choices[0].message.parsed
                
            else:
                response = self.client.chat.completions.create(
                    model=self.model_string,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                response_text = response.choices[0].message.content

            if self.enable_cache and response_text:
                self._save_cache(cache_key, response_text)
            
            return response_text

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return error_msg

    def _call_local_model_with_logits(self, base_url, messages, temperature=0, max_tokens=4000, top_p=0.99):
        request_body = {
            "model": self.model_string,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False
        }
        
        if self.capture_logits:
            request_body.update({
                "logprobs": True,
                "top_logprobs": 5
            })
        
        base_url_str = str(self.client.base_url).rstrip('/')
        if '/v1' in base_url_str:
            endpoint_url = f"{base_url_str}/chat/completions"
        else:
            endpoint_url = f"{base_url_str}/v1/chat/completions"
        
        print(f"\nDebug information:")
        print(f"Base URL: {base_url_str}")
        print(f"Making request to: {endpoint_url}")
        print(f"Using model: {self.model_string}")
        print(f"Capturing logits: {self.capture_logits}")
        
        try:
            response = requests.post(
                endpoint_url,
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code != 200:
                error_msg = (
                    f"Error: Request failed with status {response.status_code}\n"
                    f"Response: {response.text}\n"
                    f"Endpoint URL: {endpoint_url}\n"
                    f"Request body: {json.dumps(request_body, indent=2)}"
                )
                print(error_msg)
                return error_msg
                
            result = response.json()
            
            # Save logits if they were captured
            if self.capture_logits and hasattr(self, 'logits_dir'):
                import time
                import uuid
                
                timestamp = int(time.time())
                unique_id = str(uuid.uuid4())[:8]
                logits_filename = f"logits_{timestamp}_{unique_id}.json"
                logits_path = os.path.join(self.logits_dir, logits_filename)
                
                has_logprobs = 'logprobs' in result or any('logprobs' in choice for choice in result.get('choices', []))
                
                logits_data = {
                    "timestamp": timestamp,
                    "model": self.model_string,
                    "request": {
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": top_p,
                        "logprobs_requested": self.capture_logits
                    },
                    "response": result,
                    "has_logprobs": has_logprobs
                }
                
                with open(logits_path, 'w') as f:
                    json.dump(logits_data, f, indent=2)
                
                print(f"\nSaved response data to: {logits_path}")
                if not has_logprobs:
                    print("⚠️  Warning: No logprobs found in response despite being requested")
            
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            error_msg = (
                f"Error calling local model API: {str(e)}\n"
                f"Model: {self.model_string}\n"
                f"Endpoint URL: {endpoint_url}"
            )
            print(error_msg)
            return error_msg

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_multimodal(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, response_format=None
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        if self.enable_cache:
            cache_key = sys_prompt_arg + json.dumps(formatted_content)
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        if self.use_local_model:
            # For local multimodal models
            if self.is_multimodal:
                if self.capture_logits:
                    response_text = self._call_local_model_with_logits(
                        base_url=self.base_endpoint_str,
                        messages=[
                            {"role": "system", "content": sys_prompt_arg},
                            {"role": "user", "content": formatted_content},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_string,
                        messages=[
                            {"role": "system", "content": sys_prompt_arg},
                            {"role": "user", "content": formatted_content},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                    response_text = response.choices[0].message.content
            else:
                # If the model doesn't support images, extract text only
                text_only = [item["text"] for item in formatted_content if item["type"] == "text"]
                joined_text = " ".join(text_only)
                response_text = self._generate_text(
                    joined_text,
                    system_prompt=sys_prompt_arg,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                
        elif self.model_string in ['o1', 'o1-mini']: # only supports base response currently
            print(f'Max tokens: {max_tokens}')
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "user", "content": formatted_content},
                ],
                max_completion_tokens=max_tokens
            )
            if response.choices[0].finish_reason == "length":
                response_text = "Token limit exceeded"
            else:
                response_text = response.choices[0].message.content
                
        elif self.model_string in OPENAI_STRUCTURED_MODELS and response_format is not None:
            response = self.client.beta.chat.completions.parse(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": formatted_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                response_format=response_format
            )
            response_text = response.choices[0].message.parsed
            
        else:
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": formatted_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            response_text = response.choices[0].message.content

        if self.enable_cache:
            self._save_cache(cache_key, response_text)
        return response_text

