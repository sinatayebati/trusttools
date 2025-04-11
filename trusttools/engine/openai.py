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
import hashlib

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

# Add a simple structure for the return value when capturing logits
class GenerationResult(BaseModel):
    text: Optional[str] = None
    logprob_content: Optional[List[Dict]] = None
    error: Optional[str] = None
    error_details: Optional[Dict] = None


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
            root = platformdirs.user_cache_dir("trusttools")
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
                root = platformdirs.user_cache_dir("trusttools")
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
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs) -> Union[GenerationResult, BaseModel]:
        # Explicitly extract capture_logits from kwargs, falling back to instance default
        capture_logits_call = kwargs.pop('capture_logits', self.capture_logits)
        
        try:
            attempt_number = self.generate.retry.statistics.get('attempt_number', 0) + 1
            if attempt_number > 1:
                print(f"Attempt {attempt_number} of 5")

            if isinstance(content, str):
                # Pass capture_logits explicitly
                return self._generate_text(content, system_prompt=system_prompt, capture_logits=capture_logits_call, **kwargs)
            
            elif isinstance(content, list):
                # Allow multimodal for local models if model name suggests vision capability
                can_do_multimodal = self.is_multimodal or (self.use_local_model and any(vm in self.model_string for vm in ['Vision', 'vision']))
                if not can_do_multimodal:
                     raise NotImplementedError("Multimodal generation is only supported for specific models or local models with multimodal capability.")
                
                # Pass capture_logits explicitly
                return self._generate_multimodal(content, system_prompt=system_prompt, capture_logits=capture_logits_call, **kwargs)

        except openai.LengthFinishReasonError as e:
            print(f"Token limit exceeded: {str(e)}")
            details = {}
            if hasattr(e, 'completion') and hasattr(e.completion, 'usage'):
                 details = {
                    "completion_tokens": e.completion.usage.completion_tokens,
                    "prompt_tokens": e.completion.usage.prompt_tokens,
                    "total_tokens": e.completion.usage.total_tokens
                 }
            return GenerationResult(
                error="token_limit_exceeded",
                error_details=details
            )
        except openai.RateLimitError as e:
            print(f"Rate limit error encountered: {str(e)}")
            return GenerationResult(error="rate_limit", error_details={"message": str(e)})
        except Exception as e:
            print(f"Error in generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            return GenerationResult(error=type(e).__name__, error_details={"message": str(e), "args": e.args})
        
    def _generate_text(
        self, prompt, system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, response_format=None, capture_logits: bool = False
    ) -> Union[GenerationResult, BaseModel]:
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        response_text = None
        logprob_content = None
        cache_key = None

        # --- Structured Response Handling ---
        if response_format is not None:
            # Structured responses generally don't support logit capture via 'parse'
            # and caching needs careful consideration (key based on format + prompt)
            if self.capture_logits:
                print(f"Warning: Logit capture requested but response_format is {response_format}. Logits may not be captured for structured responses.")
            
            cache_key_structured = f"structured_{response_format.__name__}_{sys_prompt_arg}_{prompt}"
            if self.enable_cache:
                cache_or_none = self._check_cache(cache_key_structured)
                if cache_or_none is not None and isinstance(cache_or_none, response_format):
                    print(f"Cache hit for structured response: {response_format.__name__}")
                    return cache_or_none # Return the cached Pydantic object

            messages = [
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ]

            try:
                # Use client.beta.chat.completions.parse if model supports it and not local
                # NOTE: Adjust this logic based on actual API capabilities and model support
                if not self.use_local_model and self.model_string in OPENAI_STRUCTURED_MODELS:
                     print(f"Using client.beta.chat.completions.parse for {response_format.__name__}")
                     response_parsed = self.client.beta.chat.completions.parse(
                         model=self.model_string,
                         messages=messages,
                         temperature=temperature,
                         max_tokens=max_tokens,
                         top_p=top_p,
                         response_format=response_format
                     )
                     parsed_object = response_parsed.choices[0].message.parsed
                else:
                    # Fallback: Standard completion + manual JSON parsing
                    print(f"Using standard completion + manual parsing for {response_format.__name__}")
                    completion_args_structured = {
                        "model": self.model_string, "messages": messages,
                        "temperature": temperature, "max_tokens": max_tokens, "top_p": top_p,
                        # Ask model to output JSON matching the Pydantic schema
                        "response_format": {"type": "json_object"} 
                    }
                    # Add schema to prompt for better results with manual parsing fallback
                    schema_json = json.dumps(response_format.model_json_schema(), indent=2)
                    messages[-1]["content"] += f"\n\nPlease format your response as a JSON object matching this Pydantic schema:\n```json\n{schema_json}\n```"

                    completion_args_structured["messages"] = messages # Update messages

                    response = self.client.chat.completions.create(**completion_args_structured)
                    raw_json_text = response.choices[0].message.content
                    try:
                        parsed_data = json.loads(raw_json_text)
                        parsed_object = response_format(**parsed_data)
                    except (json.JSONDecodeError, TypeError, ValueError) as parse_error:
                        print(f"Error parsing structured response manually: {parse_error}")
                        print(f"Raw response: {raw_json_text}")
                        # Decide how to handle parse errors - raise, return None, return error object?
                        # For now, let's return an error within GenerationResult structure for consistency
                        return GenerationResult(error="parsing_error", error_details={"message": str(parse_error), "raw_text": raw_json_text})


                # Cache the successfully parsed Pydantic object
                if self.enable_cache:
                     self._save_cache(cache_key_structured, parsed_object)

                return parsed_object # Return the Pydantic object directly

            except Exception as e:
                error_msg = f"Error generating structured response ({response_format.__name__}): {str(e)}"
                print(error_msg)
                # Return None to indicate failure to get structured object
                return None

        # --- Plain Text Response Handling (response_format is None) ---
        else:
            if self.enable_cache:
                # Include capture_logits status in cache key for text responses
                cache_key = f"text_{sys_prompt_arg}_{prompt}_logits:{self.capture_logits}"
                cache_or_none = self._check_cache(cache_key)
                if cache_or_none is not None:
                    # Ensure cached data is in the new GenerationResult format
                    if isinstance(cache_or_none, dict) and "text" in cache_or_none:
                        return GenerationResult(**cache_or_none)
                    elif isinstance(cache_or_none, str) and not self.capture_logits:
                        # Handle old cache format if logits weren't requested
                        return GenerationResult(text=cache_or_none)
                    else:
                        # Invalid cache format, proceed to generate
                        print("Invalid cache format found, regenerating...")
                        pass

            messages = [
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ]

            completion_args = {
                "model": self.model_string,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }

            # Add logprobs argument if capturing logits
            if capture_logits:
                completion_args["logprobs"] = True
                completion_args["top_logprobs"] = 5

            try:
                response = None # Initialize response

                # --- Handle different model types/calling conventions ---
                if self.use_local_model:
                    if self.capture_logits:
                        local_response = self._call_local_model_and_get_response(
                            messages=messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p
                        )
                        if local_response and "choices" in local_response and local_response["choices"]:
                            response_text = local_response["choices"][0].get("message", {}).get("content")
                            
                            # Enhanced logprob extraction for local models - same as _generate_text
                            if self.capture_logits:
                                choice = local_response["choices"][0]
                                if "logprobs" in choice:
                                    print("Extracting logprobs from local model response")
                                    if isinstance(choice["logprobs"], dict) and "content" in choice["logprobs"]:
                                        # Standard content format
                                        logprob_content = choice["logprobs"]["content"]
                                        print(f"Using standard content format with {len(logprob_content)} tokens")
                                    elif isinstance(choice["logprobs"], dict):
                                        # Try to convert from tokens/token_logprobs format
                                        if "tokens" in choice["logprobs"] and "token_logprobs" in choice["logprobs"]:
                                            tokens = choice["logprobs"]["tokens"]
                                            token_logprobs = choice["logprobs"]["token_logprobs"]
                                            if len(tokens) == len(token_logprobs):
                                                print(f"Converting {len(tokens)} tokens to content format")
                                                logprob_content = [
                                                    {"token": t, "logprob": lp} 
                                                    for t, lp in zip(tokens, token_logprobs)
                                                ]
                                            else:
                                                print(f"Tokens and logprobs length mismatch: {len(tokens)} vs {len(token_logprobs)}")
                                                logprob_content = None
                                        else:
                                            print(f"Found logprobs but missing tokens/token_logprobs. Keys: {list(choice['logprobs'].keys())}")
                                            logprob_content = None
                                    else:
                                        # Unusual format, try to use directly
                                        print(f"Non-dict logprobs format: {type(choice['logprobs'])}")
                                        logprob_content = choice["logprobs"]
                                else:
                                    print("No logprobs found in local model response despite being requested")
                                    logprob_content = None
                        else:
                            error_msg = local_response.get("error", "Unknown error from local model call") if isinstance(local_response, dict) else "Invalid response from local model call"
                            return GenerationResult(error="local_model_error", error_details={"message": error_msg})
                    else:
                        response = self.client.chat.completions.create(**completion_args)
                        response_text = response.choices[0].message.content

                # O1 models and Parse - Deprecated or logic needs update based on current OpenAI lib
                elif self.model_string in ['o1', 'o1-mini']:
                     if self.capture_logits:
                         print("Warning: Logit capture may not be fully supported for o1 models. Calling standard API.")
                         response = self.client.chat.completions.create(**completion_args) # Use args with logprobs=True
                         response_text = response.choices[0].message.content
                         if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                             logprob_content = response.choices[0].logprobs.content
                     else:
                         # Use standard completion for o1 as parse is less reliable/supported
                         response = self.client.chat.completions.create(**completion_args)
                         response_text = response.choices[0].message.content


                else: # Standard OpenAI API call for text
                    response = self.client.chat.completions.create(**completion_args)
                    response_text = response.choices[0].message.content
                    
                    # Enhanced logprob extraction logic with detailed debugging
                    if self.capture_logits:
                        print(f"\nCapturing logprobs for model: {self.model_string}")
                        
                        # Check for different potential response structures
                        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                            # Standard logprobs structure
                            if hasattr(response.choices[0].logprobs, 'content'):
                                raw_logprob_content = response.choices[0].logprobs.content
                                # Convert ChatCompletionTokenLogprob objects to dictionaries if needed
                                try:
                                    logprob_content = []
                                    for item in raw_logprob_content:
                                        if hasattr(item, 'to_dict'):
                                            # If it has a to_dict method (like OpenAI's ChatCompletionTokenLogprob), use it
                                            logprob_content.append(item.to_dict())
                                        elif hasattr(item, 'model_dump'):
                                            # For Pydantic models
                                            logprob_content.append(item.model_dump())
                                        elif isinstance(item, dict):
                                            # It's already a dict
                                            logprob_content.append(item)
                                        else:
                                            # Try manual conversion - adjust based on the object's structure
                                            logprob_dict = {
                                                'token': getattr(item, 'token', ''),
                                                'logprob': getattr(item, 'logprob', -5.0),
                                                # Add other fields if present
                                                'bytes': getattr(item, 'bytes', None) if hasattr(item, 'bytes') else None,
                                                'top_logprobs': getattr(item, 'top_logprobs', None) if hasattr(item, 'top_logprobs') else None
                                            }
                                            logprob_content.append(logprob_dict)
                                    print(f"Converted {len(logprob_content)} logprob items to dict format")
                                except Exception as conv_err:
                                    print(f"Error converting logprobs to dict format: {conv_err}")
                                    # Fallback: set logprobs to None rather than failing
                                    logprob_content = None
                                print(f"Found logprobs in standard format")
                            elif isinstance(response.choices[0].logprobs, dict) and 'content' in response.choices[0].logprobs:
                                raw_logprob_content = response.choices[0].logprobs['content']
                                # Apply same conversion as above
                                try:
                                    logprob_content = []
                                    for item in raw_logprob_content:
                                        if hasattr(item, 'to_dict'):
                                            logprob_content.append(item.to_dict())
                                        elif hasattr(item, 'model_dump'):
                                            logprob_content.append(item.model_dump())
                                        elif isinstance(item, dict):
                                            logprob_content.append(item)
                                        else:
                                            logprob_dict = {
                                                'token': getattr(item, 'token', ''),
                                                'logprob': getattr(item, 'logprob', -5.0),
                                                'bytes': getattr(item, 'bytes', None) if hasattr(item, 'bytes') else None,
                                                'top_logprobs': getattr(item, 'top_logprobs', None) if hasattr(item, 'top_logprobs') else None
                                            }
                                            logprob_content.append(logprob_dict)
                                    print(f"Converted {len(logprob_content)} dict-content logprob items to dict format")
                                except Exception as conv_err:
                                    print(f"Error converting dict-content logprobs: {conv_err}")
                                    logprob_content = None
                                print(f"Found logprobs in dictionary format")
                            else:
                                # Try to get raw logprobs if available
                                logprob_content = response.choices[0].logprobs
                                print(f"Using raw logprobs object: {type(logprob_content)}")
                        elif hasattr(response, 'logprobs') and response.logprobs:
                            # Alternative position in response
                            logprob_content = response.logprobs
                            print(f"Found logprobs at response root level")
                        else:
                            print(f"Logprobs requested but not found in response. Response structure: {dir(response.choices[0])}")
                            logprob_content = None
                        
                        # --- Logits Saving (Optional Debugging) ---
                        if hasattr(self, 'logits_dir'):
                            import time, uuid, json
                            timestamp = int(time.time())
                            unique_id = str(uuid.uuid4())[:8]
                            logits_filename = f"logits_api_{timestamp}_{unique_id}.json"
                            logits_path = os.path.join(self.logits_dir, logits_filename)
                            # Ensure completion_args doesn't contain non-serializable items if needed
                            serializable_args = {k: v for k, v in completion_args.items() if k != 'response_format'}
                            
                            # Save full response for debugging
                            try:
                                response_dump = response.model_dump()
                            except Exception as dump_err:
                                print(f"Error dumping response model: {dump_err}")
                                # Try alternative serialization
                                try:
                                    import json
                                    response_dump = json.loads(json.dumps(response, default=lambda o: f"<non-serializable: {type(o)}>"))
                                except Exception as json_err:
                                    print(f"Error serializing response: {json_err}")
                                    response_dump = {"error": "Could not serialize response"}
                            
                            logits_data = {
                                "timestamp": timestamp, "model": self.model_string,
                                "request": {"messages": messages, **serializable_args},
                                "response": response_dump,
                                "logprob_content_found": logprob_content is not None,
                                "response_keys": dir(response),
                                "first_choice_keys": dir(response.choices[0]) if hasattr(response, 'choices') and len(response.choices) > 0 else []
                            }
                            try:
                                with open(logits_path, 'w') as f:
                                    json.dump(logits_data, f, indent=2, default=lambda o: f"<non-serializable: {type(o)}>")
                                print(f"\nSaved API response data to: {logits_path}")
                            except Exception as save_e:
                                print(f"Error saving logits to file: {save_e}")
                        # --- End Logits Saving ---


                # --- Prepare result ---
                result = GenerationResult(text=response_text, logprob_content=logprob_content)

                if self.enable_cache and result.text is not None:
                    # Cache the result dictionary
                    self._save_cache(cache_key, result.dict(exclude_none=True))

                return result # Return GenerationResult for plain text

            except Exception as e:
                error_msg = f"Error generating text response: {str(e)}"
                print(error_msg)
                # Return error within the standard structure
                return GenerationResult(error="generation_error", error_details={"message": error_msg})

    def _call_local_model_and_get_response(self, messages, temperature=0, max_tokens=4000, top_p=0.99) -> Optional[Dict]:
        """ Calls local model via requests and returns the full JSON response. """
        request_body = {
            "model": self.model_string,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
            "logprobs": True, # Explicitly request logprobs
            "top_logprobs": 5
        }

        base_url_str = str(self.client.base_url).rstrip('/')
        endpoint_url = f"{base_url_str}/chat/completions" # Standard v1 path

        print(f"\n[Local Call] Request to: {endpoint_url}")
        print(f"[Local Call] Model: {self.model_string}")
        print(f"[Local Call] Requesting logprobs: {request_body['logprobs']}")

        try:
            response = requests.post(
                endpoint_url,
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=120 # Increased timeout for potentially slower local models
            )

            print(f"[Local Call] Response status: {response.status_code}")

            if response.status_code != 200:
                error_msg = (
                    f"Error: Request failed with status {response.status_code}\n"
                    f"Response: {response.text}\n"
                    f"Endpoint URL: {endpoint_url}"
                )
                print(error_msg)
                return {"error": error_msg} # Return error dict

            result = response.json()
            
            # Add extra logprob extraction debugging
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                print(f"[Local Call] Result contains choices. First choice keys: {list(choice.keys())}")
                
                if 'logprobs' in choice:
                    print(f"[Local Call] Logprobs found in response. Keys: {list(choice['logprobs'].keys()) if isinstance(choice['logprobs'], dict) else 'not a dict'}")
                    
                    # Try to extract and standardize logprobs format
                    if isinstance(choice['logprobs'], dict) and 'content' in choice['logprobs']:
                        print(f"[Local Call] Found standard content logprobs with {len(choice['logprobs']['content'])} tokens")
                    elif isinstance(choice['logprobs'], dict):
                        # Try to standardize by adding 'content' key if not present
                        for potential_key in ['token_logprobs', 'tokens', 'top_logprobs']:
                            if potential_key in choice['logprobs']:
                                print(f"[Local Call] Found alternative logprobs format with '{potential_key}'")
                                # Create a synthetic 'content' field if needed
                                if 'tokens' in choice['logprobs'] and 'token_logprobs' in choice['logprobs']:
                                    tokens = choice['logprobs']['tokens']
                                    logprobs = choice['logprobs']['token_logprobs']
                                    if len(tokens) == len(logprobs):
                                        print(f"[Local Call] Creating synthetic 'content' field from {len(tokens)} tokens")
                                        choice['logprobs']['content'] = [
                                            {'token': t, 'logprob': lp} for t, lp in zip(tokens, logprobs)
                                        ]
                                break
                else:
                    print(f"[Local Call] No logprobs found in response despite being requested")

            # --- Logits Saving (Optional Debugging) ---
            if hasattr(self, 'logits_dir'):
                 import time, uuid
                 timestamp = int(time.time())
                 unique_id = str(uuid.uuid4())[:8]
                 logits_filename = f"logits_local_{timestamp}_{unique_id}.json"
                 logits_path = os.path.join(self.logits_dir, logits_filename)
                 logits_data = {
                     "timestamp": timestamp, "model": self.model_string,
                     "request": request_body,
                     "response": result,
                     "response_structure": {
                         "has_choices": 'choices' in result,
                         "first_choice_keys": list(result['choices'][0].keys()) if 'choices' in result and result['choices'] else [],
                         "has_logprobs": 'choices' in result and result['choices'] and 'logprobs' in result['choices'][0],
                         "logprobs_keys": list(result['choices'][0]['logprobs'].keys()) if 'choices' in result and result['choices'] and 'logprobs' in result['choices'][0] and isinstance(result['choices'][0]['logprobs'], dict) else []
                     }
                 }
                 try:
                      with open(logits_path, 'w') as f:
                          json.dump(logits_data, f, indent=2)
                      print(f"\nSaved local response data to: {logits_path}")
                 except Exception as save_e:
                      print(f"Error saving local logits to file: {save_e}")
            # --- End Logits Saving ---

            return result # Return the full JSON response

        except Exception as e:
            error_msg = (
                f"Error calling local model API: {str(e)}\n"
                f"Model: {self.model_string}\n"
                f"Endpoint URL: {endpoint_url}"
            )
            print(error_msg)
            return {"error": error_msg} # Return error dict

    def __call__(self, prompt, **kwargs) -> Union[GenerationResult, BaseModel]:
        # The generate method now handles the return type logic
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
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, response_format=None, capture_logits: bool = False
    ) -> Union[GenerationResult, BaseModel]:
        # NOTE: Structured responses (response_format != None) with multimodal input
        # are less common and might not be supported well by all models/APIs.
        # This implementation assumes it behaves similarly to text-only structured generation.
        # You may need to add specific error handling or checks here.

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)
        response_text = None
        logprob_content = None
        cache_key = None

        # --- Structured Response Handling ---
        if response_format is not None:
            if self.capture_logits:
                print(f"Warning: Logit capture requested but response_format is {response_format}. Logits may not be captured for structured responses.")

            # Create a cache key suitable for multimodal structured requests
            text_parts = "_".join([item for item in content if isinstance(item, str)])
            img_hashes = "_".join([hashlib.sha256(item).hexdigest()[:8] for item in content if isinstance(item, bytes)])
            cache_key_structured = f"structured_multi_{response_format.__name__}_{sys_prompt_arg}_{text_parts[:50]}_{img_hashes}"

            if self.enable_cache:
                cache_or_none = self._check_cache(cache_key_structured)
                if cache_or_none is not None and isinstance(cache_or_none, response_format):
                    print(f"Cache hit for multimodal structured response: {response_format.__name__}")
                    return cache_or_none

            messages = [
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": formatted_content}, # Use formatted multimodal content
            ]

            try:
                # Logic similar to _generate_text for structured responses
                # Check model support carefully for multimodal + structured JSON output
                if not self.use_local_model and self.model_string in OPENAI_STRUCTURED_MODELS: # Example check
                     print(f"Attempting multimodal structured response with client.beta.chat.completions.parse for {response_format.__name__}")
                     # Parse might not support multimodal input directly, this could fail
                     response_parsed = self.client.beta.chat.completions.parse(
                         model=self.model_string, messages=messages,
                         temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                         response_format=response_format
                     )
                     parsed_object = response_parsed.choices[0].message.parsed
                else:
                    # Fallback: Standard completion + manual JSON parsing
                    print(f"Using standard multimodal completion + manual parsing for {response_format.__name__}")
                    completion_args_structured = {
                        "model": self.model_string, "messages": messages,
                        "temperature": temperature, "max_tokens": max_tokens, "top_p": top_p,
                        "response_format": {"type": "json_object"}
                    }
                    # Add schema instruction to the *last text part* of the user message if possible
                    schema_json = json.dumps(response_format.model_json_schema(), indent=2)
                    schema_instruction = f"\n\nPlease format your response as a JSON object matching this Pydantic schema:\n```json\n{schema_json}\n```"
                    
                    # Find last text item to append instruction
                    last_text_idx = -1
                    for idx, item in enumerate(messages[-1]["content"]):
                         if item.get("type") == "text":
                              last_text_idx = idx
                    if last_text_idx != -1:
                         messages[-1]["content"][last_text_idx]["text"] += schema_instruction
                    else:
                         # Append as a new text item if no text part exists
                         messages[-1]["content"].append({"type": "text", "text": schema_instruction})

                    completion_args_structured["messages"] = messages

                    response = self.client.chat.completions.create(**completion_args_structured)
                    raw_json_text = response.choices[0].message.content
                    try:
                        parsed_data = json.loads(raw_json_text)
                        parsed_object = response_format(**parsed_data)
                    except (json.JSONDecodeError, TypeError, ValueError) as parse_error:
                        print(f"Error parsing structured multimodal response manually: {parse_error}")
                        print(f"Raw response: {raw_json_text}")
                        return GenerationResult(error="parsing_error", error_details={"message": str(parse_error), "raw_text": raw_json_text})

                if self.enable_cache:
                     self._save_cache(cache_key_structured, parsed_object)

                return parsed_object # Return parsed Pydantic object

            except Exception as e:
                error_msg = f"Error generating multimodal structured response ({response_format.__name__}): {str(e)}"
                print(error_msg)
                # Return None to indicate failure to get structured object
                return None

        # --- Plain Text Multimodal Handling (response_format is None) ---
        else:
            if self.enable_cache:
                # Create cache key for multimodal text requests including image hash
                text_parts = "_".join([item for item in content if isinstance(item, str)])
                img_hashes = "_".join([hashlib.sha256(item).hexdigest()[:8] for item in content if isinstance(item, bytes)])
                cache_key = f"text_multi_{sys_prompt_arg}_{text_parts[:50]}_{img_hashes}_logits:{capture_logits}"
                cache_or_none = self._check_cache(cache_key)
                if cache_or_none is not None:
                    if isinstance(cache_or_none, dict) and "text" in cache_or_none:
                        return GenerationResult(**cache_or_none)
                    elif isinstance(cache_or_none, str) and not self.capture_logits:
                        return GenerationResult(text=cache_or_none)
                    else:
                        print("Invalid cache format found, regenerating...")
                        pass


            messages = [
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": formatted_content},
            ]

            completion_args = {
                "model": self.model_string,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }

            if self.capture_logits:
                completion_args["logprobs"] = True
                completion_args["top_logprobs"] = 5

            try:
                response = None # Initialize response

                if self.use_local_model:
                    # Use the request method for local models, assuming multimodal handled by endpoint
                    local_response = self._call_local_model_and_get_response(
                        messages=messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p
                    )
                    if local_response and "choices" in local_response and local_response["choices"]:
                        response_text = local_response["choices"][0].get("message", {}).get("content")
                        
                        # Enhanced logprob extraction for local models - same as _generate_text
                        if self.capture_logits:
                            choice = local_response["choices"][0]
                            if "logprobs" in choice:
                                print("Extracting logprobs from local multimodal model response")
                                if isinstance(choice["logprobs"], dict) and "content" in choice["logprobs"]:
                                    # Standard content format
                                    logprob_content = choice["logprobs"]["content"]
                                    print(f"Using standard multimodal content format with {len(logprob_content)} tokens")
                                elif isinstance(choice["logprobs"], dict):
                                    # Try to convert from tokens/token_logprobs format
                                    if "tokens" in choice["logprobs"] and "token_logprobs" in choice["logprobs"]:
                                        tokens = choice["logprobs"]["tokens"]
                                        token_logprobs = choice["logprobs"]["token_logprobs"]
                                        if len(tokens) == len(token_logprobs):
                                            print(f"Converting {len(tokens)} multimodal tokens to content format")
                                            logprob_content = [
                                                {"token": t, "logprob": lp} 
                                                for t, lp in zip(tokens, token_logprobs)
                                            ]
                                        else:
                                            print(f"Multimodal tokens and logprobs length mismatch: {len(tokens)} vs {len(token_logprobs)}")
                                            logprob_content = None
                                    else:
                                        print(f"Found multimodal logprobs but missing tokens/token_logprobs. Keys: {list(choice['logprobs'].keys())}")
                                        logprob_content = None
                                else:
                                    # Unusual format, try to use directly
                                    print(f"Non-dict multimodal logprobs format: {type(choice['logprobs'])}")
                                    logprob_content = choice["logprobs"]
                            else:
                                print("No logprobs found in local multimodal model response despite being requested")
                                logprob_content = None
                        else:
                            logprob_content = None
                    else:
                        error_msg = local_response.get("error", "Unknown error from local model call") if isinstance(local_response, dict) else "Invalid response from local model call"
                        return GenerationResult(error="local_model_error", error_details={"message": error_msg})

                # --- Handle OpenAI models (logic similar to _generate_text) ---
                # O1 models - Parse deprecated/less reliable, use standard completion
                elif self.model_string in ['o1', 'o1-mini']:
                     response = self.client.chat.completions.create(**completion_args)
                     response_text = response.choices[0].message.content
                     if self.capture_logits and hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                          logprob_content = response.choices[0].logprobs.content

                else: # Standard multimodal OpenAI call
                    response = self.client.chat.completions.create(**completion_args)
                    response_text = response.choices[0].message.content
                    
                    # Enhanced logprob extraction logic with detailed debugging - same as _generate_text
                    if self.capture_logits:
                        print(f"\nCapturing logprobs for multimodal model: {self.model_string}")
                        
                        # Check for different potential response structures
                        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                            # Standard logprobs structure
                            if hasattr(response.choices[0].logprobs, 'content'):
                                raw_logprob_content = response.choices[0].logprobs.content
                                # Convert ChatCompletionTokenLogprob objects to dictionaries if needed
                                try:
                                    logprob_content = []
                                    for item in raw_logprob_content:
                                        if hasattr(item, 'to_dict'):
                                            # If it has a to_dict method (like OpenAI's ChatCompletionTokenLogprob), use it
                                            logprob_content.append(item.to_dict())
                                        elif hasattr(item, 'model_dump'):
                                            # For Pydantic models
                                            logprob_content.append(item.model_dump())
                                        elif isinstance(item, dict):
                                            # It's already a dict
                                            logprob_content.append(item)
                                        else:
                                            # Try manual conversion - adjust based on the object's structure
                                            logprob_dict = {
                                                'token': getattr(item, 'token', ''),
                                                'logprob': getattr(item, 'logprob', -5.0),
                                                # Add other fields if present
                                                'bytes': getattr(item, 'bytes', None) if hasattr(item, 'bytes') else None,
                                                'top_logprobs': getattr(item, 'top_logprobs', None) if hasattr(item, 'top_logprobs') else None
                                            }
                                            logprob_content.append(logprob_dict)
                                    print(f"Converted {len(logprob_content)} logprob items to dict format")
                                except Exception as conv_err:
                                    print(f"Error converting logprobs to dict format: {conv_err}")
                                    # Fallback: set logprobs to None rather than failing
                                    logprob_content = None
                                print(f"Found multimodal logprobs in standard format")
                            elif isinstance(response.choices[0].logprobs, dict) and 'content' in response.choices[0].logprobs:
                                raw_logprob_content = response.choices[0].logprobs['content']
                                # Apply same conversion as above
                                try:
                                    logprob_content = []
                                    for item in raw_logprob_content:
                                        if hasattr(item, 'to_dict'):
                                            logprob_content.append(item.to_dict())
                                        elif hasattr(item, 'model_dump'):
                                            logprob_content.append(item.model_dump())
                                        elif isinstance(item, dict):
                                            logprob_content.append(item)
                                        else:
                                            logprob_dict = {
                                                'token': getattr(item, 'token', ''),
                                                'logprob': getattr(item, 'logprob', -5.0),
                                                'bytes': getattr(item, 'bytes', None) if hasattr(item, 'bytes') else None,
                                                'top_logprobs': getattr(item, 'top_logprobs', None) if hasattr(item, 'top_logprobs') else None
                                            }
                                            logprob_content.append(logprob_dict)
                                    print(f"Converted {len(logprob_content)} dict-content logprob items to dict format")
                                except Exception as conv_err:
                                    print(f"Error converting dict-content logprobs: {conv_err}")
                                    logprob_content = None
                                print(f"Found multimodal logprobs in dictionary format")
                            else:
                                # Try to get raw logprobs if available
                                logprob_content = response.choices[0].logprobs
                                print(f"Using raw multimodal logprobs object: {type(logprob_content)}")
                        elif hasattr(response, 'logprobs') and response.logprobs:
                            # Alternative position in response
                            logprob_content = response.logprobs
                            print(f"Found multimodal logprobs at response root level")
                        else:
                            print(f"Multimodal logprobs requested but not found in response. Response structure: {dir(response.choices[0])}")
                            logprob_content = None
                        
                        # --- Logits Saving ---
                        if hasattr(self, 'logits_dir'):
                            # Make sure json is imported
                            import time, uuid, json
                            timestamp = int(time.time())
                            unique_id = str(uuid.uuid4())[:8]
                            logits_filename = f"logits_api_multi_{timestamp}_{unique_id}.json"
                            logits_path = os.path.join(self.logits_dir, logits_filename)
                            # Need to handle image serialization in request if saving
                            serializable_messages = self._serialize_messages_for_log(messages)
                            # Ensure completion_args doesn't contain non-serializable items
                            serializable_args = {k: v for k, v in completion_args.items() if k != 'response_format'}
                            
                            # Save full response for debugging (same as in _generate_text)
                            try:
                                response_dump = response.model_dump()
                            except Exception as dump_err:
                                print(f"Error dumping multimodal response model: {dump_err}")
                                # Try alternative serialization
                                try:
                                    import json
                                    response_dump = json.loads(json.dumps(response, default=lambda o: f"<non-serializable: {type(o)}>"))
                                except Exception as json_err:
                                    print(f"Error serializing multimodal response: {json_err}")
                                    response_dump = {"error": "Could not serialize response"}
                            
                            logits_data = {
                                "timestamp": timestamp, "model": self.model_string,
                                "request": {"messages": serializable_messages, **serializable_args}, # Use serialized messages
                                "response": response_dump,
                                "logprob_content_found": logprob_content is not None,
                                "response_keys": dir(response),
                                "first_choice_keys": dir(response.choices[0]) if hasattr(response, 'choices') and len(response.choices) > 0 else []
                            }
                            try:
                                with open(logits_path, 'w') as f:
                                    json.dump(logits_data, f, indent=2, default=lambda o: f"<non-serializable: {type(o)}>")
                                print(f"\nSaved API multimodal response data to: {logits_path}")
                            except Exception as save_e:
                                print(f"Error saving multimodal logits to file: {save_e}")
                       # --- End Logits Saving ---

                # --- Prepare result ---
                result = GenerationResult(text=response_text, logprob_content=logprob_content)

                if self.enable_cache and result.text is not None:
                    self._save_cache(cache_key, result.dict(exclude_none=True))

                return result # Return GenerationResult for plain text multimodal

            except Exception as e:
                error_msg = f"Error generating multimodal response: {str(e)}"
                print(error_msg)
                # Return None to indicate failure to get structured object
                return None

    def _serialize_messages_for_log(self, messages):
        """Helper to make messages JSON serializable for logging, replacing image bytes."""
        serialized_messages = []
        for msg in messages:
            new_msg = msg.copy()
            if isinstance(new_msg.get("content"), list):
                new_content = []
                for item in new_msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        # Replace image data with placeholder for logging
                        new_item = item.copy()
                        new_item["image_url"] = {"url": "data:image/...[base64 data truncated]..."}
                        new_content.append(new_item)
                    else:
                        new_content.append(item)
                new_msg["content"] = new_content
            serialized_messages.append(new_msg)
        return serialized_messages

