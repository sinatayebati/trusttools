import os
import sys
import argparse
import subprocess
import json

def serve_model_with_vllm(
    model_name="meta-llama/Llama-3.2-3B-Instruct", 
    port=8000, 
    host="0.0.0.0", 
    tensor_parallel_size=1,
    max_model_len=4096,
    dtype="auto",
    enable_logits=True
):
    # Set environment variables for logits capture
    os.environ["VLLM_ENABLE_LOGPROBS"] = "1" if enable_logits else "0"
    
    if enable_logits:
        os.environ["VLLM_LOGPROBS_TOP_K"] = "5"
        os.environ["VLLM_LOGITS_ALL"] = "1"
    
    # Extract the base model name for API usage
    base_model_name = os.path.basename(model_name)
    
    server_url = f"http://{host}:{port}"
    print(f"\nServer Configuration:")
    print(f"Base URL: {server_url}")
    print(f"OpenAI-compatible endpoint: {server_url}/v1")
    print(f"Health check endpoint: {server_url}/v1/models")
    print(f"\nTo use this server, ensure your LOCAL_LLM_ENDPOINT is set to: {server_url}/v1")
    print(f"Current LOCAL_LLM_ENDPOINT setting: {os.getenv('LOCAL_LLM_ENDPOINT', 'not set')}")
    
    if os.getenv('LOCAL_LLM_ENDPOINT') != f"{server_url}/v1":
        print("\n⚠️  Warning: Your LOCAL_LLM_ENDPOINT environment variable doesn't match the server URL.")
        print(f"Consider updating it to: {server_url}/v1")
    
    print(f"Using model name: {model_name}")
    
    # Normalize model name
    if "/" not in model_name and model_name.lower().startswith("llama-3"):
        # Convert to HuggingFace format if needed
        print(f"Converting model name from '{model_name}' to HuggingFace format")
        if model_name.lower() in ["llama-3.2-3b-instruct", "llama-3.2-3b"]:
            model_name = "meta-llama/Llama-3.2-3B-Instruct"
        elif model_name.lower() in ["llama-3.2-11b-vision-instruct", "llama-3.2-11b-vision"]:
            model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    print(f"Using model name: {model_name}")
    
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--dtype", dtype
    ]
    
    print(f"Starting VLLM OpenAI-compatible server with model: {model_name}")
    print(f"Logits capture enabled: {enable_logits}")
    if enable_logits:
        print(f"Logits configuration: VLLM_ENABLE_LOGPROBS=1, VLLM_LOGPROBS_TOP_K=5, VLLM_LOGITS_ALL=1")
    print(f"Server will be available at: http://{host}:{port}/v1")
    print(f"Running command: {' '.join(command)}")
    
    # Before running server, print info about the actual model format the VLLM will expose
    print("❗ IMPORTANT: When connecting to this server, use the following model name:")
    print(f"❗ Model name to use in API calls: {base_model_name}")
    
    try:
        subprocess.run(command, env=os.environ)
    except KeyboardInterrupt:
        print("\nServer stopped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--enable-logits", action="store_true", help="Enable logprobs in the API via environment variables")
    
    args = parser.parse_args()
    
    serve_model_with_vllm(
        model_name=args.model_name,
        port=args.port,
        host=args.host,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enable_logits=args.enable_logits
    ) 