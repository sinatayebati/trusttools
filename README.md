# 🔐 TrustTools

## 📋 Table of Contents
- [Installation](#installation)
  - [Running Tools with API Models](#running-tools-with-api-models)
  - [Running Tools with Locally Hosted LLM](#running-tools-with-locally-hosted-llm)
- [Testing](#testing)
  - [Testing Individual Tools](#testing-individual-tools)
  - [Testing All Tools](#testing-all-tools)
- [Benchmarking](#benchmarking)

## 🔧 Installation

### Running Tools with API Models

1. **Create and activate conda environment**:
   ```sh
   # Create environment from yaml file
   conda env create -f conda.yaml
   
   # Activate the environment
   conda activate trustools
   
   # Install package in development mode
   pip install -e .
   ```

2. **Set up environment variables**:
   Create a `.env` file with your API keys:
   ```sh
   # The content of the .env file
   
   # Used for GPT-4o-powered tools
   OPENAI_API_KEY=<your-api-key-here>
   
   # Used for the Google Search tool
   GOOGLE_API_KEY=<your-api-key-here>
   GOOGLE_CX=<your-cx-here>
   
   # Used for the Advanced Object Detector tool (Optional)
   DINO_KEY=<your-dino-key-here>
   
   # Local LLM settings
   LOCAL_LLM_ENDPOINT=http://localhost:8000
   ```
   > 📝 Obtain a Google API Key and Google CX from the [Google Custom Search API](https://developers.google.com/custom-search/v1/overview) documentation.

3. **Install parallel** (for benchmark experiments):
   ```sh
   sudo apt-get update
   sudo apt-get install parallel
   ```

### Running Tools with Locally Hosted LLM

1. **Create and set up local LLM environment**:
   ```sh
   # Create environment
   conda create -n local-llm python=3.10
   conda activate local-llm
   
   # Install PyTorch with CUDA support (adjust based on your GPU)
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   
   # Install VLLM and other dependencies
   pip install -r trusttools/server/requirements_vllm.txt
   ```

2. **Start the VLLM server**:
   ```sh
   # Start with model from Hugging Face
   python -m trusttools.server.server_vllm --model-name "meta-llama/Llama-3.2-3B-Instruct" --port 8000 --enable-logits
   
   # OR use an already downloaded model
   python -m trusttools.server.server_vllm --model-name /path/to/downloaded/model --port 8000 --enable-logits
   ```

## 🧪 Testing

### Testing Individual Tools

Using `Generalist_Solution_Generator_Tool` as an example:

**With OpenAI API**:
```sh
cd trusttools/tools/generalist_solution_generator

# Test with logits and cache disabled
python -m trusttools.tools.generalist_solution_generator.tool --capture-logits --logits-dir "./captured_logits" --prompt "Explain the advantages of transformer models in natural language processing" --kwargs "enable_cache=False"
```

**With locally hosted LLM**:
```sh
cd trusttools/tools/generalist_solution_generator

# Basic test
python -m trusttools.tools.generalist_solution_generator.tool --use-local-model --prompt "Explain the advantages of transformer models in natural language processing"

# Test with logits capture
python -m trusttools.tools.generalist_solution_generator.tool --use-local-model --capture-logits --logits-dir "./captured_logits" --prompt "Explain the advantages of transformer models in natural language processing"

# With specific model and multimodal mode
python -m trusttools.tools.generalist_solution_generator.tool --use-local-model --model "llama-3.2-11b-vision-instruct" --prompt "Describe this image in detail" --image "/path/to/your/image.jpg"
```

### Testing All Tools

Run all tool tests at once:
```sh
cd trusttools/tools
source test_all_tools.sh
```

Expected output:
```
Testing advanced_object_detector...
✅ advanced_object_detector passed

Testing arxiv_paper_searcher...
✅ arxiv_paper_searcher passed

...

Testing wikipedia_knowledge_searcher...
✅ wikipedia_knowledge_searcher passed

Done testing all tools
Failed: 0
```

## 📊 Benchmarking

Using [CLEVR-Math](https://huggingface.co/datasets/dali-does/clevr-math) as an example:

```sh
cd tasks

# Run inference using GPT-4o only
source clevr-math/run_gpt4o.sh

# Run inference using the base tool
source clevr-math/run_trusttools_base.sh

# Run inference using trusttools with an optimized toolset
source clevr-math/run_trusttools.sh
```

> 💡 **Note**: Enable tools for your tasks by setting the `enabled_tools` argument in [tasks/solve.py](https://github.com/sinatayebati/deeptrust/blob/main/tasks/solve.py). You can enable the entire toolset or just a specific subset of tools.