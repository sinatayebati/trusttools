## Installation

Create a conda environment for hosting local LLMs using VLLM server:

```sh
conda create -n local-llm python=3.10
conda activate local-llm

# Install PyTorch with CUDA support (adjust based on your GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install VLLM and other dependencies
pip install -r octotools/server/requirements_vllm.txt
```

Create a conda environment from the `conda.yaml` file:

```sh
conda env create -f conda.yaml
```

Activate the environment and install requirements:

```sh
conda activate octotools
pip install -e .
```

Make `.env` file, and set `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_CX`, etc. For example:

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

Obtain a Google API Key and Google CX according to the [Google Custom Search API](https://developers.google.com/custom-search/v1/overview) documation.

Install `parallel` for running benchmark experiments in parallel:

```sh
sudo apt-get update
sudo apt-get install parallel
```

## Start the VLLM server for local LLM hosting

```sh
# Start the LLM server
python -m octotools.server.server_vllm --model-name "meta-llama/Llama-3.2-3B-Instruct" --port 8000 --enable-logits
# Or use following if already downloaded the model
python -m octotools.server.server_vllm --model-name /path/to/downloaded/model --port 8000 --enable-logits
```

## Test tools in the toolbox

Testing tools using OpenAI API: Using `Python_Code_Generator_Tool` as an example, test the availability of the tool by running the following:

```sh
cd octotools/tools/python_code_generator
python tool.py
```

Expected output:

```
Execution Result: {'printed_output': 'The sum of all the numbers in the list is: 15', 'variables': {'numbers': [1, 2, 3, 4, 5], 'total_sum': 15}}
```

Testing tools using locally hosted LLM: Using `Generalist_Solution_Generator_Tool` as an example, test the availability of the tool by running the following:

```sh
cd octotools/tools/generalist_solution_generator

# test without logits
python -m octotools.tools.generalist_solution_generator.tool --use-local-model --prompt "Explain the advantages of transformer models in natural language processing"

# test with logits
python -m octotools.tools.generalist_solution_generator.tool --use-local-model --capture-logits --logits-dir "./captured_logits" --prompt "Explain the advantages of transformer models in natural language processing"

# with specific model and multimodal mode
python -m octotools.tools.generalist_solution_generator.tool --use-local-model --model "llama-3.2-11b-vision-instruct" --prompt "Describe this image in detail" --image "/path/to/your/image.jpg"
```

You can also test all tools available in the toolbox by running the following:

```sh
cd octotools/tools
source test_all_tools.sh
```

Expected testing log:

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

## Run inference on benchmarks

Using [CLEVR-Math](https://huggingface.co/datasets/dali-does/clevr-math) as an example, run inference on a benchmark by:

```sh
cd octotools/tasks

# Run inference from clevr-math using GPT-4 only
source clevr-math/run_gpt4o.sh

# Run inference from clevr-math using the base tool
source clevr-math/run_octotool_base.sh

# Run inference from clevr-math using Octotools with an optimized toolset
source clevr-math/run_octotools.sh
```

More benchmarks are available in the [tasks](https://octotools.github.io/#tasks).


## Experiments


### Main results

To demonstrate the generality of our **OctoTools** framework, we conduct comprehensive evaluations on 16 diverse benchmarks spanning two modalities, five domains, and four reasoning types. These benchmarks encompass a wide range of complex reasoning tasks, including visual understanding, numerical calculation, knowledge retrieval, and multi-step reasoning.


<p align="center">
    <img src="assets/result/result_table_1.png" width="100%">
    <!-- Text. -->
</p>


More results are available in the [paper](https://arxiv.org/pdf/2502.11271) or at the [project page](https://octotools.github.io/).


### In-depth analysis

We provide a set of in-depth analyses to help you understand the framework. For instance, we visualize the tool usage of **OctoTools** and its baselines  from 16 tasks. It turns out that **OctoTools** takes advantage of different external tools to address task-specific challenges. Explore more findings at our [paper](https://arxiv.org/pdf/2502.11271) or the [project page](https://octotools.github.io/#analysis).

<a align="center">
    <img src="assets/result/tool_usage_ours_baselines.png" width="100%">
    <!-- Text. -->
</a>

### Example visualizations

We provide a set of example visualizations to help you understand the framework. Explore them at the [project page](https://octotools.github.io/#visualization).

<p align="center">  
    <a href="https://octotools.github.io/#visualization">
        <img src="assets/result/example_visualization.png" width="80%">
    </a>
</p>


## Customize OctoTools

The design of each tool card is modular relative to the **OctoTools** framework, enabling users to integrate diverse tools without modifying the underlying framework or agent logic. New tool cards can be added, replaced, or updated with minimal effort, making **OctoTools** robust and extensible as tasks grow in complexity.

<p align="center">
    <a href="https://octotools.github.io/#tool_cards">
        <img src="assets/models/tool_cards.png" width="100%">
    </a>
</p>

To customize **OctoTools** for your own tasks:

1. **Add a new tool card**: Implement your tool following the structure in [existing tools](https://github.com/OctoTools/OctoTools/tree/main/octotools/tools).

2. **Replace or update existing tools**: You can replace or update tools in the toolbox. For example, we provide the [`Object_Detector_Tool`](https://github.com/OctoTools/OctoTools/blob/main/octotools/tools/object_detector/tool.py) to detect objects in images using an open-source model. We also provide an alternative tool called the [`Advanced_Object_Detector_Tool`](https://github.com/OctoTools/OctoTools/blob/main/octotools/tools/advanced_object_detector/tool.py) to detect objects in images using API calls.

3. **Enable tools for your tasks**: You can enable the whole toolset or a subset of tools for your own tasks by setting the `enabled_tools` argument in [tasks/solve.py](https://github.com/OctoTools/OctoTools/blob/main/octotools/tasks/solve.py).


## Resources

### Inspiration

This project draws inspiration from several remarkable projects:

- 📕 [Chameleon](https://github.com/lupantech/chameleon-llm) – Chameleon is an early attempt that augments LLMs with tools, which is a major source of inspiration. A journey of a thousand miles begins with a single step.
- 📘 [TextGrad](https://github.com/mert-y/textgrad) – We admire and appreciate TextGrad for its innovative and elegant framework design.
- 📗 [AutoGen](https://github.com/microsoft/autogen) – A trending project that excels in building agentic systems.
- 📙 [LangChain](https://github.com/langchain-ai/langchain) – A powerful framework for constructing agentic systems, known for its rich functionalities.
