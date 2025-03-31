import os
from octotools.tools.base import BaseTool
from octotools.engine.openai import ChatOpenAI
import argparse

class Generalist_Solution_Generator_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string=None, use_local_model=False, capture_logits=False, logits_dir=None):
        super().__init__(
            tool_name="Generalist_Solution_Generator_Tool",
            tool_description="A generalized tool that takes query from the user as prompt, and answers the question step by step to the best of its ability. It can also accept an image.",
            tool_version="1.0.0",
            input_types={
                "prompt": "str - The prompt that includes query from the user to guide the agent to generate response (Examples: 'Describe this image in detail').",
                "image": "str - The path to the image file if applicable (default: None).",
            },
            output_type="str - The generated response to the original query prompt",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(prompt="Summarize the following text in a few lines")',
                    "description": "Generate a short summary given the prompt from the user."
                },
                {
                    "command": 'execution = tool.execute(prompt="Explain the mood of this scene.", image="path/to/image1.png")',
                    "description": "Generate a caption focusing on the mood using a specific prompt and image."
                },
                {
                    "command": 'execution = tool.execute(prompt="Give your best coordinate estimate for the pacemaker in the image and return (x1, y1, x2, y2)", image="path/to/image2.png")',
                    "description": "Generate bounding box coordinates given the image and prompt from the user. The format should be (x1, y1, x2, y2)."
                },
                {
                    "command": 'execution = tool.execute(prompt="Is the number of tiny objects that are behind the small metal jet less than the number of tiny things left of the tiny sedan?", image="path/to/image2.png")',
                    "description": "Answer a question step by step given the image."
                }
            ],

            user_metadata = {
                "limitation": "The Generalist_Solution_Generator_Tool may provide hallucinated or incorrect responses.",
                "best_practice": "Use the Generalist_Solution_Generator_Tool for general queries or tasks that don't require specialized knowledge or specific tools in the toolbox. For optimal results:\n\n"
                "1) Provide clear, specific prompts.\n"
                "2) Use it to answer the original query through step by step reasoning for tasks without complex or multi-step reasoning.\n"
                "3) For complex queries, break them down into subtasks and use the tool multiple times.\n"
                "4) Use it as a starting point for complex tasks, then refine with specialized tools.\n"
                "5) Verify important information from its responses.\n"
                "6) For image-related tasks, ensure the image path is correct and the prompt is relevant to the image content."
            }

        )
        self.model_string = model_string
        self.use_local_model = use_local_model
        self.capture_logits = capture_logits
        self.logits_dir = logits_dir

    def execute(self, prompt, image=None):
        model_type = "local" if self.use_local_model else "API"
        print(f"\nInitializing Generalist Tool with {model_type} model: {self.model_string or 'default'}")
        
        multimodal = True if image else False
        llm_engine = ChatOpenAI(
            model_string=self.model_string, 
            is_multimodal=multimodal,
            use_local_model=self.use_local_model,
            capture_logits=self.capture_logits,
            logits_dir=self.logits_dir
        )

        try:
            input_data = [prompt]
            if multimodal:
                if not os.path.isfile(image):
                    return "Error: Invalid image file path."
                try:
                    with open(image, 'rb') as file:
                        image_bytes = file.read()
                    input_data.append(image_bytes)
                except Exception as e:
                    return f"Error reading image file: {str(e)}"

                response = llm_engine(input_data)
            else:
                response = llm_engine(input_data[0])
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/generalist_solution_generator
    python tool.py
    """
    parser = argparse.ArgumentParser(description="Run the Generalist Solution Generator Tool")
    parser.add_argument("--model", default=None, help="Model to use (defaults based on local vs API setting)")
    parser.add_argument("--prompt", default="Explain the concept of machine learning", help="Prompt to send to the model")
    parser.add_argument("--image", default=None, help="Path to image (optional)")
    parser.add_argument("--use-local-model", action="store_true", help="Use locally hosted model")
    parser.add_argument("--capture-logits", action="store_true", help="Capture and store logits (only works with local model)")
    parser.add_argument("--logits-dir", default="./captured_logits", help="Directory to store captured logits")
    
    args = parser.parse_args()
    
    # Create and run the tool
    tool = Generalist_Solution_Generator_Tool(
        model_string=args.model,
        use_local_model=args.use_local_model,
        capture_logits=args.capture_logits,
        logits_dir=args.logits_dir
    )

    metadata = tool.get_metadata()
    print(metadata)

    test_prompt = "Explain the concept of machine learning"
    
    if args.prompt is None:
        try:
            result = tool.execute(prompt=test_prompt, image=None)
            print("\nGenerated Response:")
            print(result)
        except Exception as e:
            print(f"Error: {e}")
    else:
        try:
            result = tool.execute(prompt=args.prompt, image=args.image)
            print("\nGenerated Response:")
            print(result)
        except Exception as e:
            print(f"Error: {e}")
    print("Done!")
