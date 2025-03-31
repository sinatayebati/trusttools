import os
import argparse
from octotools.tools.base import BaseTool
from octotools.engine.openai import ChatOpenAI

class Image_Captioner_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string=None, use_local_model=False, capture_logits=False, logits_dir=None):
        super().__init__(
            tool_name="Image_Captioner_Tool",
            tool_description="A tool that generates descriptive captions for images using multimodal models.",
            tool_version="1.0.0",
            input_types={
                "image": "str - The path to the image file to be captioned.",
                "prompt": "str - Optional custom prompt to guide the captioning (default: 'Describe this image in detail').",
            },
            output_type="str - Generated caption describing the image",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(image="path/to/image.png")',
                    "description": "Generate a caption for an image using the default prompt and model."
                },
                {
                    "command": 'execution = tool.execute(image="path/to/image.png", prompt="Explain the mood of this scene.")',
                    "description": "Generate a caption focusing on the mood using a specific prompt and model."
                }
            ],
            user_metadata = {
                "limitation": "The Image_Captioner_Tool provides general image descriptions but has limitations: 1) May make mistakes in complex scenes, counting, attribute detection, and understanding object relationships. 2) Might not generate comprehensive captions, especially for images with multiple objects or abstract concepts. 3) Performance varies with image complexity. 4) Struggles with culturally specific or domain-specific content. 5) May overlook details or misinterpret object relationships. For precise descriptions, consider: using it with other tools for context/verification, as an initial step before refinement, or in multi-step processes for ambiguity resolution. Verify critical information with specialized tools or human expertise when necessary."
            },
        )
        self.model_string = model_string
        self.use_local_model = use_local_model
        self.capture_logits = capture_logits
        self.logits_dir = logits_dir

    def execute(self, image, prompt="Describe this image in detail"):
        model_type = "local" if self.use_local_model else "API"
        print(f"\nInitializing Image Captioner with {model_type} model: {self.model_string or 'default'}")

        llm_engine = ChatOpenAI(
            model_string=self.model_string,
            is_multimodal=True,
            use_local_model=self.use_local_model,
            capture_logits=self.capture_logits,
            logits_dir=self.logits_dir
        )

        try:
            if not os.path.isfile(image):
                return "Error: Invalid image file path."
            
            try:
                with open(image, 'rb') as file:
                    image_bytes = file.read()
            except Exception as e:
                return f"Error reading image file: {str(e)}"

            response = llm_engine([prompt, image_bytes])
            return response
        except Exception as e:
            return f"Error generating caption: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata['require_llm_engine'] = self.require_llm_engine # NOTE: can be removed if not needed
        return metadata

if __name__ == "__main__":
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/image_captioner
    python tool.py
    """
    parser = argparse.ArgumentParser(description="Run the Image Captioner Tool")
    parser.add_argument("--model", default=None, help="Model to use (defaults based on local vs API setting)")
    parser.add_argument("--image", default=None, help="Path to image file")
    parser.add_argument("--prompt", default="Describe this image in detail", help="Custom prompt for captioning")
    parser.add_argument("--use-local-model", action="store_true", help="Use locally hosted model")
    parser.add_argument("--capture-logits", action="store_true", help="Capture and store logits (only works with local model)")
    parser.add_argument("--logits-dir", default="./captured_logits", help="Directory to store captured logits")
    
    args = parser.parse_args()
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create and run the tool
    tool = Image_Captioner_Tool(
        model_string=args.model,
        use_local_model=args.use_local_model,
        capture_logits=args.capture_logits,
        logits_dir=args.logits_dir
    )

    metadata = tool.get_metadata()
    print(metadata)

    # Construct the full path to the image using the script's directory
    relative_image_path = "examples/baseball.png"
    image_path = os.path.join(script_dir, relative_image_path)

    # Default test case if no image is provided
    if args.image is None:
        try:
            result = tool.execute(image=image_path)
            print("\nGenerated Caption:")
            print(result)
        except Exception as e:
            print(f"Error in test execution: {e}")
    else:
        try:
            result = tool.execute(image=args.image, prompt=args.prompt)
            print("\nGenerated Caption:")
            print(result)
        except Exception as e:
            print(f"Error: {e}")

    print("Done!")
