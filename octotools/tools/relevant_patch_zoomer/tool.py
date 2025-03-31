import os
import cv2
from pydantic import BaseModel
from octotools.tools.base import BaseTool
from octotools.engine.openai import ChatOpenAI
import argparse

class PatchZoomerResponse(BaseModel):
    analysis: str
    patch: list[str]

class Relevant_Patch_Zoomer_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string=None, use_local_model=False, capture_logits=False, logits_dir=None):
        super().__init__(
            tool_name="Relevant_Patch_Zoomer_Tool",
            tool_description="A tool that analyzes an image, divides it into 5 regions (4 quarters + center), and identifies the most relevant patches based on a question. The returned patches are zoomed in by a factor of 2.",
            tool_version="1.0.0",
            input_types={
                "image": "str - The path to the image file.",
                "question": "str - The question about the image content.",
            },
            output_type="dict - Contains analysis text and list of saved zoomed patch paths.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(image="path/to/image.jpg", question="What is the color of the car?")',
                    "description": "Analyze image and return relevant zoomed patches that show the car's color."
                }
            ],
            user_metadata = {
                "best_practices": [
                    "It might be helpful to zoom in on the image first to get a better look at the object(s).",
                    "It might be helpful if the question requires a close-up view of the object(s), symbols, texts, etc.",
                    "The tool should be used to provide a high-level analysis first, and then use other tools for fine-grained analysis. For example, you can use Relevant_Patch_Zoomer_Tool first to get a zoomed patch of specific objects, and then use Image_Captioner_Tool to describe the objects in detail."
                ]
            }
        )
        self.matching_dict = {
            "A": "top-left",
            "B": "top-right",
            "C": "bottom-left",
            "D": "bottom-right",
            "E": "center"
        }

        self.model_string = model_string
        self.use_local_model = use_local_model
        self.capture_logits = capture_logits
        self.logits_dir = logits_dir
        
        model_type = "local" if use_local_model else "API"
        print(f"\nInitializing Patch Zoomer Tool with {model_type} model: {model_string or 'default'}")

    def _save_patch(self, image_path, patch, save_path, zoom_factor=2):
        """Extract and save a specific patch from the image with 10% margins."""
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        quarter_h = height // 2
        quarter_w = width // 2
        
        margin_h = int(quarter_h * 0.1)
        margin_w = int(quarter_w * 0.1)
        
        patch_coords = {
            'A': ((max(0, 0 - margin_w), max(0, 0 - margin_h)),
                  (min(width, quarter_w + margin_w), min(height, quarter_h + margin_h))),
            'B': ((max(0, quarter_w - margin_w), max(0, 0 - margin_h)),
                  (min(width, width + margin_w), min(height, quarter_h + margin_h))),
            'C': ((max(0, 0 - margin_w), max(0, quarter_h - margin_h)),
                  (min(width, quarter_w + margin_w), min(height, height + margin_h))),
            'D': ((max(0, quarter_w - margin_w), max(0, quarter_h - margin_h)),
                  (min(width, width + margin_w), min(height, height + margin_h))),
            'E': ((max(0, quarter_w//2 - margin_w), max(0, quarter_h//2 - margin_h)),
                  (min(width, quarter_w//2 + quarter_w + margin_w), 
                   min(height, quarter_h//2 + quarter_h + margin_h)))
        }
        
        (x1, y1), (x2, y2) = patch_coords[patch]
        patch_img = img[y1:y2, x1:x2]
        
        zoomed_patch = cv2.resize(patch_img, 
                                (patch_img.shape[1] * zoom_factor, 
                                 patch_img.shape[0] * zoom_factor), 
                                interpolation=cv2.INTER_LINEAR)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, zoomed_patch)
        return save_path

    def execute(self, image, question, zoom_factor=2):
        try:
            llm_engine = ChatOpenAI(
                model_string=self.model_string,
                is_multimodal=True,
                use_local_model=self.use_local_model,
                capture_logits=self.capture_logits,
                logits_dir=self.logits_dir
            )
            
            # Prepare the prompt
            prompt = f"""
Analyze this image to identify the most relevant region(s) for answering the question:

Question: {question}

The image is divided into 5 regions:
- (A) Top-left quarter
- (B) Top-right quarter
- (C) Bottom-left quarter
- (D) Bottom-right quarter
- (E) Center region (1/4 size, overlapping middle section)

Instructions:
1. First describe what you see in each of the five regions.
2. Then select the most relevant region(s) to answer the question.
3. Choose only the minimum necessary regions - avoid selecting redundant areas that show the same content. For example, if one patch contains the entire object(s), do not select another patch that only shows a part of the same object(s).

Response format:
<analysis>: Describe the image and five patches first. Then analyze the question and select the most relevant patch or list of patches.
<patch>: List of letters (A-E)
"""
            # Read image and create input data
            with open(image, 'rb') as file:
                image_bytes = file.read()
            input_data = [prompt, image_bytes]
            
            # Get response from LLM
            response = llm_engine(input_data, response_format=PatchZoomerResponse)
            
            # Save patches
            image_dir = os.path.dirname(image)
            image_name = os.path.splitext(os.path.basename(image))[0]
            
            # Update the return structure
            patch_info = []
            for patch in response.patch:
                patch_name = self.matching_dict[patch]
                save_path = os.path.join(self.output_dir, 
                                       f"{image_name}_{patch_name}_zoomed_{zoom_factor}x.png")
                saved_path = self._save_patch(image, patch, save_path, zoom_factor)
                save_path = os.path.abspath(saved_path)
                patch_info.append({
                    "path": save_path,
                    "description": f"The {self.matching_dict[patch]} region of the image: {image}."
                })
            
            return {
                "analysis": response.analysis,
                "patches": patch_info
            }
            
        except Exception as e:
            print(f"Error in patch zooming: {e}")
            return None

    def get_metadata(self):
        return super().get_metadata()

if __name__ == "__main__":
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/relevant_patch_zoomer
    python tool.py
    """
    parser = argparse.ArgumentParser(description="Run the Relevant Patch Zoomer Tool")
    parser.add_argument("--model", default=None, help="Model to use (defaults based on local vs API setting)")
    parser.add_argument("--image", default=None, help="Path to image file")
    parser.add_argument("--question", default=None, help="Question about the image content")
    parser.add_argument("--use-local-model", action="store_true", help="Use locally hosted model")
    parser.add_argument("--capture-logits", action="store_true", help="Capture and store logits (only works with local model)")
    parser.add_argument("--logits-dir", default="./captured_logits", help="Directory to store captured logits")
    
    args = parser.parse_args()
    
    # Create and run the tool
    tool = Relevant_Patch_Zoomer_Tool(
        model_string=args.model,
        use_local_model=args.use_local_model,
        capture_logits=args.capture_logits,
        logits_dir=args.logits_dir
    )

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tool.set_custom_output_dir(f"{script_dir}/zoomed_patches")

    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    # Default test case if no image/question is provided
    if args.image is None or args.question is None:
        # Construct the full path to the image using the script's directory
        relative_image_path = "examples/car.png"
        test_image_path = os.path.join(script_dir, relative_image_path)
        test_question = "What is the color of the car?"
        
        print(f"\nRunning test with default image: {test_image_path}")
        print(f"Test question: {test_question}")
        
        try:
            result = tool.execute(image=test_image_path, question=test_question)
            if result:
                print("\nDetected Patches:")
                for patch in result['patches']:
                    print(f"Path: {patch['path']}")
                    print(f"Description: {patch['description']}")
                    print()
        except Exception as e:
            print(f"Execution failed: {e}")
    else:
        # Run with provided arguments
        try:
            result = tool.execute(image=args.image, question=args.question)
            if result:
                print("\nDetected Patches:")
                for patch in result['patches']:
                    print(f"Path: {patch['path']}")
                    print(f"Description: {patch['description']}")
                    print()
        except Exception as e:
            print(f"Execution failed: {e}")

    print("Done!")
