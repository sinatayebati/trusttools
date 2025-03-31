# octotools/tools/python_code_generator/tool.py

import os
import re
import sys
from io import StringIO
import contextlib
import argparse

import threading
from octotools.tools.base import BaseTool
from octotools.engine.openai import ChatOpenAI

import signal
from contextlib import contextmanager

import platform
def is_windows_os():
    system=platform.system()
    return system == 'Windows'

# Custom exception for code execution timeout
class TimeoutException(Exception):
    pass

# Custom context manager for code execution timeout
@contextmanager
def timeout(seconds):
    
    if is_windows_os():
        # Windows timeout using threading.Timer
        def raise_timeout():
            raise TimeoutException("Code execution timed out")
        timer = threading.Timer(seconds, raise_timeout)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
            
    else:
        def timeout_handler(signum, frame):
            raise TimeoutException("Code execution timed out")

        # Set the timeout handler
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Restore the original handler and disable the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)


class Python_Code_Generator_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string=None, use_local_model=False, capture_logits=False, logits_dir=None):
        super().__init__(
            tool_name="Python_Code_Generator_Tool",
            tool_description="A tool that generates and executes simple Python code snippets for basic arithmetical calculations and math-related problems. The generated code runs in a highly restricted environment with only basic mathematical operations available.",
            tool_version="1.0.0",
            input_types={
                "query": "str - A clear, specific description of the arithmetic calculation or math problem to be solved, including any necessary numerical inputs."},
            output_type="dict - A dictionary containing the generated code, calculation result, and any error messages.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Calculate the factorial of 5")',
                    "description": "Generate a Python code snippet to calculate the factorial of 5."
                },
                {
                    "command": 'execution = tool.execute(query="Find the sum of prime numbers up to 50")',
                    "description": "Generate a Python code snippet to find the sum of prime numbers up to 50."
                },
                {
                    "command": 'query="Given the list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], calculate the sum of squares of odd numbers"\nexecution = tool.execute(query=query)',
                    "description": "Generate a Python function for a specific mathematical operation on a given list of numbers."
                },
            ],
            user_metadata = {
                "limitations": [
                    "Restricted to basic Python arithmetic operations and built-in mathematical functions.",
                    "Cannot use any external libraries or modules, including those in the Python standard library.",
                    "Limited to simple mathematical calculations and problems.",
                    "Cannot perform any string processing, data structure manipulation, or complex algorithms.",
                    "No access to any system resources, file operations, or network requests.",
                    "Cannot use 'import' statements.",
                    "All calculations must be self-contained within a single function or script.",
                    "Input must be provided directly in the query string.",
                    "Output is limited to numerical results or simple lists/tuples of numbers."
                ],
                "best_practices": [
                    "Provide clear and specific queries that describe the desired mathematical calculation.",
                    "Include all necessary numerical inputs directly in the query string.",
                    "Keep tasks focused on basic arithmetic, algebraic calculations, or simple mathematical algorithms.",
                    "Ensure all required numerical data is included in the query.",
                    "Verify that the query only involves mathematical operations and does not require any data processing or complex algorithms.",
                    "Review generated code to ensure it only uses basic Python arithmetic operations and built-in math functions."
                ]
            }
        )
        self.model_string = model_string
        self.use_local_model = use_local_model
        self.capture_logits = capture_logits
        self.logits_dir = logits_dir
        
        model_type = "local" if use_local_model else "API"
        print(f"\nInitializing Python_Code_Generator_Tool with {model_type} model: {model_string or 'default'}")

    @staticmethod
    def preprocess_code(code):
        """
        Preprocesses the generated code snippet by extracting it from the response.
        Returns only the first Python code block found.

        Parameters:
            code (str): The response containing the code snippet.

        Returns:
            str: The extracted code snippet from the first Python block.
            
        Raises:
            ValueError: If no Python code block is found.
        """
        # Look for the first occurrence of a Python code block
        match = re.search(r"```python\s*(.*?)\s*```", code, re.DOTALL)
        if not match:
            raise ValueError("No Python code block found in the response")
        return match.group(1).strip()

    @contextlib.contextmanager
    def capture_output(self):
        """
        Context manager to capture the standard output.

        Yields:
            StringIO: The captured output.
        """
        new_out = StringIO()
        old_out = sys.stdout
        sys.stdout = new_out
        try:
            yield sys.stdout
        finally:
            sys.stdout = old_out

    def execute_code_snippet(self, code):
        """
        Executes the given Python code snippet.

        Parameters:
            code (str): The Python code snippet to be executed.

        Returns:
            dict: A dictionary containing the printed output and local variables.
        """
        # Check for dangerous functions and remove them
        dangerous_functions = ['exit', 'quit', 'sys.exit']
        for func in dangerous_functions:
            if func in code:
                print(f"Warning: Removing unsafe '{func}' call from code")
                # Use regex to remove function calls with any arguments
                code = re.sub(rf'{func}\s*\([^)]*\)', 'break', code)

        try:
            execution_code = self.preprocess_code(code)

            # Execute with 10-second timeout
            with timeout(10):
                try:
                    exec(execution_code)
                except TimeoutException:
                    print("Error: Code execution exceeded 60 seconds timeout")
                    return {"error": "Execution timed out after 60 seconds"}
                except Exception as e:
                    print(f"Error executing code: {e}")
                    return {"error": str(e)}
                
            # Capture the output and local variables
            local_vars = {}
            with self.capture_output() as output:
                exec(execution_code, {}, local_vars)
            printed_output = output.getvalue().strip()

            # Filter out built-in variables and modules
            """
            only the variables used in the code are returned, 
            excluding built-in variables (which start with '__') and imported modules.
            """
            used_vars = {k: v for k, v in local_vars.items() 
                         if not k.startswith('__') and not isinstance(v, type(sys))}
            
            return {"printed_output": printed_output, "variables": used_vars}
        
        except Exception as e:
            print(f"Error executing code: {e}")
            return {"error": str(e)}

    def execute(self, query):
        """
        Generates and executes Python code based on the provided query.

        Parameters:
            query (str): A query describing the desired operation.

        Returns:
            dict: A dictionary containing the executed output, local variables, or any error message.
        """
        llm_engine = ChatOpenAI(
            model_string=self.model_string,
            is_multimodal=False,
            use_local_model=self.use_local_model,
            capture_logits=self.capture_logits,
            logits_dir=self.logits_dir
        )

        task_description = """
        Given a query, generate a Python code snippet that performs the specified operation on the provided data. Please think step by step. Ensure to break down the process into clear, logical steps. Make sure to print the final result in the generated code snippet with a descriptive message explaining what the output represents. The final output should be presented in the following format:

        ```python
        <code snippet>
        ```
        """
        task_description = task_description.strip()
        full_prompt = f"Task:\n{task_description}\n\nQuery:\n{query}"

        try:
            response = llm_engine(full_prompt)
            result_or_error = self.execute_code_snippet(response)
            return result_or_error
        except Exception as e:
            return {"error": f"Error generating or executing code: {str(e)}"}

    def get_metadata(self):
        """
        Returns the metadata for the Python_Code_Generator_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        metadata["require_llm_engine"] = self.require_llm_engine # NOTE: can be removed if not needed
        return metadata


if __name__ == "__main__":
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/python_code_generator
    python tool.py
    """
    parser = argparse.ArgumentParser(description="Run the Python Code Generator Tool")
    parser.add_argument("--model", default=None, help="Model to use (defaults based on local vs API setting)")
    parser.add_argument("--query", default=None, help="Query to generate code for")
    parser.add_argument("--use-local-model", action="store_true", help="Use locally hosted model")
    parser.add_argument("--capture-logits", action="store_true", help="Capture and store logits (only works with local model)")
    parser.add_argument("--logits-dir", default="./captured_logits", help="Directory to store captured logits")
    
    args = parser.parse_args()
    
    # Create and run the tool
    tool = Python_Code_Generator_Tool(
        model_string=args.model,
        use_local_model=args.use_local_model,
        capture_logits=args.capture_logits,
        logits_dir=args.logits_dir
    )

    metadata = tool.get_metadata()
    print(metadata)

    # Default test case if no query is provided
    if args.query is None:
        test_query = "Given the number list: [1, 2, 3, 4, 5], calculate the sum of all the numbers in the list."
        print(f"\nRunning test with default query: {test_query}")
        try:
            execution = tool.execute(query=test_query)
            print("\nExecution Result:", execution)
        except Exception as e:
            print(f"Execution failed: {e}")
    else:
        try:
            execution = tool.execute(query=args.query)
            print("\nExecution Result:", execution)
        except Exception as e:
            print(f"Execution failed: {e}")

    print("Done!")
