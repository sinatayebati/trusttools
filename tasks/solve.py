import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from trusttools.engine.openai import GenerationResult
from trusttools.models.initializer import Initializer
from trusttools.models.planner import Planner
from trusttools.models.memory import Memory
from trusttools.models.executor import Executor
from trusttools.models.utlis import make_json_serializable_truncated
from trusttools.models.validator import ConformalValidator, Atom

class Solver:
    def __init__(
        self,
        planner: Planner,
        memory: Memory,
        executor: Executor,
        validator: ConformalValidator,
        task: str,
        data_file: str,
        task_description: str,
        output_types: str = "base,final,direct,validated_final,validated_direct",
        index: int = 0,
        verbose: bool = True,
        max_steps: int = 10,
        max_time: int = 60,
        max_tokens: int = 4000,
        output_json_dir: str = "results",
        root_cache_dir: str = "cache",
        validation_alpha: float = 0.1
    ):
        self.planner = planner
        self.memory = memory
        self.executor = executor
        self.validator = validator
        self.task = task
        self.data_file = data_file
        self.task_description = task_description
        self.index = index
        self.verbose = verbose
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.output_json_dir = output_json_dir
        self.root_cache_dir = root_cache_dir
        self.validation_alpha = validation_alpha

        self.output_types = output_types.lower().split(',')
        self.requires_validation = any(vt in output_types for vt in ['validated_final', 'validated_direct'])
        if self.requires_validation:
            print("Validation requested for final/direct outputs.")
            self.planner.capture_logits_for_outputs = True
            print("Planner configured to capture logits for outputs.")

        assert all(output_type in ["base", "final", "direct", "validated_final", "validated_direct"] for output_type in self.output_types), \
            "Invalid output type. Supported: base, final, direct, validated_final, validated_direct."

        self.benchmark_data = self.load_benchmark_data()

    def load_benchmark_data(self) -> List[Dict[str, Any]]:
        # Add task description to the query
        if self.task_description:
            print(f"Task description: {self.task_description}")
            self.task_description = f"Task description: {self.task_description}\n"

        with open(self.data_file, 'r') as f:
            data = json.load(f) 
        for problem in data:
            problem['query'] = problem['query'] if 'query' in problem else problem['question']
            if self.task_description:
                problem['query'] = self.task_description + problem['query']

            if 'image' in problem and problem['image'] not in [None, ""]:
                # NOTE: This is a hack to make the absolute image path relative to the data file
                problem['image'] = os.path.abspath(os.path.join(os.path.dirname(self.data_file), problem['image']))
                assert os.path.exists(problem['image']), f"Error: Image file {problem['image']} does not exist."

        return data

    def solve(self):
        total_problems = len(self.benchmark_data)

        # Solve a single problem
        if self.index is not None:
            if not 0 <= self.index < total_problems:
                print(f"Error: Invalid problem index {self.index}. Valid indices are 0 to {total_problems-1}).")
            else:
                self.solve_single_problem(self.index)
            return

    def solve_single_problem(self, index: int):
        """
        Solve a single problem from the benchmark dataset.
        
        Args:
            index (int): Index of the problem to solve
        """
        # Update cache directory for the executor
        _cache_dir = os.path.join(self.root_cache_dir, f"{index}")
        self.executor.set_query_cache_dir(_cache_dir)
    
        # Create output directory and file path
        json_dir = os.path.join(self.output_json_dir)
        os.makedirs(json_dir, exist_ok=True)
        output_file = os.path.join(json_dir, f"output_{index}.json")

        # Get the problem
        problem = self.benchmark_data[index]
        # use 'query' by default for LLM inputs
        question = problem.get("query") if "query" in problem else problem["question"]
        image_path = problem['image']
        print(f"image_path: {image_path}")  
        pid = problem['pid']
        answer = problem['answer']

        if self.verbose:
            print("\n\n")
            print("#"*100)
            print(f"## Problem {index}:")
            print(f"Question:\n{question}")
            print(f"Image: {image_path}")
            print("#"*100)

        # Initialize json_data with basic problem information
        json_data = {
            "pid": pid,
            "query": question,
            "image": image_path,
            "answer": answer,
            "query_analysis": "",  # Initialize empty, will be populated later
            "outputs": {},
            "validation_results": {}
        }

        if 'metadata' in problem:
            json_data['metadata'] = problem['metadata']

        # Generate base response if requested
        if 'base' in self.output_types:
            base_result: GenerationResult = self.planner.llm_engine_mm.generate(
                [question] + ([image_path] if image_path else []),
                max_tokens=self.max_tokens,
                capture_logits=False
            )
            base_response = base_result.text or "[Error generating base response]"
            json_data["outputs"]["base"] = base_response
            if self.verbose:
                print("\n## Base Response:")
                print("#"*50)
                print(f"{base_response}")
                print("#"*50)

        # If only base response is needed, save and return
        if set(self.output_types) == {'base'}:
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=4)
                print(f"\n==>Base response output saved to: {output_file}")
            return
    
        # Continue with query analysis and tool execution if final or direct responses are needed
        if {'final', 'direct', 'validated_final', 'validated_direct'} & set(self.output_types):

             # Analyze query
            query_analysis = self.planner.analyze_query(question, image_path)
            json_data["query_analysis"] = query_analysis

            if self.verbose:
                print("\n## Query Analysis:")
                print("#"*50)
                print(f"{query_analysis}")
                print("#"*50)

            start_time = time.time()
            step_count = 0
            action_times = []

            # Main execution loop
            while step_count < self.max_steps and (time.time() - start_time) < self.max_time:
                step_count += 1
                if self.verbose:
                    print(f"\n## [Step {step_count}]")

                # Generate next step
                start_time_step = time.time()
                next_step = self.planner.generate_next_step(
                    question, 
                    image_path, 
                    query_analysis, 
                    self.memory, 
                    step_count, 
                    self.max_steps
                )
                context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)

                if self.verbose:
                    print(f"\n## [{step_count}] Next Step:")
                    print("#"*50)
                    print(f"Next Step:\n{next_step}")
                    print("#"*50)
                    print(f"\n==>Extracted Context:\n{context}")
                    print(f"\n==>Extracted Sub-goal:\n{sub_goal}\n")
                    print(f"\n==>Extracted Tool:\n{tool_name}")

                if tool_name is None or tool_name not in self.planner.available_tools:
                    print(f"Error: Tool '{tool_name}' is not available or not found.")
                    command = "Not command is generated due to the tool not found."
                    result = "Not result is generated due to the tool not found."

                else:
                    # Generate the tool command
                    tool_command = self.executor.generate_tool_command(
                        question, 
                        image_path, 
                        context, 
                        sub_goal, 
                        tool_name, 
                        self.planner.toolbox_metadata[tool_name]
                    )
                    explanation, command = self.executor.extract_explanation_and_command(tool_command)
                    
                    if self.verbose:
                        print(f"\n## [{step_count}] Tool Command:")
                        print("#"*50)
                        print(f"{tool_command}")
                        print("#"*50)
                        print(f"\n==>Extracted Command:\n{command}\n")

                    # Execute command
                    execution_result = self.executor.execute_tool_command(tool_name, command)
                    
                    # --- Normalize result structure ---
                    # execute_tool_command returns a list, extract single result if applicable
                    if isinstance(execution_result, list) and len(execution_result) == 1:
                         single_result = execution_result[0]
                    else:
                         # Handle multiple results or unexpected format if necessary
                         # For now, keep as is if not a single-item list
                         single_result = execution_result
                    # --------------------------------
                    
                    # Make result JSON serializable *before* adding to memory
                    result = make_json_serializable_truncated(single_result) 

                    if self.verbose:
                        print(f"\n## [{step_count}] Tool Execution:")
                        print("\n==>Executed Result:")
                        print(json.dumps(result, indent=4))

                # Track execution time
                end_time_step = time.time()
                execution_time_step = round(end_time_step - start_time_step, 2)
                action_times.append(execution_time_step)

                if self.verbose:
                    print(f"Execution time for step {step_count}: {execution_time_step:.2f} seconds")

                # Update memory
                self.memory.add_action(step_count, tool_name, sub_goal, command, result, logprob_content=None)
                memeory_actions = self.memory.get_actions()

                # Verify memory
                stop_verification = self.planner.verificate_context(
                    question, 
                    image_path, 
                    query_analysis, 
                    self.memory
                )
                conclusion = self.planner.extract_conclusion(stop_verification)
                
                if self.verbose:
                    print(f"\n## [{step_count}] Stopping Verification:")
                    print("#"*50)
                    print(f"{stop_verification}")
                    print("#"*50)
                    print(f"\n==>Extracted Conclusion:\n{conclusion}")

                if conclusion == 'STOP':
                    break

            # Check if we've hit a limit
            if self.verbose:
                if step_count >= self.max_steps:
                    print(f"\n==>Maximum number of steps ({self.max_steps}) reached. Stopping execution.")
                elif (time.time() - start_time) >= self.max_time:
                    print(f"\n==>Maximum time limit ({self.max_time} seconds) reached. Stopping execution.")

                # Print memory
                print(f"\n## [{step_count}] Memory:")
                print("#"*50)
                if isinstance(memeory_actions, dict):
                    print(json.dumps(memeory_actions, indent=4))
                elif isinstance(memeory_actions, list):
                    print(json.dumps(memeory_actions, indent=4))
                else:
                    print(memeory_actions)
                print("#"*50)

            # Add memory and statistics to json_data
            json_data.update({
                "memory": memeory_actions,
                "step_count": step_count,
                "execution_time": round(time.time() - start_time, 2),
            })

            # Generate final output if requested
            if 'final' in self.output_types:
                final_output = self.planner.generate_final_output(question, image_path, self.memory, step_count + 1)
                json_data["outputs"]["final"] = final_output
                if self.verbose:
                    print("\n## Final Output:")
                    print("#"*50)
                    print(f"{final_output}")
                    print("#"*50)

            # Generate direct output if requested
            if 'direct' in self.output_types:
                direct_output = self.planner.generate_direct_output(question, image_path, self.memory, step_count + 2)
                json_data["outputs"]["direct"] = direct_output
                if self.verbose:
                    print("\n## Direct Output:")
                    print("#"*50)
                    print(f"{direct_output}")
                    print("#"*50)

        # Save results
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=4)
            print(f"\n==>Output saved to: {output_file}")

        # Print execution statistics if we ran the full pipeline
        if {'final', 'direct', 'validated_final', 'validated_direct'} & set(self.output_types):
            print(f"\n## Execution Statistics for Problem {index}:")
            print(f"==>Total steps executed: {step_count}")
            print(f"==>Total execution time: {time.time() - start_time:.2f} seconds")
            
        # Conformal Validation Step
        if self.requires_validation:
            if self.verbose: print("\n--- Performing Conformal Validation ---")

            if 'final' in json_data["outputs"] and any(t == 'validated_final' for t in self.output_types):
                action_name = f"Action Step {step_count + 1}"
                if f"{action_name}_FinalOutputGenerator" in json_data["memory"]:
                    action_name = f"{action_name}_FinalOutputGenerator"

                final_action = self.memory.get_action(action_name)
                if final_action and final_action.get('result'):
                    original_final = final_action['result']
                    logprobs_final = final_action.get('logprob_content')
                    if logprobs_final:
                        validated_final, final_atoms = self.validator.validate(
                            original_final, logprobs_final, alpha=self.validation_alpha
                        )
                        json_data["outputs"]["validated_final"] = validated_final
                        json_data["validation_results"]["final_output_atoms"] = [atom.dict_without_logprobs() for atom in final_atoms]
                        if self.verbose:
                            print(f"\n## Validated Final Output (alpha={self.validation_alpha}):")
                            print("#"*50 + f"\n{validated_final}\n" + "#"*50)
                    else:
                        print("Skipping validation for final_output: Logprobs not found in memory.")
                        json_data["outputs"]["validated_final"] = "[Validation Skipped: No Logprobs]"
                else:
                     print(f"Skipping validation for final_output: Action '{action_name}' not found or has no result in memory.")
                     json_data["outputs"]["validated_final"] = "[Validation Skipped: Action/Result Not Found]"

            if 'direct' in json_data["outputs"] and any(t == 'validated_direct' for t in self.output_types):
                action_name = f"Action Step {step_count + 2}"
                if f"{action_name}_DirectOutputGenerator" in json_data["memory"]:
                     action_name = f"{action_name}_DirectOutputGenerator"

                direct_action = self.memory.get_action(action_name)
                if direct_action and direct_action.get('result'):
                     original_direct = direct_action['result']
                     logprobs_direct = direct_action.get('logprob_content')
                     if logprobs_direct:
                          validated_direct, direct_atoms = self.validator.validate(
                              original_direct, logprobs_direct, alpha=self.validation_alpha
                          )
                          json_data["outputs"]["validated_direct"] = validated_direct
                          json_data["validation_results"]["direct_output_atoms"] = [atom.dict_without_logprobs() for atom in direct_atoms]
                          if self.verbose:
                              print(f"\n## Validated Direct Output (alpha={self.validation_alpha}):")
                              print("#"*50 + f"\n{validated_direct}\n" + "#"*50)
                     else:
                          print("Skipping validation for direct_output: Logprobs not found in memory.")
                          json_data["outputs"]["validated_direct"] = "[Validation Skipped: No Logprobs]"
                else:
                      print(f"Skipping validation for direct_output: Action '{action_name}' not found or has no result in memory.")
                      json_data["outputs"]["validated_direct"] = "[Validation Skipped: Action/Result Not Found]"

        # --- Save Final Results ---
        # Clean up memory before saving (logprobs can be large)
        # if "memory" in json_data:
        #     for action in json_data["memory"].values():
        #         action.pop('logprob_content', None) # Remove logprobs before saving JSON

        # Final save
        with open(output_file, 'w') as f:
            # Use the utility to handle potential non-serializable types in results/metadata
            serializable_json_data = make_json_serializable_truncated(json_data)
            json.dump(serializable_json_data, f, indent=4, ensure_ascii=False)
            print(f"\n==> Output saved to: {output_file}")

        # Print execution statistics if we ran the full pipeline
        if {'final', 'direct', 'validated_final', 'validated_direct'} & set(self.output_types):
            print(f"\n## Execution Statistics for Problem {index}:")
            print(f"==>Total steps executed: {step_count}")
            print(f"==>Total execution time: {time.time() - start_time:.2f} seconds")
            
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the trusttools demo with specified parameters.")
    parser.add_argument("--llm_engine_name", default="gpt-4o-mini", help="LLM engine name.")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum tokens for LLM generation.")
    parser.add_argument("--run_baseline_only", type=bool, default=False, help="Run only the baseline (no toolbox).")
    parser.add_argument("--task", default="minitoolbench", help="Task to run.")
    parser.add_argument("--data_file", default="data/data.json", help="Data file to run.")
    parser.add_argument("--task_description", default="", help="Task description.")
    parser.add_argument(
        "--output_types",
        default="base,final,direct,validated_final,validated_direct",
        help="Comma-separated list of required outputs (base,final,direct,validated_final,validated_direct)"
    )
    parser.add_argument("--enabled_tools", default="Generalist_Solution_Generator_Tool", help="List of enabled tools.")
    parser.add_argument("--index", type=int, default=0, help="Index of the problem in the benchmark file.")
    parser.add_argument("--root_cache_dir", default="solver_cache", help="Path to solver cache directory.")
    parser.add_argument("--output_json_dir", default="results", help="Path to output JSON directory.")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to execute.")
    parser.add_argument("--max_time", type=int, default=300, help="Maximum time allowed in seconds.")
    parser.add_argument("--verbose", type=bool, default=True, help="Enable verbose output.")
    parser.add_argument("--validation_alpha", type=float, default=0.1, help="Significance level for conformal validation trimming.")
    parser.add_argument("--use_local_model", action='store_true', help="Use local LLM endpoint instead of OpenAI API.")
    parser.add_argument("--local_llm_endpoint", default=None, help="Endpoint URL for local LLM (e.g., http://localhost:8000/v1). Defaults to env var or localhost.")
    parser.add_argument("--capture_logits", action='store_true', help="Capture and save logits to file (for debugging). Validation enables capture for specific steps regardless.")
    parser.add_argument("--logits_dir", default=None, help="Directory to save captured logits JSON files.")
    return parser.parse_args()


def main(args):
    # Initialize Tools
    enabled_tools = args.enabled_tools.split(",") if args.enabled_tools else []

    # Instantiate Initializer
    initializer = Initializer(
        enabled_tools=enabled_tools,
        model_string=args.llm_engine_name
    )

    # Instantiate Planner
    planner = Planner(
        llm_engine_name=args.llm_engine_name,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        capture_logits_for_outputs=any(vt in args.output_types for vt in ['validated_final', 'validated_direct'])
    )

    # Instantiate Memory
    memory = Memory()

    # Instantiate Executor
    executor = Executor(
        llm_engine_name=args.llm_engine_name,
        root_cache_dir=args.root_cache_dir,
    )

    # Instantiate Validator
    validator = ConformalValidator(
        llm_engine_name=args.llm_engine_name
    )

    # Instantiate Solver
    solver = Solver(
        planner=planner,
        memory=memory,
        executor=executor,
        validator=validator,
        task=args.task,
        data_file=args.data_file,
        task_description=args.task_description,
        output_types=args.output_types,
        index=args.index,
        verbose=args.verbose,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        output_json_dir=args.output_json_dir,
        root_cache_dir=args.root_cache_dir,
        validation_alpha=args.validation_alpha
    )

    # Solve the task or problem
    solver.solve()

if __name__ == "__main__":
    args = parse_arguments()
    if args.capture_logits:
         print("Note: Global logit capture flag enabled. Engines configured to capture may save logs to file.")
    main(args)