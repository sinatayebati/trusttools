import os
import re
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional

from trusttools.engine.openai import ChatOpenAI, GenerationResult
from trusttools.models.memory import Memory
from trusttools.models.formatters import QueryAnalysis, NextStep, MemoryVerification

class Planner:
    def __init__(self, llm_engine_name: str, toolbox_metadata: dict = None, available_tools: List = None, capture_logits_for_outputs: bool = False):
        self.llm_engine_name = llm_engine_name
        self.llm_engine = ChatOpenAI(model_string=llm_engine_name, is_multimodal=False, capture_logits=False)
        self.llm_engine_mm = ChatOpenAI(
            model_string=llm_engine_name,
            is_multimodal=True,
            capture_logits=True,
            exclude_top_logprobs=True, 
            exclude_bytes=True
        )
        self.capture_logits_for_outputs = capture_logits_for_outputs
        print(f"Planner initialized with capture_logits_for_outputs={capture_logits_for_outputs}, using model {llm_engine_name}")
        print(f"Note: Excluding top_logprobs and bytes from captured logits for validation memory efficiency")
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []

    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        image_info = {}
        if image_path and os.path.isfile(image_path):
            image_info["image_path"] = image_path
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                image_info.update({
                    "width": width,
                    "height": height
                })
            except Exception as e:
                print(f"Error processing image file: {str(e)}")
        return image_info

    def generate_base_response(self, question: str, image: str, max_tokens: str = 4000) -> str:
        image_info = self.get_image_info(image)

        input_data = [question]
        if image_info and "image_path" in image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        self.base_response = self.llm_engine_mm(input_data, max_tokens=max_tokens)

        return self.base_response

    def analyze_query(self, question: str, image: str) -> str:
        image_info = self.get_image_info(image)
        print("image_info: ", image_info)

        query_prompt = f"""
Task: Analyze the given query with accompanying inputs and determine the skills and tools needed to address it effectively.

Available tools: {self.available_tools}

Metadata for the tools: {self.toolbox_metadata}

Image: {image_info}

Query: {question}

Instructions:
1. Carefully read and understand the query and any accompanying inputs.
2. Identify the main objectives or tasks within the query.
3. List the specific skills that would be necessary to address the query comprehensively.
4. Examine the available tools in the toolbox and determine which ones might relevant and useful for addressing the query. Make sure to consider the user metadata for each tool, including limitations and potential applications (if available).
5. Provide a brief explanation for each skill and tool you've identified, describing how it would contribute to answering the query.

Your response should include:
1. A concise summary of the query's main points and objectives, as well as content in any accompanying inputs.
2. A list of required skills, with a brief explanation for each.
3. A list of relevant tools from the toolbox, with a brief explanation of how each tool would be utilized and its potential limitations.
4. Any additional considerations that might be important for addressing the query effectively.

Please present your analysis in a clear, structured format.
"""

        input_data = [query_prompt]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        self.query_analysis = self.llm_engine_mm(input_data, response_format=QueryAnalysis)

        return str(self.query_analysis).strip()

    def extract_context_subgoal_and_tool(self, response: Optional[NextStep]) -> Tuple[Optional[str], Optional[str], Optional[str]]:

        def normalize_tool_name(tool_name: str) -> str:
            # Normalize the tool name to match the available tools
            if not tool_name: return "No tool name provided"
            for tool in self.available_tools:
                # Use case-insensitive matching and check if tool name is part of the response
                if tool.lower() in tool_name.lower():
                    return tool
            return "No matched tool given: " + tool_name

        if response is None:
            print("Error: Received None instead of NextStep object in extract_context_subgoal_and_tool.")
            return None, None, None
        
        # Check if it's an error GenerationResult (shouldn't happen with previous fix, but defensive check)
        if not isinstance(response, NextStep):
             print(f"Warning: Received unexpected type {type(response)} in extract_context_subgoal_and_tool. Expected NextStep.")
             # Attempt to extract info if it looks like the old error format, otherwise fail
             if hasattr(response, 'error'):
                  print(f"  Contained error: {getattr(response, 'error')}")
             return None, None, None

        try:
            context = getattr(response, 'context', '').strip()
            sub_goal = getattr(response, 'sub_goal', '').strip()
            tool_name_raw = getattr(response, 'tool_name', '').strip()
            tool_name = normalize_tool_name(tool_name_raw)
            return context, sub_goal, tool_name
        except Exception as e:
            print(f"Error extracting context, sub-goal, and tool name: {str(e)}")
            return None, None, None
        
    def generate_next_step(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int, max_step_count: int) -> NextStep:
        prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the given query based on the provided analysis, available tools, and previous steps taken.

Context:
Query: {question}
Image: {image}
Query Analysis: {query_analysis}

Available Tools:
{self.available_tools}

Tool Metadata:
{self.toolbox_metadata}

Previous Steps and Their Results:
{memory.get_actions()}

Current Step: {step_count} in {max_step_count} steps
Remaining Steps: {max_step_count - step_count}

Instructions:
1. Analyze the context thoroughly, including the query, its analysis, any image, available tools and their metadata, and previous steps taken.

2. Determine the most appropriate next step by considering:
   - Key objectives from the query analysis
   - Capabilities of available tools
   - Logical progression of problem-solving
   - Outcomes from previous steps
   - Current step count and remaining steps

3. Select ONE tool best suited for the next step, keeping in mind the limited number of remaining steps.

4. Formulate a specific, achievable sub-goal for the selected tool that maximizes progress towards answering the query.

Output Format:
<justification>: detailed explanation of why the selected tool is the best choice for the next step, considering the context and previous outcomes.
<context>: MUST include ALL necessary information for the tool to function, structured as follows:
    * Relevant data from previous steps
    * File names or paths created or used in previous steps (list EACH ONE individually)
    * Variable names and their values from previous steps' results
    * Any other context-specific information required by the tool
<sub_goal>: a specific, achievable objective for the tool, based on its metadata and previous outcomes. It MUST contain any involved data, file names, and variables from Previous Steps and Their Results that the tool can act upon.
<tool_name>: MUST be the exact name of a tool from the available tools list.

Rules:
- Select only ONE tool for this step.
- The sub-goal MUST directly address the query and be achievable by the selected tool.
- The Context section MUST include ALL necessary information for the tool to function, including ALL relevant file paths, data, and variables from previous steps.
- The tool name MUST exactly match one from the available tools list: {self.available_tools}.
- Avoid redundancy by considering previous steps and building on prior results.

Example (do not copy, use only as reference):
<justification>: [Your detailed explanation here]
<context>: Image path: "example/image.jpg", Previous detection results: [list of objects]
<sub_goal>: Detect and count the number of specific objects in the image "example/image.jpg"
<tool_name>: Object_Detector_Tool
"""
        next_step = self.llm_engine(prompt_generate_next_step, response_format=NextStep)
        return next_step

    def verificate_context(self, question: str, image: str, query_analysis: str, memory: Memory) -> MemoryVerification:
        image_info = self.get_image_info(image)

        prompt_memory_verification = f"""
Task: Thoroughly evaluate the completeness and accuracy of the memory for fulfilling the given query, considering the potential need for additional tool usage.

Context:
Query: {question}
Image: {image_info}
Available Tools: {self.available_tools}
Toolbox Metadata: {self.toolbox_metadata}
Initial Analysis: {query_analysis}
Memory (tools used and results): {memory.get_actions()}

Detailed Instructions:
1. Carefully analyze the query, initial analysis, and image (if provided):
   - Identify the main objectives of the query.
   - Note any specific requirements or constraints mentioned.
   - If an image is provided, consider its relevance and what information it contributes.

2. Review the available tools and their metadata:
   - Understand the capabilities and limitations and best practices of each tool.
   - Consider how each tool might be applicable to the query.

3. Examine the memory content in detail:
   - Review each tool used and its execution results.
   - Assess how well each tool's output contributes to answering the query.

4. Critical Evaluation (address each point explicitly):
   a) Completeness: Does the memory fully address all aspects of the query?
      - Identify any parts of the query that remain unanswered.
      - Consider if all relevant information has been extracted from the image (if applicable).

   b) Unused Tools: Are there any unused tools that could provide additional relevant information?
      - Specify which unused tools might be helpful and why.

   c) Inconsistencies: Are there any contradictions or conflicts in the information provided?
      - If yes, explain the inconsistencies and suggest how they might be resolved.

   d) Verification Needs: Is there any information that requires further verification due to tool limitations?
      - Identify specific pieces of information that need verification and explain why.

   e) Ambiguities: Are there any unclear or ambiguous results that could be clarified by using another tool?
      - Point out specific ambiguities and suggest which tools could help clarify them.

5. Final Determination:
   Based on your thorough analysis, decide if the memory is complete and accurate enough to generate the final output, or if additional tool usage is necessary.

Response Format:
<analysis>: Provide a detailed analysis of why the memory is sufficient. Reference specific information from the memory and explain its relevance to each aspect of the task. Address how each main point of the query has been satisfied.
<stop_signal>: Whether to stop the problem solving process and proceed to generating the final output.
    * "True": if the memory is sufficient for addressing the query to proceed and no additional available tools need to be used. If ONLY manual verification without tools is needed, choose "True".
    * "False": if the memory is insufficient and needs more information from additional tool usage.
"""

        input_data = [prompt_memory_verification]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        stop_verification = self.llm_engine_mm(input_data, response_format=MemoryVerification)

        return stop_verification

    def extract_conclusion(self, response: Optional[MemoryVerification]) -> str:
        if response is None:
             print("Error: Received None instead of MemoryVerification object in extract_conclusion.")
             return 'CONTINUE' # Default to continue if verification failed

        # Check if it's an error GenerationResult (defensive check)
        if not isinstance(response, MemoryVerification):
            print(f"Warning: Received unexpected type {type(response)} in extract_conclusion. Expected MemoryVerification.")
            if hasattr(response, 'error'):
                print(f"  Contained error: {getattr(response, 'error')}")
            return 'CONTINUE' # Default to continue

        try:
            if response.stop_signal:
                return 'STOP'
            else:
                return 'CONTINUE'
        except AttributeError:
            print("Error: MemoryVerification object missing 'stop_signal' attribute.")
            return 'CONTINUE' # Default to continue on unexpected attribute error

    def generate_final_output(self, question: str, image: str, memory: Memory, step_count: int) -> str:
        image_info = self.get_image_info(image)

        prompt_generate_final_output = f"""
Task: Generate the final output based on the query, image, and tools used in the process.

Context:
Query: {question}
Image: {image_info}
Actions Taken:
{memory.get_actions()}

Instructions:
1. Review the query, image, and all actions taken during the process.
2. Consider the results obtained from each tool execution.
3. Incorporate the relevant information from the memory to generate the step-by-step final output.
4. The final output should be consistent and coherent using the results from the tools.

Output Structure:
Your response should be well-organized and include the following sections:

1. Summary:
   - Provide a brief overview of the query and the main findings.

2. Detailed Analysis:
   - Break down the process of answering the query step-by-step.
   - For each step, mention the tool used, its purpose, and the key results obtained.
   - Explain how each step contributed to addressing the query.

3. Key Findings:
   - List the most important discoveries or insights gained from the analysis.
   - Highlight any unexpected or particularly interesting results.

4. Answer to the Query:
   - Directly address the original question with a clear and concise answer.
   - If the query has multiple parts, ensure each part is answered separately.

5. Additional Insights (if applicable):
   - Provide any relevant information or insights that go beyond the direct answer to the query.
   - Discuss any limitations or areas of uncertainty in the analysis.

6. Conclusion:
   - Summarize the main points and reinforce the answer to the query.
   - If appropriate, suggest potential next steps or areas for further investigation.
"""

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")
                return f"Error generating final output: Could not read image {str(e)}"

        try:
            # Generate with potential logit capture
            generation_result: GenerationResult = self.llm_engine_mm.generate(
                input_data,
                capture_logits=self.capture_logits_for_outputs
            )

            # Check if generation_result is None (indicates a severe error)
            if generation_result is None:
                error_msg = "Critical error: generation_result is None. API call likely failed."
                print(f"Error generating final output: {error_msg}")
                # Create a fallback response from memory
                final_output_text = self._create_fallback_output_from_memory(question, memory)
                logprob_content = None
            else:
                final_output_text = generation_result.text
                logprob_content = generation_result.logprob_content

                # Handle potential generation errors (like rate limits)
                if final_output_text is None:
                    error_msg = generation_result.error or "Unknown generation error"
                    print(f"Error generating final output: {error_msg}")
                    # Create a fallback response from memory
                    final_output_text = self._create_fallback_output_from_memory(question, memory)
                    logprob_content = None
        except Exception as e:
            print(f"Unexpected error in generate_final_output: {str(e)}")
            # Create a fallback response from memory
            final_output_text = self._create_fallback_output_from_memory(question, memory)
            logprob_content = None

        memory.add_action(
            step_count=step_count,
            tool_name="FinalOutputGenerator",
            sub_goal="Generate comprehensive final answer",
            command=None,
            result=final_output_text,
            logprob_content=logprob_content
        )

        return final_output_text

    def generate_direct_output(self, question: str, image: str, memory: Memory, step_count: int) -> str:
        image_info = self.get_image_info(image)

        prompt_generate_direct_output = f"""
Context:
Query: {question}
Image: {image_info}
Initial Analysis:
{self.query_analysis if hasattr(self, 'query_analysis') else 'N/A'}
Actions Taken:
{memory.get_actions()}

Please generate the concise output based on the query, image information, initial analysis, and actions taken. Break down the process into clear, logical, and coherent steps. Conclude with a precise and direct answer to the query.

Answer:
"""
        input_data = [prompt_generate_direct_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")
                return f"Error generating direct output: Could not read image {str(e)}"

        try:
            # Generate with potential logit capture
            generation_result: GenerationResult = self.llm_engine_mm.generate(
                input_data,
                capture_logits=self.capture_logits_for_outputs
            )

            # Check if generation_result is None (indicates a severe error)
            if generation_result is None:
                error_msg = "Critical error: generation_result is None. API call likely failed."
                print(f"Error generating direct output: {error_msg}")
                # Create a fallback response from memory
                direct_output_text = self._create_fallback_output_from_memory(question, memory)
                logprob_content = None
            else:
                direct_output_text = generation_result.text
                logprob_content = generation_result.logprob_content

                # Handle potential generation errors (like rate limits)
                if direct_output_text is None:
                    error_msg = generation_result.error or "Unknown generation error"
                    print(f"Error generating direct output: {error_msg}")
                    # Create a fallback response from memory
                    direct_output_text = self._create_fallback_output_from_memory(question, memory)
                    logprob_content = None
        except Exception as e:
            print(f"Unexpected error in generate_direct_output: {str(e)}")
            # Create a fallback response from memory
            direct_output_text = self._create_fallback_output_from_memory(question, memory)
            logprob_content = None

        memory.add_action(
            step_count=step_count,
            tool_name="DirectOutputGenerator",
            sub_goal="Generate concise direct answer",
            command=None,
            result=direct_output_text,
            logprob_content=logprob_content
        )

        return direct_output_text
        
    def _create_fallback_output_from_memory(self, question: str, memory: Memory) -> str:
        """
        Creates a fallback output from memory if the API call fails.
        This ensures we still return something useful to the user.
        """
        print("Creating fallback output from memory...")
        actions = memory.get_actions()
        
        # Start with a standard header
        fallback_output = f"[Fallback Response] Answer to: '{question}'\n\n"
        
        # Extract results from memory
        for step_name, action in actions.items():
            if action.get('result'):
                result = action['result']
                # Handle different result formats
                if isinstance(result, dict) and 'text' in result:
                    text_content = result['text']
                    fallback_output += f"Based on {action.get('tool_name', 'analysis')}: {text_content}\n\n"
                elif isinstance(result, str):
                    fallback_output += f"Based on {action.get('tool_name', 'analysis')}: {result}\n\n"
        
        # If we couldn't extract anything useful, provide a generic message
        if len(fallback_output.strip().split('\n')) <= 1:
            fallback_output += "Unable to generate a direct answer due to system limitations. Please try again later."
            
        return fallback_output
    