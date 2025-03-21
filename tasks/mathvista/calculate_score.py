import re
import io
import os
import json
import argparse
import tqdm

try:
    from Levenshtein import distance
except:
    print("Please install the 'python-Levenshtein' package using the following command:")
    print("pip install python-Levenshtein")

from tasks.utils import ResultAnalyzer

from octotools.engine.openai import ChatOpenAI

# Demos (pids = 852,  104,  824,  506,  540) from MathVista
demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(response, problem, quick_extract=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']

    if response == "":
        return ""
    
    if question_type == 'multi_choice' and response in choices:
        return response
    
    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except:
            pass

    # quick extraction
    if quick_extract:
        print("Quickly extracting answer...")
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except:
            pass

    # general extraction
    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        extraction = local_llm_engine(full_prompt)
        return extraction
    except Exception as e:
        print(e)
        print(f"Error in extracting answer for {pid}")

    return ""


def get_most_similar(prediction, choices):
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]
    # return min(choices, key=lambda choice: distance(prediction, choice))


def normalize_extracted_answer(extraction, question_data):
    """
    Normalize the extracted answer to match the answer type
    """
    choices = question_data["choices"]
    question_type = question_data["question_type"]
    answer_type = question_data["answer_type"]
    precision = question_data["precision"]

    if question_type == 'multi_choice':
        # make sure the extraction is a string
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except:
                extraction = ""
    
        # extract "A" from "(A) text"
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()
        
        options = [chr(ord('A') + i) for i in range(len(choices))]
            
        if extraction in options:
            # convert option letter to text, e.g. "A" -> "text"
            ind = options.index(extraction)
            extraction = choices[ind]
        else:
            # select the most similar option
            extraction = get_most_similar(extraction, choices)
        assert extraction in choices

    elif answer_type == 'integer':
        try:
            extraction = str(int(float(extraction)))
        except:
            extraction = None

    elif answer_type == 'float':
        try:
            extraction = str(round(float(extraction), int(precision)))
        except:
            extraction = None

    elif answer_type == 'list':
        try:
            extraction = str(extraction)
        except:
            extraction = None

    return extraction
    

def safe_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types
    """
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and score the results from the benchmark data")
    parser.add_argument("--data_file", type=str, default="data/testmini.json", help="The file containing the benchmark data")
    parser.add_argument("--result_dir", type=str, default=None, help="The directory containing the results")
    parser.add_argument("--output_file", type=str, default="final_results.json", help="The file to save the extracted results")
    parser.add_argument("--log_dir", type=str, default=None, help="The directory containing the logs")
    parser.add_argument("--response_type", 
                        type=str, 
                        default="direct_output", 
                        choices=["final_output", "direct_output", "base_response"],
                        help="The type of response to extract from the results")
    args = parser.parse_args()

    # Print the arguments
    print("#"*50)
    print(f"Arguments: {args}")
    for arg, value in args.__dict__.items():
        print(f"# {arg}: {value}")
    print("#"*50)

    # Initialize OpenAI engine
    local_llm_engine = ChatOpenAI(model_string="gpt-4o-mini", is_multimodal=False, enable_cache=True)
    print(f"\nLocal OpenAI engine {local_llm_engine.model_string} initialized.\n")

    # Load the benchmark data
    with open(args.data_file, 'r') as f:
        benchmark_data = json.load(f)
    # convert the benchmark data to a dictionary
    benchmark_data = {data["pid"]: data for data in benchmark_data}
    
    # Load the results
    results = {}
    for file in os.listdir(args.result_dir):
        if file.endswith(".json") and "output_" in file:
            with open(os.path.join(args.result_dir, file), 'r') as f:
                result = json.load(f)
        
            index = file.replace(".json", "").replace("output_", "") # "0", "1", "2", ...
            pid = str(int(index) + 1) # NOTE adjust the index to match the pid
            assert result["pid"] == benchmark_data[pid]["pid"] # check the pid

            results[pid] = benchmark_data[pid]

            assert args.response_type in result
            results[pid]["response"] = result[args.response_type]

            results[pid]["correct_answer"] = benchmark_data[pid]["answer"]

    # Extract and score the results
    correct = 0

    # for pid, question_data in results.items():
    for pid in tqdm.tqdm(results.keys(), desc="Scoring results"):
        question_data = results[pid]

        # Get response and correct answer
        response = question_data["response"]
        correct_answer = question_data["answer"]

        # Extract the precited answer text from the response
        extracted_answer = extract_answer(response, question_data)

        # Normalize the extracted answer to match the answer type
        normalized_answer = normalize_extracted_answer(extracted_answer, question_data)

        # Verify the prediction is true or false
        true_false = safe_equal(normalized_answer, correct_answer)

        # Count the number of correct predictions
        correct += 1 if true_false else 0
        
        # Save the results
        results[pid]["extracted_answer"] = extracted_answer
        results[pid]["normalized_answer"] = normalized_answer
        results[pid]["true_false"] = true_false

    acc = round(correct / len(results) * 100, 2)
    print(f"\nAccuracy: {acc}% ({correct}/{len(results)})")

    # Save the results
    output_file = os.path.join(args.result_dir, args.output_file)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_file}")

    wrong_pids = [pid for pid, data in results.items() if not data["true_false"]]
    wrong_pids = sorted(wrong_pids, key=lambda x: int(x))
    wrong_indices = [int(pid) - 1 for pid in wrong_pids]
    print(f"Wrong PIDs: {wrong_pids}")
    print(f"Wrong Indices: {wrong_indices}")

    scores = {
        "correct": correct,
        "total": len(results),
        "accuracy": acc,
        "wrong_pids": wrong_pids,
        "wrong_indices": wrong_indices
    }

    analyzer = ResultAnalyzer()

    # Calculate additional statistics if log directory is provided
    log_dir = args.log_dir or args.result_dir.replace("results", "logs")
    if os.path.exists(log_dir):

        if args.response_type == "base_response":
            print("Base response is not supported for scoring.")
            print("Exited.\n")
            exit()

         # Calculate the average time and steps
        step_stats = analyzer.calculate_time_steps(log_dir)
        print(f"\nStep stats:")
        for key, value in step_stats.items():
            print(f"- {key}: \t{value}")

        # Calculate the usage of tools 
        tool_usage = analyzer.calculate_tool_usage(args.result_dir)
        print(f"\nTool usage:")
        for tool, ratio in tool_usage.items():
            print(f"- {tool}: \t{ratio}")

        # Update the scores 
        scores.update({
            "step_stats": step_stats,
            "tool_usage": tool_usage
        })
        
    # Save the scores
    score_file = os.path.join(args.result_dir, f"final_scores_{args.response_type}.json")
    with open(score_file, 'w') as f:
        json.dump(scores, f, indent=4)
        print(f"Scores saved to {score_file}")
