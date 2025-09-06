import re
import json
from litellm import completion
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os

MODEL = "gemini/gemini-2.0-flash"
# MODEL = "gemini/gemini-2.5-pro"

def evaluate_single_pair(args):
    """
    Evaluates a single query-answer pair.
    This function is designed to be run in parallel by an Executor.
    """
    i, query, answer1, answer2 = args

    sys_prompt = """
    ---Role---
    You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
    """
    
    prompt = f"""
    You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

    - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
    - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
    - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

    For each criterion, your job is to either declare a winner of the better answer for each of the categories, or assign them as a Tie if neither answer is significantly better than the other. You must also provide a rationalisation as to why you made the decision you made.

    Here is the question:
    {query}

    Here are the two answers:

    **Answer 1:**
    {answer1}

    **Answer 2:**
    {answer2}

    Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

    Output your evaluation in the following JSON format:

    {{
        "Comprehensiveness": {{
            "Winner": "[Answer 1 or Answer 2, or Tie]",
            "Explanation": "[Provide explanation here]"
        }},
        "Diversity": {{
            "Winner": "[Answer 1 or Answer 2, or Tie]",
            "Explanation": "[Provide explanation here]"
        }},
        "Empowerment": {{
            "Winner": "[Answer 1 or Answer 2, or Tie]",
            "Explanation": "[Provide explanation here]"
        }},
        "Overall Winner": {{
            "Winner": "[Answer 1 or Answer 2, or Tie]",
            "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
        }}
    }}
    """

    
    while True:
        try:
            # Use LiteLLM with Gemini
            response = completion(
                model=MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            if not response.choices[0].message.content:
                continue
            
            return {
                "request_id": f"request-{i+1}",
                "query": query,
                "answer1": answer1,
                "answer2": answer2,
                "evaluation": response.choices[0].message.content,
                "model": MODEL
            }
            
        except Exception as e:
            print(f"Error processing evaluation {i+1}: {str(e)}")
            print("Retrying...")

def batch_eval_parallel(query_file, result1_file, result2_file, output_file_path):
    print("Loading data files...")
    
    with open(query_file, "r") as f:
        data = f.read()
    queries = re.findall(r"- Question \d+: (.+)", data)

    with open(result1_file, "r") as f:
        answers1 = json.load(f)
    answers1 = [i["result"] for i in answers1]
    
    with open(result2_file, "r") as f:
        answers2 = json.load(f)
    answers2 = [i["result"] for i in answers2]
    
    print(f"Processing {len(queries)} query-answer pairs in parallel...")
    
    # Prepare the list of tasks (arguments for the worker function)
    tasks = [(i, query, answer1, answer2) for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2))]
    
    results = []
    
    # Use ThreadPoolExecutor for parallel execution
    # The number of workers can be adjusted, a common practice is to use the number of CPUs
    max_workers = os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use executor.map to apply the evaluation function to each task
        # tqdm is used to show a progress bar for the parallel execution
        for result in tqdm(executor.map(evaluate_single_pair, tasks), total=len(tasks)):
            results.append(result)
    
    with open(output_file_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results written to {output_file_path}")
    return results

if __name__ == "__main__":
    load_dotenv()
    cls = "agri3"
    query_file = f"UltraDomain/{cls}_questions_500.txt"
    result1_file = f"results/muvera_{cls}_result.json" 
    result2_file = f"results/lightrag_{cls}_result.json"
    output_file = f"results/eval/{cls}_evaluation_muvera_lightrag.json"
    
    batch_eval_parallel(query_file, result1_file, result2_file, output_file)
