import re
import json
from litellm import completion
from dotenv import load_dotenv
from tqdm import tqdm

MODEL = "gemini/gemini-2.5-pro"


def batch_eval(query_file, result1_file, result2_file, output_file_path):
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
    
    print(f"Processing {len(queries)} query-answer pairs...")
    
    results = []
    
    for i, (query, answer1, answer2) in tqdm(enumerate(zip(queries, answers1, answers2))):
        
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """
        
        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

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
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """
        
        try:
            # Use LiteLLM with Gemini
            response = completion(
                model=MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            
            evaluation_result = {
                "request_id": f"request-{i+1}",
                "query": query,
                "answer1": answer1,
                "answer2": answer2,
                "evaluation": response.choices[0].message.content,
                "model": MODEL
            }
            
            results.append(evaluation_result)
            
        except Exception as e:
            print(f"Error processing evaluation {i+1}: {str(e)}")
            error_result = {
                "request_id": f"request-{i+1}",
                "query": query,
                "answer1": answer1,
                "answer2": answer2,
                "evaluation": f"Error: {str(e)}",
                "model": MODEL
            }
            results.append(error_result)
    
    with open(output_file_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results written to {output_file_path}")
    return results

if __name__ == "__main__":
    load_dotenv()
    cls = "small_agri"
    query_file = f"UltraDomain/{cls}_questions.txt"
    result1_file = f"results/{cls}_result_simplerag.json" 
    result2_file = f"results/{cls}_result.json"
    output_file = f"results/{cls}_evaluation_simple_lightrag.json"
    
    batch_eval(query_file, result1_file, result2_file, output_file)
