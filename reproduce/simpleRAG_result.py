import os
import re
import json

from dotenv import load_dotenv
from reproduce.simpleRAG import SimpleRAG


def insert_text(rag, file_path):
    with open(file_path, mode="r") as f:
        unique_contexts = json.load(f)
        
    rag.insert(unique_contexts)

def extract_queries(file_path):
    with open(file_path, "r") as f:
        data = f.read()
    

    data = data.replace("**", "")

    queries = re.findall(r"-   Question \d+: (.+)", data)

    return queries


def process_query(query_text, rag_instance):
    try:
        result = rag_instance.query(query_text)
        return {"query": query_text, "result": result}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}


def run_queries_and_save_to_json(
    queries, rag_instance, output_file, error_file
):

    with open(output_file, "a", encoding="utf-8") as result_file, open(
        error_file, "a", encoding="utf-8"
    ) as err_file:
        result_file.write("[\n")
        first_entry = True

        for query_text in queries:
            result, error = process_query(query_text, rag_instance,)
            
            if result:
                if not first_entry:
                    result_file.write(",\n")
                json.dump(result, result_file, ensure_ascii=False, indent=4)
                first_entry = False
            elif error:
                json.dump(error, err_file, ensure_ascii=False, indent=4)
                err_file.write("\n")

        result_file.write("\n]")


if __name__ == "__main__":
    load_dotenv()

    LLM_API_KEY = os.getenv("LLM_API_KEY")
    if LLM_API_KEY:
        os.environ["LLM_API_KEY"] = LLM_API_KEY
    cls = "agriculture"
    mode = "hybrid"
    WORKING_DIR = f"{cls}"

    rag = SimpleRAG(f"SimpleRAG_{cls}")
    insert_text(rag, f"UltraDomain/{cls}_unique_contexts.json")
    queries = extract_queries(f"UltraDomain/{cls}_questions.txt")
    run_queries_and_save_to_json(
        queries, rag, f"{cls}_result_simplerag.json", f"{cls}_errors_simplerag.json"
    )