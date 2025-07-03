import json
import os
import time
import asyncio
import re
from dotenv import load_dotenv
import litellm

from lightrag import QueryParam
from lightrag.lightrag import LightRAG
from lightrag.utils import always_get_an_event_loop, setup_logger

from utils import initialize_rag

def insert_text(rag: LightRAG, file_path):
    with open(file_path, mode="r") as f:
        unique_contexts = json.load(f)

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")


def extract_queries(file_path):
    with open(file_path, "r") as f:
        data = f.read()
    

    data = data.replace("**", "")

    queries = re.findall(r"-   Question \d+: (.+)", data)

    return queries

async def process_query(query_text, rag_instance: LightRAG, query_param):
    try:
        result = await rag_instance.aquery(query_text, param=query_param)
        return {"query": query_text, "result": result}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}


def run_queries_and_save_to_json(
    queries, rag_instance, query_param, output_file, error_file
):
    loop = always_get_an_event_loop()

    with open(output_file, "a", encoding="utf-8") as result_file, open(
        error_file, "a", encoding="utf-8"
    ) as err_file:
        result_file.write("[\n")
        first_entry = True

        for query_text in queries:
            result, error = loop.run_until_complete(
                process_query(query_text, rag_instance, query_param)
            )

            if result:
                if not first_entry:
                    result_file.write(",\n")
                json.dump(result, result_file, ensure_ascii=False, indent=4)
                first_entry = False
            elif error:
                json.dump(error, err_file, ensure_ascii=False, indent=4)
                err_file.write("\n")

        result_file.write("\n]")

    
def main():
    load_dotenv()

    cls = "agriculture"
    mode = "late-interaction"

    query_param = QueryParam(mode=mode)
    setup_logger(f"{mode}_{cls}", level="DEBUG", log_file_path=f"colbert_{cls}/{mode}_{cls}.log")
    rag = asyncio.run(initialize_rag(f"colbert_{cls}", "ColbertVectorDBStorage", debug=True))
    insert_text(rag, f"UltraDomain/{cls}_unique_contexts.json")
    queries = extract_queries(f"UltraDomain/{cls}_questions.txt")
    run_queries_and_save_to_json(
        queries, rag, query_param, f"colbert_{cls}_result.json", f"colbert_{cls}_errors.json"
    )


if __name__ == "__main__":
    main()