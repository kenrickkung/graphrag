import os
import json
import time
import asyncio

from utils import initialize_rag

def insert_text(rag, file_path):
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


cls = "agriculture"
WORKING_DIR = f"{cls}"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)



def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag(WORKING_DIR))
    insert_text(rag, f"UltraDomain/{cls}_unique_contexts.json")


if __name__ == "__main__":
    main()