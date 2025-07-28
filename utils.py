import litellm
import os
import numpy as np
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from litellm import completion, embedding
import logging
import asyncio
from transformers import AutoTokenizer, AutoModel
import torch

class Config:
    use_gemini = True
    if use_gemini:
        # LLM_MODEL = "gemini/gemini-2.5-pro"
        LLM_MODEL = "gemini/gemini-2.0-flash"
        EMBEDDING_MODEL = "gemini/text-embedding-004"
    else:
        LLM_MODEL = "gpt-4.1"
        EMBEDDING_MODEL = "text-embedding-3-large"
    
    tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # 2. Combine prompts: system prompt, history, and user prompt
    if history_messages is None:
        history_messages = []

    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"

    for msg in history_messages:
        # Each msg is expected to be a dict: {"role": "...", "content": "..."}
        combined_prompt += f"{msg['role']}: {msg['content']}\n"

    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"

    attempt = 1

    while True:
        try:
            # Call the model
            response = completion(
                model=f"{Config.LLM_MODEL}",
                messages=[{"role": "user", "content": combined_prompt}],
                **kwargs
            )
            
            # Return the response text
            return response.choices[0].message.content
            
        except Exception as e:
            last_exception = e
            
            # Log the error
            logging.warning(f"LLM API call failed (attempt {attempt}): {str(e)}")
             
            attempt += 1
            await asyncio.sleep(3 * attempt)
            
    raise last_exception


async def llm_embedding_func(texts):
    attempt = 1

    while True:  
        try:
            response = embedding(
                model=f"{Config.EMBEDDING_MODEL}",
                input=texts
            )
            embeddings = [item['embedding'] for item in response['data']]
            return np.array(embeddings)
        except Exception as e:
            last_exception = e
                
            # Log the error
            logging.warning(f"LLM API call failed (attempt {attempt}): {str(e)}")
            attempt += 1
            
            await asyncio.sleep(3)
            
    raise last_exception

async def colbert_embedding_func(texts):
    inputs = Config.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = Config.model(**inputs).last_hidden_state.numpy()
    return embeddings



async def initialize_rag(working_dir, vector_storage="NanoVectorDBStorage", debug=False):
    if debug:
        os.environ["LITELLM_LOG"] = "DEBUG"
        litellm._turn_on_debug()
    
    rag = LightRAG(
        working_dir=working_dir,
        vector_storage=vector_storage,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=20480 if vector_storage == "MuveraNanoVectorDBStorage" else 768,
            max_token_size=8192,
            func=colbert_embedding_func if vector_storage == "MuveraNanoVectorDBStorage" else llm_embedding_func,
        ),
        embedding_batch_num=16,
        embedding_func_max_async=1,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag