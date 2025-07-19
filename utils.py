import litellm
import os
import numpy as np
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from litellm import completion, embedding
import logging
import asyncio

class Config:
    use_gemini = True
    if use_gemini:
        LLM_MODEL = "gemini/gemini-2.5-pro"
        EMBEDDING_MODEL = "gemini/text-embedding-004"
    else:
        LLM_MODEL = "gpt-4.1"
        EMBEDDING_MODEL = "text-embedding-3-large"


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

    max_retries = 3

    for attempt in range(max_retries + 1):  # +1 for the initial attempt
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
            logging.warning(f"LLM API call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
            
            # If this was the last attempt, raise the exception
            if attempt == max_retries:
                logging.error(f"All {max_retries + 1} attempts failed. Last error: {str(e)}")
                raise e
            
            await asyncio.sleep(3)
            
    raise last_exception


async def embedding_func(texts):
    max_retries = 3

    for attempt in range(max_retries + 1):  
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
            logging.warning(f"LLM API call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
            
            # If this was the last attempt, raise the exception
            if attempt == max_retries:
                logging.error(f"All {max_retries + 1} attempts failed. Last error: {str(e)}")
                raise e
            
            await asyncio.sleep(3)
            
    raise last_exception



async def initialize_rag(working_dir, vector_storage="NanoVectorDBStorage", debug=False):
    if debug:
        os.environ["LITELLM_LOG"] = "DEBUG"
        litellm._turn_on_debug()
    
    rag = LightRAG(
        working_dir=working_dir,
        vector_storage=vector_storage,
        llm_model_func=llm_model_func,
        llm_model_max_async=1,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=embedding_func,
        ),
        embedding_batch_num=16,
        embedding_func_max_async=1,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag