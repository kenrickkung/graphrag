import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from litellm import completion, embedding
from dotenv import load_dotenv


class SimpleRAG:
    def __init__(self, working_dir, model = "gpt-4.1", embedding_model = "text-embedding-3-large"):
        self.working_dir = working_dir
        self.model = model
        self.embedding_model = embedding_model
        
        self.documents = []
        self.document_embeddings = None
        
        os.makedirs(working_dir, exist_ok=True)
        
        self.docs_file = os.path.join(working_dir, "documents.json")
        self.embeddings_file = os.path.join(working_dir, "embeddings.npy")
        
        print(f"Traditional RAG initialized with {len(self.documents)} documents")
    


    def insert(self, texts, chunk_size=1000, chunk_overlap=100): 

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        all_chunks = []
        for text in texts:
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)
    
        self.documents.extend(all_chunks)
        
        new_embeddings = self.embedding_func(all_chunks)
        
        if self.document_embeddings is None:
            self.document_embeddings = new_embeddings
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])
        
        self.save_data()
        
        print(f"{len(all_chunks)} chunks inserted. Total documents: {len(self.documents)}")
    
    def embedding_func(self, texts, batch_size=64):        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            response = embedding(
                model=self.embedding_model,
                input=batch
            )
            
            # Extract embeddings from the response
            batch_embeddings = [item['embedding'] for item in response['data']]
            all_embeddings.extend(batch_embeddings)
            
            if len(texts) > batch_size:  # Only show progress for large batches
                print(f"Processed batch {batch_num}/{total_batches} ({len(batch)} texts)")
    
        return np.array(all_embeddings, dtype=np.float32)

    def query(self, query_text, top_k = 5):
        if len(self.documents) == 0:
            return "No documents available in the knowledge base."
        
        query_embedding = self.embedding_func([query_text])
        
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_documents = [self.documents[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]
        
        context = "\n\n".join([
            f"Document {i+1} (similarity: {score:.3f}):\n{doc}" 
            for i, (doc, score) in enumerate(zip(top_documents, top_scores))
        ])
        
        response = self.generate_response(query_text, context)
        
        return response
    
    def generate_response(self, query, context):
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
        Use the context to provide accurate and informative answers. If the context doesn't contain 
        enough information to answer the question, say so clearly."""
        
        prompt = f"""
        Context:
        {context}
        
        Question: {query}
        
        Please provide a comprehensive answer based on the context above.
        """
        
        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def save_data(self):
        with open(self.docs_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        if self.document_embeddings is not None:
            np.save(self.embeddings_file, self.document_embeddings)


