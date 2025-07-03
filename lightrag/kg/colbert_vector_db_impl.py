import os
from typing import Any, final
from dataclasses import dataclass
import time
import shutil

from lightrag.utils import (
    logger,
    compute_mdhash_id,
)
import pipmaster as pm
from lightrag.base import BaseVectorStorage

if not pm.is_installed("ragatouille"):
    pm.install("ragatouille")

from ragatouille import RAGPretrainedModel
from .shared_storage import (
    get_storage_lock,
    get_update_flag,
    set_all_update_flags,
)


@final
@dataclass
class ColbertVectorDBStorage(BaseVectorStorage):
    def __post_init__(self):
        # Initialize basic attributes
        self._client = None
        self._storage_lock = None
        self.storage_updated = None

        # Use global config value if specified, otherwise use default
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        # RAGatouille uses index directories, not single files
        self._index_name = f"vdb_{self.namespace}"
        self._index_path = os.path.join(
            self.global_config["working_dir"],"colbert/indexes", self._index_name
        )

        # Initialize RAGatouille model
        self.client = self._get_client()

    async def initialize(self):
        """Initialize storage data"""
        # Get the update flag for cross-process update notification
        self.storage_updated = await get_update_flag(self.namespace)
        # Get the storage lock for use in other methods
        self._storage_lock = get_storage_lock(enable_logging=False)
        
        # Initialize or load existing index
        # await self._get_client()

    def _get_client(self):
        """Check if the storage should be reloaded"""
    
        if os.path.exists(self._index_path):
            client = RAGPretrainedModel.from_index(
                self._index_path
            )
        else:
            client = RAGPretrainedModel.from_pretrained(
                "colbert-ir/colbertv2.0",
                index_root=self.global_config["working_dir"]
            )

        return client

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """
        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """

        logger.debug(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        current_time = int(time.time())
        
        # Prepare documents and metadata for upsertion
        documents = []
        metadatas = []
        ids = []

        for doc_id, doc_data in data.items():
            content = doc_data.get("content")
            if content is None:
                logger.warning(f"Content not found for ID {doc_id}, skipping.")
                continue

            documents.append(content)
            
            # Prepare metadata
            metadata = {
                "__id__": doc_id,
                "__created_at__": current_time,
                **{k: v for k, v in doc_data.items() if k in self.meta_fields and k != "content"}
            }
            metadatas.append(metadata)
            ids.append(doc_id)

        if not documents:
            logger.warning("No valid documents to upsert.")
            return
        
            
        # RAGatouille's index method
        if not os.path.exists(self._index_path):
            # Create new index
            self.client.index(
                collection=documents,
                document_ids=ids,
                document_metadatas=metadatas,
                index_name=self._index_name,
                max_document_length=512,
            )
        else:
            # Add to existing index
            self.client.add_to_index(
                new_collection=documents,
                new_document_ids=ids,
                new_document_metadatas=metadatas,
                index_name=self._index_name,
            )
        
        logger.debug(f"Upserted {len(documents)} documents to {self.namespace}")

    async def query(
        self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Query the vector storage"""
        
        # RAGatouille search
        results = self.client.search(
            query=query,
            k=top_k,
            index_name=self._index_name,
        )

        # Convert results to expected format
        formatted_results = []
        for result in results:
            formatted_result = {
                "id": result.get("document_id", result.get("id")),
                "content": result.get("content", result.get("document")),
                "distance": result.get("score", 0.0),  # RAGatouille uses 'score'
                "created_at": result.get("document_metadata", {}).get("__created_at__"),
            }
            
            # Add metadata
            metadata = result.get("document_metadata", {})
            for key, value in metadata.items():
                if not key.startswith("__"):
                    formatted_result[key] = value
                    
            formatted_results.append(formatted_result)

        # Filter by IDs if specified
        if ids:
            formatted_results = [r for r in formatted_results if r["id"] in ids]

        return formatted_results

    async def delete(self, ids: list[str]):
        """Delete vectors with specified IDs

        Note: RAGatouille doesn't support direct deletion by ID.
        This is a limitation of the ColBERT architecture.
        A workaround would be to rebuild the index without the specified documents.
        """
        logger.warning(
            f"RAGatouille/ColBERT doesn't support direct deletion of documents by ID. "
            f"Attempted to delete {len(ids)} documents from {self.namespace}. "
            f"Consider rebuilding the index without these documents."
        )

    async def delete_entity(self, entity_name: str) -> None:
        """Delete entity (limited support in RAGatouille)"""
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        logger.warning(
            f"RAGatouille doesn't support direct deletion. "
            f"Entity {entity_name} (ID: {entity_id}) deletion logged but not executed."
        )

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete entity relations (limited support in RAGatouille)"""
        logger.warning(
            f"RAGatouille doesn't support direct deletion. "
            f"Relations for entity {entity_name} deletion logged but not executed."
        )

    async def index_done_callback(self) -> bool:
        """Save/persist the index"""
        async with self._storage_lock:
            try:
                # RAGatouille automatically persists indexes to disk
                # The index is already saved when created/updated
                
                # Notify other processes that data has been updated
                await set_all_update_flags(self.namespace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False
                
                logger.debug(f"Index persistence completed for {self.namespace}")
                return True
                
            except Exception as e:
                logger.error(f"Error in index_done_callback for {self.namespace}: {e}")
                return False

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get document by ID (requires search since RAGatouille doesn't have direct ID lookup)"""
        try:
            # Since RAGatouille doesn't have direct ID lookup, we need to search
            # This is not efficient but necessary given ColBERT's architecture
            client = await self._get_client()
            
            # We can't directly get by ID, so this is a limitation
            # One approach is to search with a unique term from the document
            # But without knowing the content, this is challenging
            
            logger.warning(
                f"get_by_id is not efficiently supported in RAGatouille. "
                f"Consider using query() instead for ID: {id}"
            )
            return None
            
        except Exception as e:
            logger.error(f"Error getting document by ID {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple documents by IDs (limited support)"""
        if not ids:
            return []
            
        logger.warning(
            f"get_by_ids is not efficiently supported in RAGatouille. "
            f"Consider using query() instead for {len(ids)} IDs"
        )
        
        # Return empty list as direct ID lookup is not supported
        return []

    async def drop(self) -> dict[str, str]:
        """Drop all vector data from storage and clean up resources"""
        try:
            async with self._storage_lock:
                # Remove index directory if it exists
                if os.path.exists(self._index_path):
                    shutil.rmtree(self._index_path)
                    logger.info(f"Removed index directory: {self._index_path}")

                # Reinitialize the client
                self._client = self._model

                # Notify other processes that data has been updated
                await set_all_update_flags(self.namespace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False

                logger.info(
                    f"Process {os.getpid()} dropped {self.namespace} (path: {self._index_path})"
                )
                
            return {"status": "success", "message": "data dropped"}
            
        except Exception as e:
            logger.error(f"Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}

    @property
    async def client_storage(self):
        """Get client storage (RAGatouille doesn't expose internal storage like NanoVectorDB)"""
        logger.warning("RAGatouille doesn't expose internal storage like NanoVectorDB")
        return None