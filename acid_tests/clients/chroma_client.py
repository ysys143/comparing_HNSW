from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from .base_client import BaseVectorDBClient, VectorData
import uuid


class ChromaDBClient(BaseVectorDBClient):
    """ChromaDB vector database client"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.collections = {}
    
    async def connect(self) -> bool:
        try:
            settings = Settings(
                chroma_api_impl="rest",
                chroma_server_host=self.config.get('host', 'localhost'),
                chroma_server_http_port=self.config.get('port', 8000),
                chroma_server_headers={
                    "Authorization": f"Bearer {self.config.get('auth_token', 'test-token')}"
                }
            )
            
            self.connection = chromadb.HttpClient(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 8000),
                settings=settings
            )
            
            # Test connection
            self.connection.heartbeat()
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Chroma: {e}")
            return False
    
    async def disconnect(self) -> bool:
        self.is_connected = False
        self.collections.clear()
        return True
    
    async def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        try:
            # Chroma requires an embedding function, but for testing we'll store raw vectors
            collection = self.connection.create_collection(
                name=name,
                metadata={"dimension": dimension}
            )
            self.collections[name] = collection
            return True
        except Exception as e:
            print(f"Failed to create collection: {e}")
            return False
    
    async def drop_collection(self, name: str) -> bool:
        try:
            self.connection.delete_collection(name)
            if name in self.collections:
                del self.collections[name]
            return True
        except Exception as e:
            print(f"Failed to drop collection: {e}")
            return False
    
    async def insert_single(self, collection: str, data: VectorData) -> bool:
        try:
            coll = self._get_collection(collection)
            
            coll.add(
                ids=[data.id],
                embeddings=[data.vector],
                metadatas=[data.metadata]
            )
            return True
        except Exception as e:
            print(f"Failed to insert vector: {e}")
            return False
    
    async def insert_batch(self, collection: str, data: List[VectorData]) -> bool:
        try:
            coll = self._get_collection(collection)
            
            ids = [item.id for item in data]
            embeddings = [item.vector for item in data]
            metadatas = [item.metadata for item in data]
            
            coll.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            return True
        except Exception as e:
            print(f"Failed to insert batch: {e}")
            return False
    
    async def update_vector(self, collection: str, vector_id: str,
                          vector: Optional[List[float]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        try:
            coll = self._get_collection(collection)
            
            # ChromaDB supports upsert operations
            if vector and metadata:
                coll.update(
                    ids=[vector_id],
                    embeddings=[vector],
                    metadatas=[metadata]
                )
            elif metadata:
                # Update metadata only
                coll.update(
                    ids=[vector_id],
                    metadatas=[metadata]
                )
            elif vector:
                # Update vector only - need to get existing metadata
                existing = coll.get(ids=[vector_id], include=['metadatas'])
                if existing['ids']:
                    coll.update(
                        ids=[vector_id],
                        embeddings=[vector],
                        metadatas=existing['metadatas']
                    )
            
            return True
        except Exception as e:
            print(f"Failed to update vector: {e}")
            return False
    
    async def delete_vector(self, collection: str, vector_id: str) -> bool:
        try:
            coll = self._get_collection(collection)
            coll.delete(ids=[vector_id])
            return True
        except Exception as e:
            print(f"Failed to delete vector: {e}")
            return False
    
    async def get_vector(self, collection: str, vector_id: str) -> Optional[VectorData]:
        try:
            coll = self._get_collection(collection)
            
            result = coll.get(
                ids=[vector_id],
                include=['embeddings', 'metadatas']
            )
            
            if result['ids']:
                return VectorData(
                    id=result['ids'][0],
                    vector=result['embeddings'][0] if result['embeddings'] else [],
                    metadata=result['metadatas'][0] if result['metadatas'] else {}
                )
            return None
        except Exception as e:
            print(f"Failed to get vector: {e}")
            return None
    
    async def search(self, collection: str, query_vector: List[float],
                    limit: int = 10, **kwargs) -> List[Tuple[str, float]]:
        try:
            coll = self._get_collection(collection)
            
            results = coll.query(
                query_embeddings=[query_vector],
                n_results=limit,
                include=['distances']
            )
            
            if results['ids'] and results['ids'][0]:
                # Convert distances to similarities (1 - distance for cosine)
                return [
                    (id_, 1.0 - dist) 
                    for id_, dist in zip(results['ids'][0], results['distances'][0])
                ]
            return []
        except Exception as e:
            print(f"Failed to search: {e}")
            return []
    
    async def count(self, collection: str) -> int:
        try:
            coll = self._get_collection(collection)
            return coll.count()
        except Exception as e:
            print(f"Failed to count: {e}")
            return 0
    
    async def flush(self, collection: str) -> bool:
        # ChromaDB doesn't have explicit flush
        return True
    
    async def list_collections(self) -> List[str]:
        try:
            collections = self.connection.list_collections()
            return [c.name for c in collections]
        except Exception:
            return []
    
    def _get_collection(self, name: str):
        """Get or load collection"""
        if name not in self.collections:
            self.collections[name] = self.connection.get_collection(name)
        return self.collections[name]