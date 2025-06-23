from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue,
    UpdateStatus, SearchRequest, Record
)
from .base_client import BaseVectorDBClient, VectorData
import uuid


class QdrantDBClient(BaseVectorDBClient):
    """Qdrant vector database client"""
    
    async def connect(self) -> bool:
        try:
            self.connection = QdrantClient(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 6333),
                grpc_port=self.config.get('grpc_port', 6334),
                prefer_grpc=self.config.get('prefer_grpc', True)
            )
            # Test connection
            self.connection.get_collections()
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}")
            return False
    
    async def disconnect(self) -> bool:
        if self.connection:
            self.connection.close()
            self.is_connected = False
        return True
    
    async def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        try:
            self.connection.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                ),
                timeout=30
            )
            return True
        except Exception as e:
            print(f"Failed to create collection: {e}")
            return False
    
    async def drop_collection(self, name: str) -> bool:
        try:
            self.connection.delete_collection(collection_name=name)
            return True
        except Exception as e:
            print(f"Failed to drop collection: {e}")
            return False
    
    async def insert_single(self, collection: str, data: VectorData) -> bool:
        try:
            point = PointStruct(
                id=data.id,
                vector=data.vector,
                payload=data.metadata
            )
            
            result = self.connection.upsert(
                collection_name=collection,
                points=[point],
                wait=True
            )
            
            return result.status == UpdateStatus.COMPLETED
        except Exception as e:
            print(f"Failed to insert vector: {e}")
            return False
    
    async def insert_batch(self, collection: str, data: List[VectorData]) -> bool:
        try:
            points = [
                PointStruct(
                    id=item.id,
                    vector=item.vector,
                    payload=item.metadata
                )
                for item in data
            ]
            
            result = self.connection.upsert(
                collection_name=collection,
                points=points,
                wait=True
            )
            
            return result.status == UpdateStatus.COMPLETED
        except Exception as e:
            print(f"Failed to insert batch: {e}")
            return False
    
    async def update_vector(self, collection: str, vector_id: str,
                          vector: Optional[List[float]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        try:
            if vector and metadata:
                # Update both vector and metadata
                point = PointStruct(
                    id=vector_id,
                    vector=vector,
                    payload=metadata
                )
                result = self.connection.upsert(
                    collection_name=collection,
                    points=[point],
                    wait=True
                )
            elif vector:
                # Update vector only
                result = self.connection.update_vectors(
                    collection_name=collection,
                    points=[{
                        "id": vector_id,
                        "vector": vector
                    }]
                )
            elif metadata:
                # Update metadata only
                result = self.connection.set_payload(
                    collection_name=collection,
                    payload=metadata,
                    points=[vector_id],
                    wait=True
                )
            else:
                return True
            
            return result.status == UpdateStatus.COMPLETED
        except Exception as e:
            print(f"Failed to update vector: {e}")
            return False
    
    async def delete_vector(self, collection: str, vector_id: str) -> bool:
        try:
            result = self.connection.delete(
                collection_name=collection,
                points_selector=[vector_id],
                wait=True
            )
            return result.status == UpdateStatus.COMPLETED
        except Exception as e:
            print(f"Failed to delete vector: {e}")
            return False
    
    async def get_vector(self, collection: str, vector_id: str) -> Optional[VectorData]:
        try:
            points = self.connection.retrieve(
                collection_name=collection,
                ids=[vector_id],
                with_vectors=True,
                with_payload=True
            )
            
            if points:
                point = points[0]
                return VectorData(
                    id=str(point.id),
                    vector=point.vector,
                    metadata=point.payload if point.payload else {}
                )
            return None
        except Exception as e:
            print(f"Failed to get vector: {e}")
            return None
    
    async def search(self, collection: str, query_vector: List[float],
                    limit: int = 10, **kwargs) -> List[Tuple[str, float]]:
        try:
            results = self.connection.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=False,
                with_vectors=False
            )
            
            return [(str(hit.id), hit.score) for hit in results]
        except Exception as e:
            print(f"Failed to search: {e}")
            return []
    
    async def count(self, collection: str) -> int:
        try:
            info = self.connection.get_collection(collection_name=collection)
            return info.points_count
        except Exception as e:
            print(f"Failed to count: {e}")
            return 0
    
    async def flush(self, collection: str) -> bool:
        # Qdrant doesn't have explicit flush, but we can wait for indexing
        try:
            # Force optimizer to run
            self.connection.update_collection(
                collection_name=collection,
                optimizer_config={
                    "indexing_threshold": 0
                }
            )
            return True
        except Exception:
            return True
    
    async def list_collections(self) -> List[str]:
        try:
            collections = self.connection.get_collections()
            return [c.name for c in collections.collections]
        except Exception:
            return []