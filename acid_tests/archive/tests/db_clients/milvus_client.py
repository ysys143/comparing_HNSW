from typing import List, Dict, Any, Optional, Tuple
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, MilvusException
)
import json
from .base_client import BaseVectorDBClient, VectorData


class MilvusDBClient(BaseVectorDBClient):
    """Milvus vector database client"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.collections = {}
    
    async def connect(self) -> bool:
        try:
            connections.connect(
                alias="default",
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 19530)
            )
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            return False
    
    async def disconnect(self) -> bool:
        try:
            connections.disconnect("default")
            self.is_connected = False
            return True
        except Exception:
            return False
    
    async def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        try:
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description=f"Collection {name} for ACID testing"
            )
            
            # Create collection
            collection = Collection(
                name=name,
                schema=schema,
                consistency_level=kwargs.get('consistency_level', 'Strong')
            )
            
            # Create index
            index_params = {
                "metric_type": "IP",  # Inner product for cosine similarity
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(
                field_name="vector",
                index_params=index_params
            )
            
            # Load collection into memory
            collection.load()
            
            self.collections[name] = collection
            return True
        except Exception as e:
            print(f"Failed to create collection: {e}")
            return False
    
    async def drop_collection(self, name: str) -> bool:
        try:
            utility.drop_collection(name)
            if name in self.collections:
                del self.collections[name]
            return True
        except Exception as e:
            print(f"Failed to drop collection: {e}")
            return False
    
    async def insert_single(self, collection: str, data: VectorData) -> bool:
        try:
            coll = self._get_collection(collection)
            
            entities = [
                [data.id],
                [data.vector],
                [json.dumps(data.metadata)]
            ]
            
            coll.insert(entities)
            coll.flush()
            return True
        except Exception as e:
            print(f"Failed to insert vector: {e}")
            return False
    
    async def insert_batch(self, collection: str, data: List[VectorData]) -> bool:
        try:
            coll = self._get_collection(collection)
            
            ids = [item.id for item in data]
            vectors = [item.vector for item in data]
            metadatas = [json.dumps(item.metadata) for item in data]
            
            entities = [ids, vectors, metadatas]
            
            coll.insert(entities)
            coll.flush()
            return True
        except Exception as e:
            print(f"Failed to insert batch: {e}")
            return False
    
    async def update_vector(self, collection: str, vector_id: str,
                          vector: Optional[List[float]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        try:
            # Milvus doesn't support direct updates, so we need to delete and re-insert
            coll = self._get_collection(collection)
            
            # Get existing data
            existing = await self.get_vector(collection, vector_id)
            if not existing:
                return False
            
            # Delete old vector
            expr = f'id == "{vector_id}"'
            coll.delete(expr)
            
            # Insert updated vector
            new_vector = vector if vector else existing.vector
            new_metadata = metadata if metadata else existing.metadata
            
            entities = [
                [vector_id],
                [new_vector],
                [json.dumps(new_metadata)]
            ]
            
            coll.insert(entities)
            coll.flush()
            return True
        except Exception as e:
            print(f"Failed to update vector: {e}")
            return False
    
    async def delete_vector(self, collection: str, vector_id: str) -> bool:
        try:
            coll = self._get_collection(collection)
            expr = f'id == "{vector_id}"'
            coll.delete(expr)
            coll.flush()
            return True
        except Exception as e:
            print(f"Failed to delete vector: {e}")
            return False
    
    async def get_vector(self, collection: str, vector_id: str) -> Optional[VectorData]:
        try:
            coll = self._get_collection(collection)
            
            expr = f'id == "{vector_id}"'
            results = coll.query(
                expr=expr,
                output_fields=["id", "vector", "metadata"]
            )
            
            if results:
                result = results[0]
                return VectorData(
                    id=result['id'],
                    vector=result['vector'],
                    metadata=json.loads(result['metadata'])
                )
            return None
        except Exception as e:
            print(f"Failed to get vector: {e}")
            return None
    
    async def search(self, collection: str, query_vector: List[float],
                    limit: int = 10, **kwargs) -> List[Tuple[str, float]]:
        try:
            coll = self._get_collection(collection)
            
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }
            
            results = coll.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=limit,
                output_fields=["id"]
            )
            
            if results:
                return [(hit.id, hit.score) for hit in results[0]]
            return []
        except Exception as e:
            print(f"Failed to search: {e}")
            return []
    
    async def count(self, collection: str) -> int:
        try:
            coll = self._get_collection(collection)
            return coll.num_entities
        except Exception as e:
            print(f"Failed to count: {e}")
            return 0
    
    async def flush(self, collection: str) -> bool:
        try:
            coll = self._get_collection(collection)
            coll.flush()
            return True
        except Exception:
            return False
    
    async def list_collections(self) -> List[str]:
        try:
            return utility.list_collections()
        except Exception:
            return []
    
    def _get_collection(self, name: str) -> Collection:
        """Get or load collection"""
        if name not in self.collections:
            self.collections[name] = Collection(name)
            self.collections[name].load()
        return self.collections[name]