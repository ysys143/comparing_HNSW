from .base_client import BaseVectorDBClient
from .pgvector_client import PgVectorClient
from .qdrant_client import QdrantDBClient
from .milvus_client import MilvusDBClient
from .chroma_client import ChromaDBClient


def get_client(db_name: str, config: dict) -> BaseVectorDBClient:
    """Get database client by name"""
    clients = {
        'pgvector': PgVectorClient,
        'qdrant': QdrantDBClient,
        'milvus': MilvusDBClient,
        'chroma': ChromaDBClient
    }
    
    if db_name not in clients:
        raise ValueError(f"Unknown database: {db_name}")
    
    return clients[db_name](config)


__all__ = [
    'BaseVectorDBClient',
    'PgVectorClient',
    'QdrantDBClient',
    'MilvusDBClient',
    'ChromaDBClient',
    'get_client'
]