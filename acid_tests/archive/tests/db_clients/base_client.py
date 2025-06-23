from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import time
import asyncio
from dataclasses import dataclass


@dataclass
class VectorData:
    id: str
    vector: List[float]
    metadata: Dict[str, Any]


class BaseVectorDBClient(ABC):
    """Abstract base class for vector database clients"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
        self.is_connected = False
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the database"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to the database"""
        pass
    
    @abstractmethod
    async def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """Create a new collection/table"""
        pass
    
    @abstractmethod
    async def drop_collection(self, name: str) -> bool:
        """Drop a collection/table"""
        pass
    
    @abstractmethod
    async def insert_single(self, collection: str, data: VectorData) -> bool:
        """Insert a single vector"""
        pass
    
    @abstractmethod
    async def insert_batch(self, collection: str, data: List[VectorData]) -> bool:
        """Insert multiple vectors in a batch"""
        pass
    
    @abstractmethod
    async def update_vector(self, collection: str, vector_id: str, 
                          vector: Optional[List[float]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update a vector and/or its metadata"""
        pass
    
    @abstractmethod
    async def delete_vector(self, collection: str, vector_id: str) -> bool:
        """Delete a vector by ID"""
        pass
    
    @abstractmethod
    async def get_vector(self, collection: str, vector_id: str) -> Optional[VectorData]:
        """Retrieve a vector by ID"""
        pass
    
    @abstractmethod
    async def search(self, collection: str, query_vector: List[float], 
                    limit: int = 10, **kwargs) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    async def count(self, collection: str) -> int:
        """Count vectors in a collection"""
        pass
    
    @abstractmethod
    async def flush(self, collection: str) -> bool:
        """Force flush/commit any pending operations"""
        pass
    
    async def health_check(self) -> bool:
        """Check if the database is healthy and responsive"""
        try:
            # Simple connectivity test
            if not self.is_connected:
                await self.connect()
            # Try a simple operation
            collections = await self.list_collections()
            return True
        except Exception:
            return False
    
    async def list_collections(self) -> List[str]:
        """List all collections/tables"""
        return []
    
    async def wait_for_ready(self, timeout: int = 60, interval: int = 1) -> bool:
        """Wait for the database to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await self.health_check():
                return True
            await asyncio.sleep(interval)
        return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            'config': self.config,
            'is_connected': self.is_connected,
            'type': self.__class__.__name__
        }