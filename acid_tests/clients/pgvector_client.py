import asyncio
import asyncpg
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from .base_client import BaseVectorDBClient, VectorData


class PgVectorClient(BaseVectorDBClient):
    """PostgreSQL with pgvector extension client"""
    
    async def connect(self) -> bool:
        try:
            # Handle both connection string and individual parameters
            if 'connection_string' in self.config:
                self.connection = await asyncpg.connect(self.config['connection_string'])
            else:
                self.connection = await asyncpg.connect(
                    host=self.config.get('host', 'localhost'),
                    port=self.config.get('port', 5432),
                    user=self.config.get('user', 'postgres'),
                    password=self.config.get('password', 'postgres'),
                    database=self.config.get('database', 'vectordb')
                )
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    async def disconnect(self) -> bool:
        if self.connection:
            await self.connection.close()
            self.is_connected = False
        return True
    
    async def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        try:
            # Create table with vector column
            await self.connection.execute(f"""
                CREATE TABLE IF NOT EXISTS {name} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    vector vector({dimension}),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index
            index_type = kwargs.get('index_type', 'ivfflat')
            if index_type == 'ivfflat':
                await self.connection.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{name}_vector 
                    ON {name} USING ivfflat (vector vector_cosine_ops)
                    WITH (lists = 100)
                """)
            
            return True
        except Exception as e:
            print(f"Failed to create collection: {e}")
            return False
    
    async def drop_collection(self, name: str) -> bool:
        try:
            await self.connection.execute(f"DROP TABLE IF EXISTS {name} CASCADE")
            return True
        except Exception as e:
            print(f"Failed to drop collection: {e}")
            return False
    
    async def insert_single(self, collection: str, data: VectorData) -> bool:
        try:
            vector_str = '[' + ','.join(map(str, data.vector)) + ']'
            metadata_json = json.dumps(data.metadata)
            
            await self.connection.execute(f"""
                INSERT INTO {collection} (id, vector, metadata)
                VALUES ($1::uuid, $2::vector, $3::jsonb)
            """, uuid.UUID(data.id), vector_str, metadata_json)
            
            return True
        except Exception as e:
            print(f"Failed to insert vector: {e}")
            return False
    
    async def insert_batch(self, collection: str, data: List[VectorData]) -> bool:
        try:
            # Use transaction for atomicity
            async with self.connection.transaction():
                for item in data:
                    vector_str = '[' + ','.join(map(str, item.vector)) + ']'
                    metadata_json = json.dumps(item.metadata)
                    
                    await self.connection.execute(f"""
                        INSERT INTO {collection} (id, vector, metadata)
                        VALUES ($1::uuid, $2::vector, $3::jsonb)
                    """, uuid.UUID(item.id), vector_str, metadata_json)
            
            return True
        except Exception as e:
            print(f"Failed to insert batch: {e}")
            return False
    
    async def update_vector(self, collection: str, vector_id: str,
                          vector: Optional[List[float]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        try:
            if vector and metadata:
                vector_str = '[' + ','.join(map(str, vector)) + ']'
                metadata_json = json.dumps(metadata)
                await self.connection.execute(f"""
                    UPDATE {collection} 
                    SET vector = $2::vector, metadata = $3::jsonb, updated_at = CURRENT_TIMESTAMP
                    WHERE id = $1::uuid
                """, uuid.UUID(vector_id), vector_str, metadata_json)
            elif vector:
                vector_str = '[' + ','.join(map(str, vector)) + ']'
                await self.connection.execute(f"""
                    UPDATE {collection} 
                    SET vector = $2::vector, updated_at = CURRENT_TIMESTAMP
                    WHERE id = $1::uuid
                """, uuid.UUID(vector_id), vector_str)
            elif metadata:
                metadata_json = json.dumps(metadata)
                await self.connection.execute(f"""
                    UPDATE {collection} 
                    SET metadata = $2::jsonb, updated_at = CURRENT_TIMESTAMP
                    WHERE id = $1::uuid
                """, uuid.UUID(vector_id), metadata_json)
            
            return True
        except Exception as e:
            print(f"Failed to update vector: {e}")
            return False
    
    async def delete_vector(self, collection: str, vector_id: str) -> bool:
        try:
            await self.connection.execute(f"""
                DELETE FROM {collection} WHERE id = $1::uuid
            """, uuid.UUID(vector_id))
            return True
        except Exception as e:
            print(f"Failed to delete vector: {e}")
            return False
    
    async def get_vector(self, collection: str, vector_id: str) -> Optional[VectorData]:
        try:
            row = await self.connection.fetchrow(f"""
                SELECT id, vector, metadata 
                FROM {collection} 
                WHERE id = $1::uuid
            """, uuid.UUID(vector_id))
            
            if row:
                return VectorData(
                    id=str(row['id']),
                    vector=list(row['vector']),
                    metadata=dict(row['metadata']) if row['metadata'] else {}
                )
            return None
        except Exception as e:
            print(f"Failed to get vector: {e}")
            return None
    
    async def search(self, collection: str, query_vector: List[float],
                    limit: int = 10, **kwargs) -> List[Tuple[str, float]]:
        try:
            vector_str = '[' + ','.join(map(str, query_vector)) + ']'
            rows = await self.connection.fetch(f"""
                SELECT id, 1 - (vector <=> $1::vector) as similarity
                FROM {collection}
                ORDER BY vector <=> $1::vector
                LIMIT $2
            """, vector_str, limit)
            
            return [(str(row['id']), row['similarity']) for row in rows]
        except Exception as e:
            print(f"Failed to search: {e}")
            return []
    
    async def count(self, collection: str) -> int:
        try:
            row = await self.connection.fetchrow(f"SELECT COUNT(*) FROM {collection}")
            return row['count']
        except Exception as e:
            print(f"Failed to count: {e}")
            return 0
    
    async def flush(self, collection: str) -> bool:
        # PostgreSQL commits automatically, but we can force a checkpoint
        try:
            await self.connection.execute("CHECKPOINT")
            return True
        except Exception:
            return True  # Not critical if checkpoint fails
    
    async def list_collections(self) -> List[str]:
        try:
            rows = await self.connection.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
            """)
            return [row['table_name'] for row in rows]
        except Exception:
            return []