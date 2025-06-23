#!/usr/bin/env python3
"""Simple test for Qdrant to debug UUID issue"""
import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.data_generator import DataGenerator
from clients.qdrant_client import QdrantDBClient

async def test_qdrant():
    client = QdrantDBClient({"host": "localhost", "port": 16333})
    
    try:
        await client.connect()
        print("Connected to Qdrant")
        
        # Create collection
        collection_name = "test_uuid_debug"
        await client.drop_collection(collection_name)
        await client.create_collection(collection_name, 128)
        
        # Generate vectors
        gen = DataGenerator(128)
        vectors = gen.generate_vectors(5, prefix="test")
        
        print("\nGenerated vectors:")
        for vec in vectors:
            print(f"  ID: {vec.id} (type: {type(vec.id)})")
        
        # Insert one by one to see which fails
        for i, vec in enumerate(vectors):
            try:
                await client.insert_single(collection_name, vec)
                print(f"✓ Inserted vector {i}: {vec.id}")
            except Exception as e:
                print(f"✗ Failed vector {i}: {vec.id}")
                print(f"  Error: {e}")
        
        # Clean up
        await client.drop_collection(collection_name)
        
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_qdrant())