#!/usr/bin/env python3
"""Debug version of query consistency test"""
import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from scenarios.test_consistency_query import test_query_consistency
from clients.pgvector_client import PgVectorClient
from clients.qdrant_client import QdrantDBClient
from clients.milvus_client import MilvusDBClient
from clients.chroma_client import ChromaDBClient
from utils.data_generator import DataGenerator

async def debug_test():
    # Test vector generation first
    print("=== Testing Vector Generation ===")
    gen = DataGenerator(128)
    test_vecs = gen.generate_vectors(3, prefix="debug")
    for vec in test_vecs:
        print(f"ID: {vec.id}")
    
    print("\n=== Testing Each Database ===")
    
    # Test Qdrant specifically
    print("\n--- Testing Qdrant ---")
    qdrant_client = QdrantDBClient({"host": "localhost", "port": 16333})
    try:
        await qdrant_client.connect()
        print("Connected to Qdrant")
        
        # Create test collection
        await qdrant_client.drop_collection("test_debug")
        await qdrant_client.create_collection("test_debug", 128)
        
        # Test insertion
        vec = gen.generate_vectors(1)[0]
        print(f"Inserting vector with ID: {vec.id}")
        await qdrant_client.insert_single("test_debug", vec)
        print("✓ Insert successful")
        
        await qdrant_client.drop_collection("test_debug")
    except Exception as e:
        print(f"✗ Qdrant error: {e}")
    finally:
        await qdrant_client.disconnect()
    
    # Now run the actual test
    print("\n=== Running Full Query Consistency Test ===")
    from scenarios.test_consistency_query import main
    await main()

if __name__ == "__main__":
    asyncio.run(debug_test())