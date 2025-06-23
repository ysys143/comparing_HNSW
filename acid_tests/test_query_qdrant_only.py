#!/usr/bin/env python3
"""Test only Qdrant query consistency"""
import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from scenarios.test_consistency_query import test_query_consistency
from clients.qdrant_client import QdrantDBClient

async def main():
    client = QdrantDBClient({"host": "localhost", "port": 16333})
    
    try:
        await client.connect()
        results = await test_query_consistency("qdrant", client)
        print(f"\nFinal results: {results}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())