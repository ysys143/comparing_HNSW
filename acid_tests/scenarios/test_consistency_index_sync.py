"""
Stage 2: Consistency Tests - Index-Data Synchronization
Assesses whether newly inserted vectors are immediately reflected in search results under high insert load
"""
import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_generator import DataGenerator
from clients.base_client import VectorData
from clients.pgvector_client import PgVectorClient
from clients.qdrant_client import QdrantDBClient  
from clients.milvus_client import MilvusDBClient
from clients.chroma_client import ChromaDBClient

async def test_index_sync(db_name: str, client: Any) -> Dict[str, Any]:
    """Test index-data synchronization under load"""
    print(f"\n{'='*60}")
    print(f"Testing Index-Data Synchronization - {db_name}")
    print(f"{'='*60}")
    
    collection_name = f"test_index_sync_{db_name}"
    dimension = 128  # Smaller dimension for faster testing
    results = {
        "database": db_name,
        "test_type": "index_data_synchronization",
        "tests": {}
    }
    
    # Initialize collection
    try:
        if hasattr(client, 'drop_collection'):
            await client.drop_collection(collection_name)
    except:
        pass
        
    await client.create_collection(collection_name, dimension)
    
    # Test 1: Immediate visibility of single insert
    print("\n1. Testing immediate visibility of single insert...")
    data_gen = DataGenerator(dimension)
    test_vector = data_gen.generate_vectors(1)[0]
    test_vector.id = "immediate-test"
    
    await client.insert_batch(collection_name, [test_vector])
    
    # Search immediately
    search_results = await client.search(
        collection_name, 
        test_vector.vector, 
        limit=1
    )
    
    if search_results and len(search_results) > 0 and search_results[0][0] == test_vector.id:
        results["tests"]["immediate_visibility"] = {
            "status": "✓",
            "message": "Vector immediately searchable after insert"
        }
        print("   ✓ Vector immediately searchable")
    else:
        results["tests"]["immediate_visibility"] = {
            "status": "✗",
            "message": "Vector not immediately searchable"
        }
        print("   ✗ Vector not immediately searchable")
    
    # Test 2: Visibility under continuous insert load
    print("\n2. Testing visibility under continuous insert load...")
    visibility_delays = []
    missing_vectors = []
    
    async def insert_and_check(index: int):
        """Insert a vector and measure time until searchable"""
        vec = data_gen.generate_vectors(1, prefix="load-test")[0]
        
        insert_time = time.time()
        await client.insert_batch(collection_name, [vec])
        
        # Poll for visibility (max 5 seconds)
        visible = False
        for _ in range(50):  # 50 * 0.1 = 5 seconds max
            search_results = await client.search(
                collection_name,
                vec.vector,
                limit=1
            )
            if search_results and len(search_results) > 0 and search_results[0][0] == vec.id:
                visible = True
                visibility_time = time.time() - insert_time
                visibility_delays.append(visibility_time)
                break
            await asyncio.sleep(0.1)
        
        if not visible:
            missing_vectors.append(vec.id)
    
    # Insert vectors concurrently
    insert_tasks = []
    for i in range(20):
        insert_tasks.append(insert_and_check(i))
    
    await asyncio.gather(*insert_tasks)
    
    if not visibility_delays:
        results["tests"]["load_visibility"] = {
            "status": "✗",
            "message": f"No vectors became searchable under load"
        }
    else:
        avg_delay = sum(visibility_delays) / len(visibility_delays)
        max_delay = max(visibility_delays)
        
        if len(missing_vectors) == 0:
            status = "✓"
            message = f"All vectors searchable. Avg delay: {avg_delay:.3f}s, Max: {max_delay:.3f}s"
        elif len(missing_vectors) < 5:
            status = "⚠"
            message = f"{len(missing_vectors)}/20 vectors not searchable. Avg delay: {avg_delay:.3f}s"
        else:
            status = "✗"
            message = f"{len(missing_vectors)}/20 vectors not searchable"
        
        results["tests"]["load_visibility"] = {
            "status": status,
            "message": message,
            "details": {
                "visible_count": len(visibility_delays),
                "missing_count": len(missing_vectors),
                "avg_delay_ms": int(avg_delay * 1000),
                "max_delay_ms": int(max_delay * 1000) if visibility_delays else 0
            }
        }
        print(f"   {status} {message}")
    
    # Test 3: Bulk insert visibility
    print("\n3. Testing bulk insert visibility...")
    bulk_vectors = data_gen.generate_vectors(100, prefix="bulk")
    
    bulk_insert_time = time.time()
    await client.insert_batch(collection_name, bulk_vectors)
    
    # Check visibility of random samples
    sample_indices = [0, 25, 50, 75, 99]
    visible_count = 0
    visibility_times = []
    
    for idx in sample_indices:
        test_vec = bulk_vectors[idx]
        visible = False
        
        for _ in range(50):  # 5 seconds max
            search_results = await client.search(
                collection_name,
                test_vec.vector,
                limit=1
            )
            if search_results and len(search_results) > 0 and search_results[0][0] == test_vec.id:
                visible = True
                visible_count += 1
                visibility_time = time.time() - bulk_insert_time
                visibility_times.append(visibility_time)
                break
            await asyncio.sleep(0.1)
    
    if visible_count == len(sample_indices):
        avg_time = sum(visibility_times) / len(visibility_times)
        results["tests"]["bulk_visibility"] = {
            "status": "✓",
            "message": f"All sampled vectors searchable. Avg time: {avg_time:.3f}s"
        }
        print(f"   ✓ All sampled vectors searchable")
    elif visible_count > 0:
        results["tests"]["bulk_visibility"] = {
            "status": "⚠",
            "message": f"{visible_count}/{len(sample_indices)} sampled vectors searchable"
        }
        print(f"   ⚠ {visible_count}/{len(sample_indices)} sampled vectors searchable")
    else:
        results["tests"]["bulk_visibility"] = {
            "status": "✗",
            "message": "No sampled vectors became searchable"
        }
        print("   ✗ No sampled vectors became searchable")
    
    # Test 4: Search consistency during inserts
    print("\n4. Testing search consistency during continuous inserts...")
    
    # Create a reference vector for repeated searches
    ref_vector = data_gen.generate_vectors(1)[0]
    ref_vector.id = "reference-vector"
    await client.insert_batch(collection_name, [ref_vector])
    
    # Wait for it to be searchable
    await asyncio.sleep(0.5)
    
    # Perform searches while inserting
    search_results_consistency = []
    
    async def continuous_insert():
        """Continuously insert vectors"""
        for i in range(50):
            vec = data_gen.generate_vectors(1, prefix="continuous")[0]
            await client.insert_batch(collection_name, [vec])
            await asyncio.sleep(0.02)  # 20ms between inserts
    
    async def continuous_search():
        """Continuously search and check consistency"""
        for _ in range(10):
            results = await client.search(
                collection_name,
                ref_vector.vector,
                limit=5
            )
            if results and len(results) > 0:
                # Check if reference vector is in top results
                found_ref = any(r[0] == "reference-vector" for r in results)
                search_results_consistency.append(found_ref)
            else:
                search_results_consistency.append(False)
            await asyncio.sleep(0.1)
    
    # Run inserts and searches concurrently
    await asyncio.gather(
        continuous_insert(),
        continuous_search()
    )
    
    consistency_rate = sum(search_results_consistency) / len(search_results_consistency) if search_results_consistency else 0
    
    if consistency_rate == 1.0:
        results["tests"]["search_consistency"] = {
            "status": "✓",
            "message": "Search results remain consistent during inserts"
        }
        print("   ✓ Search results consistent")
    elif consistency_rate > 0.8:
        results["tests"]["search_consistency"] = {
            "status": "⚠",
            "message": f"Search results {consistency_rate*100:.0f}% consistent during inserts"
        }
        print(f"   ⚠ Search results {consistency_rate*100:.0f}% consistent")
    else:
        results["tests"]["search_consistency"] = {
            "status": "✗",
            "message": f"Search results only {consistency_rate*100:.0f}% consistent"
        }
        print(f"   ✗ Search results only {consistency_rate*100:.0f}% consistent")
    
    # Cleanup
    if hasattr(client, 'drop_collection'):
        await client.drop_collection(collection_name)
    
    # Calculate overall result
    statuses = [test["status"] for test in results["tests"].values()]
    passed = statuses.count("✓")
    partial = statuses.count("⚠")
    failed = statuses.count("✗")
    
    results["summary"] = {
        "total_tests": len(statuses),
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "overall_status": "✓" if passed == len(statuses) else 
                         "⚠" if passed + partial >= len(statuses) // 2 else "✗"
    }
    
    print(f"\nSummary: {passed}/{len(statuses)} tests passed")
    print(f"Overall Status: {results['summary']['overall_status']}")
    
    return results

async def main():
    """Run index synchronization tests for all databases"""
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test each database
    clients = {
        'pgvector': PgVectorClient({"connection_string": "postgresql://postgres:postgres@localhost:15432/vectordb"}),
        'qdrant': QdrantDBClient({"host": "localhost", "port": 16333}),
        'milvus': MilvusDBClient({"host": "localhost", "port": 19530}),
        'chroma': ChromaDBClient({"host": "localhost", "port": 18000})
    }
    
    for db_name, client in clients.items():
        try:
            if hasattr(client, 'connect'):
                await client.connect()
            
            results = await test_index_sync(db_name, client)
            all_results.append(results)
            
        except Exception as e:
            print(f"\nError testing {db_name}: {str(e)}")
            all_results.append({
                "database": db_name,
                "error": str(e),
                "test_type": "index_data_synchronization"
            })
        finally:
            if hasattr(client, 'disconnect'):
                await client.disconnect()
    
    # Save results
    results_file = Path(__file__).parent.parent / "results" / f"consistency_index_sync_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "test": "consistency_index_synchronization",
            "timestamp": timestamp,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # Generate report
    from utils.report_generator import ReportGenerator
    report_gen = ReportGenerator()
    report_path = report_gen.generate_consistency_report(
        {"consistency_index_sync": all_results},
        "index_synchronization"
    )
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())