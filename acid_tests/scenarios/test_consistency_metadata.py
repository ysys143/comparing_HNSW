"""
Stage 2: Consistency Tests - Metadata Consistency
Verifies consistency of concurrent metadata updates and their final stored state
"""
import asyncio
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_generator import DataGenerator
from clients.pgvector_client import PgVectorClient
from clients.qdrant_client import QdrantDBClient  
from clients.milvus_client import MilvusDBClient
from clients.chroma_client import ChromaDBClient
from clients.base_client import VectorData

async def test_metadata_consistency(db_name: str, client: Any) -> Dict[str, Any]:
    """Test metadata consistency under concurrent updates"""
    print(f"\n{'='*60}")
    print(f"Testing Metadata Consistency - {db_name}")
    print(f"{'='*60}")
    
    collection_name = f"test_metadata_{db_name}"
    dimension = 128
    results = {
        "database": db_name,
        "test_type": "metadata_consistency",
        "tests": {}
    }
    
    # Initialize collection
    try:
        if hasattr(client, 'drop_collection'):
            await client.drop_collection(collection_name)
    except:
        pass
        
    await client.create_collection(collection_name, dimension)
    
    # Test 1: Single metadata update consistency
    print("\n1. Testing single metadata update...")
    data_gen = DataGenerator(dimension)
    test_vector = data_gen.generate_vectors(1)[0]
    test_vector.id = "metadata-test-1"
    test_vector.metadata = {"version": 1, "status": "initial"}
    
    # Insert vector
    await client.insert_batch(collection_name, [test_vector])
    
    # Update metadata (if supported)
    if hasattr(client, 'update_vector'):
        updated_metadata = {"version": 2, "status": "updated", "timestamp": datetime.now().isoformat()}
        try:
            await client.update_vector(collection_name, test_vector.id, metadata=updated_metadata)
            
            # Retrieve and verify
            retrieved = await client.get_vector(collection_name, test_vector.id)
            if retrieved and retrieved.metadata.get("version") == 2:
                results["tests"]["single_update"] = {
                    "status": "✓",
                    "message": "Metadata update successful and consistent"
                }
                print("   ✓ Metadata update successful")
            else:
                results["tests"]["single_update"] = {
                    "status": "✗",
                    "message": "Metadata update failed or inconsistent"
                }
                print("   ✗ Metadata update failed")
        except Exception as e:
            results["tests"]["single_update"] = {
                "status": "N/A",
                "message": f"Metadata updates not supported: {str(e)}"
            }
            print(f"   N/A Metadata updates not supported")
    else:
        # Test via re-insertion
        test_vector.metadata = {"version": 2, "status": "updated"}
        await client.insert_batch(collection_name, [test_vector])
        
        retrieved = await client.get_vector(collection_name, test_vector.id)
        if retrieved and retrieved.metadata.get("version") == 2:
            results["tests"]["single_update"] = {
                "status": "✓",
                "message": "Metadata update via re-insertion successful"
            }
            print("   ✓ Metadata update via re-insertion successful")
        else:
            results["tests"]["single_update"] = {
                "status": "✗",
                "message": "Metadata update failed"
            }
            print("   ✗ Metadata update failed")
    
    # Test 2: Concurrent metadata updates
    print("\n2. Testing concurrent metadata updates...")
    
    # Create test vectors
    num_vectors = 10
    test_vectors = data_gen.generate_vectors(num_vectors, prefix="concurrent")
    for i, vec in enumerate(test_vectors):
        vec.metadata = {"counter": 0, "updates": []}
    
    await client.insert_batch(collection_name, test_vectors)
    
    # Concurrent update function
    update_counts = {}
    update_errors = []
    
    async def update_metadata_concurrent(vec_id: str, update_num: int):
        """Perform a metadata update"""
        try:
            # Get current metadata
            current = await client.get_vector(collection_name, vec_id)
            if not current:
                update_errors.append(f"Vector {vec_id} not found")
                return
            
            # Update metadata
            new_metadata = current.metadata.copy()
            new_metadata["counter"] = new_metadata.get("counter", 0) + 1
            new_metadata["updates"] = new_metadata.get("updates", []) + [update_num]
            
            # Re-insert with updated metadata
            updated_vec = VectorData(
                id=vec_id,
                vector=current.vector,
                metadata=new_metadata
            )
            await client.insert_batch(collection_name, [updated_vec])
            
            if vec_id not in update_counts:
                update_counts[vec_id] = 0
            update_counts[vec_id] += 1
            
        except Exception as e:
            update_errors.append(f"Error updating {vec_id}: {str(e)}")
    
    # Perform concurrent updates
    update_tasks = []
    updates_per_vector = 5
    
    for i in range(num_vectors):
        vec_id = test_vectors[i].id
        for j in range(updates_per_vector):
            update_tasks.append(update_metadata_concurrent(vec_id, j))
    
    # Shuffle to randomize order
    random.shuffle(update_tasks)
    
    # Execute concurrently
    await asyncio.gather(*update_tasks, return_exceptions=True)
    
    # Verify final state
    consistency_issues = []
    correct_counts = 0
    
    for i in range(num_vectors):
        vec_id = test_vectors[i].id
        final = await client.get_vector(collection_name, vec_id)
        
        if final:
            expected_counter = updates_per_vector
            actual_counter = final.metadata.get("counter", 0)
            
            if actual_counter == expected_counter:
                correct_counts += 1
            else:
                consistency_issues.append({
                    "vector": vec_id,
                    "expected": expected_counter,
                    "actual": actual_counter
                })
    
    if correct_counts == num_vectors:
        results["tests"]["concurrent_updates"] = {
            "status": "✓",
            "message": f"All {num_vectors} vectors have consistent metadata after concurrent updates"
        }
        print(f"   ✓ All vectors have consistent metadata")
    elif correct_counts > num_vectors // 2:
        results["tests"]["concurrent_updates"] = {
            "status": "⚠",
            "message": f"{correct_counts}/{num_vectors} vectors have consistent metadata",
            "issues": consistency_issues[:3]  # First 3 issues
        }
        print(f"   ⚠ {correct_counts}/{num_vectors} vectors consistent")
    else:
        results["tests"]["concurrent_updates"] = {
            "status": "✗",
            "message": f"Only {correct_counts}/{num_vectors} vectors have consistent metadata",
            "issues": consistency_issues[:3]
        }
        print(f"   ✗ Only {correct_counts}/{num_vectors} vectors consistent")
    
    # Test 3: Metadata consistency in search results
    print("\n3. Testing metadata consistency in search results...")
    
    # Create vectors with specific metadata
    search_vectors = data_gen.generate_vectors(20, prefix="search")
    for i, vec in enumerate(search_vectors):
        vec.metadata = {
            "category": "A" if i < 10 else "B",
            "value": i,
            "timestamp": datetime.now().isoformat()
        }
    
    await client.insert_batch(collection_name, search_vectors)
    
    # Search and verify metadata
    query_vector = data_gen.generate_vectors(1)[0]
    search_results = await client.search(collection_name, query_vector.vector, limit=10)
    
    metadata_intact = True
    missing_metadata = []
    
    if search_results:
        for result in search_results:
            # result is (id, score) tuple, need to get full vector data
            vec_data = await client.get_vector(collection_name, result[0])
            if vec_data and vec_data.id.startswith("search-"):
                if not vec_data.metadata or "category" not in vec_data.metadata:
                    metadata_intact = False
                    missing_metadata.append(vec_data.id)
    
    if metadata_intact and search_results:
        results["tests"]["search_metadata"] = {
            "status": "✓",
            "message": "All metadata preserved in search results"
        }
        print("   ✓ All metadata preserved in search results")
    elif len(missing_metadata) < len(search_results) // 2:
        results["tests"]["search_metadata"] = {
            "status": "⚠",
            "message": f"{len(missing_metadata)} vectors missing metadata in search results"
        }
        print(f"   ⚠ {len(missing_metadata)} vectors missing metadata")
    else:
        results["tests"]["search_metadata"] = {
            "status": "✗",
            "message": "Metadata not preserved in search results"
        }
        print("   ✗ Metadata not preserved in search results")
    
    # Test 4: Metadata type consistency
    print("\n4. Testing metadata type consistency...")
    
    # Test different metadata types
    type_test_vectors = []
    
    # String metadata
    vec1 = data_gen.generate_vectors(1)[0]
    vec1.id = "type-string"
    vec1.metadata = {"type": "string", "value": "test_string"}
    type_test_vectors.append(vec1)
    
    # Integer metadata
    vec2 = data_gen.generate_vectors(1)[0]
    vec2.id = "type-int"
    vec2.metadata = {"type": "integer", "value": 42}
    type_test_vectors.append(vec2)
    
    # Float metadata
    vec3 = data_gen.generate_vectors(1)[0]
    vec3.id = "type-float"
    vec3.metadata = {"type": "float", "value": 3.14159}
    type_test_vectors.append(vec3)
    
    # Boolean metadata
    vec4 = data_gen.generate_vectors(1)[0]
    vec4.id = "type-bool"
    vec4.metadata = {"type": "boolean", "value": True}
    type_test_vectors.append(vec4)
    
    # Nested metadata (if supported)
    vec5 = data_gen.generate_vectors(1)[0]
    vec5.id = "type-nested"
    vec5.metadata = {"type": "nested", "value": {"nested_key": "nested_value"}}
    type_test_vectors.append(vec5)
    
    # Insert and verify
    type_preservation = {}
    
    for vec in type_test_vectors:
        try:
            await client.insert_batch(collection_name, [vec])
            retrieved = await client.get_vector(collection_name, vec.id)
            
            if retrieved and retrieved.metadata:
                original_value = vec.metadata["value"]
                retrieved_value = retrieved.metadata.get("value")
                
                if type(original_value) == type(retrieved_value) and original_value == retrieved_value:
                    type_preservation[vec.metadata["type"]] = "✓"
                else:
                    type_preservation[vec.metadata["type"]] = "✗"
            else:
                type_preservation[vec.metadata["type"]] = "✗"
                
        except Exception as e:
            type_preservation[vec.metadata["type"]] = f"Error: {str(e)[:30]}"
    
    preserved_count = sum(1 for v in type_preservation.values() if v == "✓")
    
    if preserved_count == len(type_test_vectors):
        results["tests"]["type_consistency"] = {
            "status": "✓",
            "message": "All metadata types preserved correctly",
            "details": type_preservation
        }
        print("   ✓ All metadata types preserved")
    elif preserved_count > len(type_test_vectors) // 2:
        results["tests"]["type_consistency"] = {
            "status": "⚠",
            "message": f"{preserved_count}/{len(type_test_vectors)} metadata types preserved",
            "details": type_preservation
        }
        print(f"   ⚠ {preserved_count}/{len(type_test_vectors)} types preserved")
    else:
        results["tests"]["type_consistency"] = {
            "status": "✗",
            "message": f"Only {preserved_count}/{len(type_test_vectors)} metadata types preserved",
            "details": type_preservation
        }
        print(f"   ✗ Only {preserved_count}/{len(type_test_vectors)} types preserved")
    
    # Cleanup
    if hasattr(client, 'drop_collection'):
        await client.drop_collection(collection_name)
    
    # Calculate overall result
    statuses = [test["status"] for test in results["tests"].values() if test["status"] != "N/A"]
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
    """Run metadata consistency tests for all databases"""
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
            
            results = await test_metadata_consistency(db_name, client)
            all_results.append(results)
            
        except Exception as e:
            print(f"\nError testing {db_name}: {str(e)}")
            all_results.append({
                "database": db_name,
                "error": str(e),
                "test_type": "metadata_consistency"
            })
        finally:
            if hasattr(client, 'disconnect'):
                await client.disconnect()
    
    # Save results
    results_file = Path(__file__).parent.parent / "results" / f"consistency_metadata_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "test": "consistency_metadata",
            "timestamp": timestamp,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # Generate report
    from utils.report_generator import ReportGenerator
    report_gen = ReportGenerator()
    report_path = report_gen.generate_consistency_report(
        {"consistency_metadata": all_results},
        "metadata_consistency"
    )
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())