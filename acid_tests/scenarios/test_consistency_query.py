"""
Stage 2: Consistency Tests - Query Result Consistency
Observes whether ongoing updates cause transient inconsistencies or stale reads in search responses
"""
import asyncio
import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_generator import DataGenerator
from clients.pgvector_client import PgVectorClient
from clients.qdrant_client import QdrantDBClient  
from clients.milvus_client import MilvusDBClient
from clients.chroma_client import ChromaDBClient
from clients.base_client import VectorData

async def test_query_consistency(db_name: str, client: Any) -> Dict[str, Any]:
    """Test query result consistency during updates"""
    print(f"\n{'='*60}")
    print(f"Testing Query Result Consistency - {db_name}")
    print(f"{'='*60}")
    
    collection_name = f"test_query_consistency_{db_name}"
    dimension = 128
    results = {
        "database": db_name,
        "test_type": "query_result_consistency",
        "tests": {}
    }
    
    # Initialize collection
    try:
        if hasattr(client, 'drop_collection'):
            await client.drop_collection(collection_name)
    except:
        pass
        
    await client.create_collection(collection_name, dimension)
    
    # Test 1: Read-after-write consistency
    print("\n1. Testing read-after-write consistency...")
    
    # Insert initial dataset
    data_gen = DataGenerator(dimension)
    initial_vectors = data_gen.generate_vectors(50, prefix="initial")
    for i, vec in enumerate(initial_vectors):
        vec.metadata["group"] = "initial"
    
    await client.insert_batch(collection_name, initial_vectors)
    
    # Immediately query for specific vectors
    consistency_checks = []
    
    for i in range(10):
        target_vec = initial_vectors[i * 5]  # Sample every 5th vector
        
        # Search for the exact vector
        results_immediate = await client.search(
            collection_name,
            target_vec.vector,
            limit=1
        )
        
        if results_immediate and len(results_immediate) > 0:
            if results_immediate[0][0] == target_vec.id:
                consistency_checks.append("exact_match")
            else:
                consistency_checks.append("different_match")
        else:
            consistency_checks.append("no_match")
        
        await asyncio.sleep(0.1)
    
    exact_matches = consistency_checks.count("exact_match")
    
    if exact_matches == len(consistency_checks):
        results["tests"]["read_after_write"] = {
            "status": "✓",
            "message": "Perfect read-after-write consistency"
        }
        print("   ✓ Perfect read-after-write consistency")
    elif exact_matches > len(consistency_checks) * 0.8:
        results["tests"]["read_after_write"] = {
            "status": "⚠",
            "message": f"{exact_matches}/{len(consistency_checks)} exact matches after write"
        }
        print(f"   ⚠ {exact_matches}/{len(consistency_checks)} exact matches")
    else:
        results["tests"]["read_after_write"] = {
            "status": "✗",
            "message": f"Only {exact_matches}/{len(consistency_checks)} exact matches"
        }
        print(f"   ✗ Only {exact_matches}/{len(consistency_checks)} exact matches")
    
    # Test 2: Query consistency during continuous updates
    print("\n2. Testing query consistency during continuous updates...")
    
    # Create reference query vector
    query_vector = data_gen.generate_vectors(1)[0]
    
    # Get initial top-k results
    k = 10
    initial_results = await client.search(collection_name, query_vector.vector, limit=k)
    initial_ids = [r[0] for r in initial_results] if initial_results else []
    
    # Track consistency during updates
    consistency_observations = []
    result_changes = []
    
    async def continuous_updates():
        """Continuously add new vectors"""
        for i in range(30):
            new_vec = data_gen.generate_vectors(1, prefix="update")[0]
            new_vec.metadata["group"] = "update"
            new_vec.metadata["update_index"] = i
            await client.insert_batch(collection_name, [new_vec])
            await asyncio.sleep(0.05)  # 50ms between updates
    
    async def monitor_query_results():
        """Monitor query results for consistency"""
        for i in range(15):
            current_results = await client.search(
                collection_name, 
                query_vector.vector, 
                limit=k
            )
            
            if current_results:
                current_ids = [r[0] for r in current_results]
                
                # Check if results are stable subset of initial
                initial_overlap = len(set(current_ids) & set(initial_ids))
                consistency_observations.append({
                    "iteration": i,
                    "overlap": initial_overlap,
                    "total": len(current_ids)
                })
                
                # Track if order changed significantly
                if i > 0 and len(result_changes) > 0:
                    prev_ids = result_changes[-1]
                    order_preserved = all(
                        c == p for c, p in zip(current_ids[:5], prev_ids[:5])
                    )
                    consistency_observations[-1]["order_preserved"] = order_preserved
                
                result_changes.append(current_ids)
            
            await asyncio.sleep(0.1)  # 100ms between checks
    
    # Run updates and monitoring concurrently
    await asyncio.gather(
        continuous_updates(),
        monitor_query_results()
    )
    
    # Analyze consistency
    if consistency_observations:
        avg_overlap = sum(obs["overlap"] for obs in consistency_observations) / len(consistency_observations)
        order_changes = sum(
            1 for obs in consistency_observations 
            if "order_preserved" in obs and not obs["order_preserved"]
        )
        
        if avg_overlap >= k * 0.8 and order_changes < 3:
            results["tests"]["query_during_updates"] = {
                "status": "✓",
                "message": f"Stable results during updates (avg {avg_overlap:.1f}/{k} overlap)"
            }
            print(f"   ✓ Stable results during updates")
        elif avg_overlap >= k * 0.5:
            results["tests"]["query_during_updates"] = {
                "status": "⚠",
                "message": f"Moderate stability (avg {avg_overlap:.1f}/{k} overlap, {order_changes} order changes)"
            }
            print(f"   ⚠ Moderate stability during updates")
        else:
            results["tests"]["query_during_updates"] = {
                "status": "✗",
                "message": f"Unstable results (avg {avg_overlap:.1f}/{k} overlap)"
            }
            print(f"   ✗ Unstable results during updates")
    
    # Test 3: Concurrent read consistency
    print("\n3. Testing concurrent read consistency...")
    
    # Perform multiple concurrent searches with same query
    concurrent_results = []
    
    async def concurrent_search(search_id: int):
        """Perform a search and return results"""
        results = await client.search(
            collection_name,
            query_vector.vector,
            limit=5
        )
        return {
            "search_id": search_id,
            "ids": [r[0] for r in results] if results else [],
            "scores": [r[1] for r in results] if results else []
        }
    
    # Execute 10 concurrent searches
    search_tasks = [concurrent_search(i) for i in range(10)]
    concurrent_results = await asyncio.gather(*search_tasks)
    
    # Check consistency across concurrent reads
    if concurrent_results and all(r["ids"] for r in concurrent_results):
        first_ids = concurrent_results[0]["ids"]
        all_identical = all(r["ids"] == first_ids for r in concurrent_results)
        
        if all_identical:
            results["tests"]["concurrent_reads"] = {
                "status": "✓",
                "message": "All concurrent reads returned identical results"
            }
            print("   ✓ All concurrent reads identical")
        else:
            # Count variations
            unique_results = []
            for r in concurrent_results:
                if not any(r["ids"] == u for u in unique_results):
                    unique_results.append(r["ids"])
            
            if len(unique_results) <= 2:
                results["tests"]["concurrent_reads"] = {
                    "status": "⚠",
                    "message": f"{len(unique_results)} different result sets across concurrent reads"
                }
                print(f"   ⚠ {len(unique_results)} different result sets")
            else:
                results["tests"]["concurrent_reads"] = {
                    "status": "✗",
                    "message": f"{len(unique_results)} different result sets - high inconsistency"
                }
                print(f"   ✗ {len(unique_results)} different result sets")
    
    # Test 4: Stale read detection
    print("\n4. Testing for stale reads after deletions...")
    
    # Create vectors to be deleted
    delete_vectors = data_gen.generate_vectors(10, prefix="delete")
    for i, vec in enumerate(delete_vectors):
        vec.metadata["marked_for_deletion"] = True
        vec.metadata["delete_index"] = i
    
    await client.insert_batch(collection_name, delete_vectors)
    
    # Delete half of them (if deletion is supported)
    deleted_ids = [vec.id for vec in delete_vectors[:5]]
    
    if hasattr(client, 'delete_vector'):
        try:
            for del_id in deleted_ids:
                await client.delete_vector(collection_name, del_id)
            deletion_supported = True
        except Exception as e:
            deletion_supported = False
            print(f"   Deletion not supported: {str(e)}")
    else:
        deletion_supported = False
        print("   Deletion not supported by client")
    
    # Check if we actually inserted vectors before testing deletion
    if not deletion_supported:
        results["tests"]["stale_reads"] = {
            "status": "N/A",
            "message": "Deletion not supported - test skipped"
        }
        print("   N/A Deletion not supported")
    else:
        # First verify vectors were actually inserted
        try:
            initial_search = await client.search(
                collection_name,
                delete_vectors[0].vector,
                limit=10
            )
            vectors_exist = bool(initial_search)
        except:
            vectors_exist = False
        
        if not vectors_exist:
            results["tests"]["stale_reads"] = {
                "status": "N/A",
                "message": "No vectors to delete - insertion failed"
            }
            print("   N/A No vectors to delete")
        else:
            # Search and check for stale results
            stale_reads = []
            
            for _ in range(5):
                search_results = await client.search(
                    collection_name,
                    query_vector.vector,
                    limit=20
                )
                
                if search_results:
                    found_deleted = [
                        r[0] for r in search_results 
                        if r[0] in deleted_ids
                    ]
                    if found_deleted:
                        stale_reads.extend(found_deleted)
                
                await asyncio.sleep(0.2)
            
            if not stale_reads:
                results["tests"]["stale_reads"] = {
                    "status": "✓",
                    "message": "No stale reads detected after deletion"
                }
                print("   ✓ No stale reads detected")
            elif len(set(stale_reads)) < 2:
                results["tests"]["stale_reads"] = {
                    "status": "⚠",
                    "message": f"Minor stale reads detected: {len(set(stale_reads))} deleted IDs found"
                }
                print(f"   ⚠ Minor stale reads: {len(set(stale_reads))} deleted IDs")
            else:
                results["tests"]["stale_reads"] = {
                    "status": "✗",
                    "message": f"Significant stale reads: {len(set(stale_reads))} deleted IDs found"
                }
                print(f"   ✗ Significant stale reads: {len(set(stale_reads))} deleted IDs")
    
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
    """Run query consistency tests for all databases"""
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
            
            results = await test_query_consistency(db_name, client)
            all_results.append(results)
            
        except Exception as e:
            print(f"\nError testing {db_name}: {str(e)}")
            all_results.append({
                "database": db_name,
                "error": str(e),
                "test_type": "query_result_consistency"
            })
        finally:
            if hasattr(client, 'disconnect'):
                await client.disconnect()
    
    # Save results
    results_file = Path(__file__).parent.parent / "results" / f"consistency_query_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "test": "consistency_query_results",
            "timestamp": timestamp,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # Generate report
    from utils.report_generator import ReportGenerator
    report_gen = ReportGenerator()
    report_path = report_gen.generate_consistency_report(
        {"consistency_query": all_results},
        "query_consistency"
    )
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())