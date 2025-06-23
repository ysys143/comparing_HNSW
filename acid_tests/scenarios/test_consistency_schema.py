"""
Stage 2: Consistency Tests - Schema Constraint Validation
Validates dimension enforcement and rejection of structurally invalid vectors
"""
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import uuid

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_generator import DataGenerator
from clients.pgvector_client import PgVectorClient
from clients.qdrant_client import QdrantDBClient  
from clients.milvus_client import MilvusDBClient
from clients.chroma_client import ChromaDBClient

async def test_schema_constraints(db_name: str, client: Any) -> Dict[str, Any]:
    """Test schema constraint validation for vectors"""
    print(f"\n{'='*60}")
    print(f"Testing Schema Constraints - {db_name}")
    print(f"{'='*60}")
    
    collection_name = f"test_schema_{db_name}"
    dimension = 768
    results = {
        "database": db_name,
        "test_type": "schema_constraint_validation",
        "tests": {}
    }
    
    # Initialize collection with specific dimension
    try:
        if hasattr(client, 'delete_collection'):
            await client.delete_collection(collection_name)
    except:
        pass
        
    await client.create_collection(collection_name, dimension)
    
    # Initialize data generator
    data_gen = DataGenerator(dimension)
    
    # Test 1: Correct dimension vector
    print("\n1. Testing correct dimension vector...")
    try:
        correct_vector = data_gen.generate_vectors(1)[0]
        await client.insert_single(collection_name, correct_vector)
        results["tests"]["correct_dimension"] = {
            "status": "✓",
            "message": "Accepted vector with correct dimension"
        }
        print("   ✓ Accepted")
    except Exception as e:
        results["tests"]["correct_dimension"] = {
            "status": "✗",
            "message": f"Rejected correct dimension: {str(e)}"
        }
        print(f"   ✗ Rejected: {str(e)}")
    
    # Test 2: Wrong dimension vector (too small)
    print("\n2. Testing wrong dimension vector (too small)...")
    try:
        small_gen = DataGenerator(dimension - 100)
        small_vector = small_gen.generate_vectors(1)[0]
        # small_vector.id = f"small-{small_vector.id}"  # Keep pure UUID
        await client.insert_single(collection_name, small_vector)
        results["tests"]["small_dimension"] = {
            "status": "✗",
            "message": "Incorrectly accepted vector with wrong dimension"
        }
        print("   ✗ Incorrectly accepted")
    except Exception as e:
        results["tests"]["small_dimension"] = {
            "status": "✓", 
            "message": f"Correctly rejected: {str(e)}"
        }
        print(f"   ✓ Correctly rejected: {str(e)}")
    
    # Test 3: Wrong dimension vector (too large)
    print("\n3. Testing wrong dimension vector (too large)...")
    try:
        large_gen = DataGenerator(dimension + 100)
        large_vector = large_gen.generate_vectors(1)[0]
        # large_vector.id = f"large-{large_vector.id}"  # Keep pure UUID
        await client.insert_single(collection_name, large_vector)
        results["tests"]["large_dimension"] = {
            "status": "✗",
            "message": "Incorrectly accepted vector with wrong dimension"
        }
        print("   ✗ Incorrectly accepted")
    except Exception as e:
        results["tests"]["large_dimension"] = {
            "status": "✓",
            "message": f"Correctly rejected: {str(e)}"
        }
        print(f"   ✓ Correctly rejected: {str(e)}")
    
    # Test 4: Batch with mixed dimensions
    print("\n4. Testing batch with mixed dimensions...")
    try:
        mixed_batch = []
        small_gen = DataGenerator(dimension - 50)
        large_gen = DataGenerator(dimension + 50)
        
        for i in range(10):
            if i % 3 == 0:
                vec = small_gen.generate_vectors(1)[0]
            elif i % 3 == 1:
                vec = large_gen.generate_vectors(1)[0]
            else:
                vec = data_gen.generate_vectors(1)[0]
            vec.metadata["mixed_batch_index"] = i
            mixed_batch.append(vec)
        
        await client.insert_batch(collection_name, mixed_batch)
        
        # Check how many were actually inserted
        if hasattr(client, 'get_stats'):
            stats = await client.get_stats(collection_name)
            count = stats.get('count', 0)
            initial_count = 1  # From test 1
            inserted = count - initial_count
            results["tests"]["mixed_batch"] = {
                "status": "⚠",
                "message": f"Partial acceptance: {inserted}/10 vectors with mixed dimensions"
            }
            print(f"   ⚠ Partial acceptance: {inserted}/10 vectors")
        else:
            results["tests"]["mixed_batch"] = {
                "status": "?",
                "message": "Batch accepted - unable to verify count"
            }
    except Exception as e:
        results["tests"]["mixed_batch"] = {
            "status": "✓",
            "message": f"Correctly rejected entire batch: {str(e)}"
        }
        print(f"   ✓ Correctly rejected entire batch: {str(e)}")
    
    # Test 5: Empty vector
    print("\n5. Testing empty vector...")
    try:
        from clients.base_client import VectorData
        empty_vector = VectorData(
            id=str(uuid.uuid4()),  # Use pure UUID
            vector=[],  # Empty array
            metadata={"type": "empty"}
        )
        await client.insert_single(collection_name, empty_vector)
        results["tests"]["empty_vector"] = {
            "status": "✗",
            "message": "Incorrectly accepted empty vector"
        }
        print("   ✗ Incorrectly accepted")
    except Exception as e:
        results["tests"]["empty_vector"] = {
            "status": "✓",
            "message": f"Correctly rejected: {str(e)}"
        }
        print(f"   ✓ Correctly rejected: {str(e)}")
    
    # Test 6: Null/None values in vector
    print("\n6. Testing vector with null values...")
    try:
        from clients.base_client import VectorData
        null_vector = VectorData(
            id=str(uuid.uuid4()),  # Use pure UUID
            vector=[0.1] * (dimension // 2) + [None] * (dimension // 2),
            metadata={"type": "null_values"}
        )
        await client.insert_single(collection_name, null_vector)
        results["tests"]["null_values"] = {
            "status": "✗",
            "message": "Incorrectly accepted vector with null values"
        }
        print("   ✗ Incorrectly accepted")
    except Exception as e:
        results["tests"]["null_values"] = {
            "status": "✓",
            "message": f"Correctly rejected: {str(e)}"
        }
        print(f"   ✓ Correctly rejected: {str(e)}")
    
    # Cleanup
    if hasattr(client, 'drop_collection'):
        await client.drop_collection(collection_name)
    
    # Calculate overall result
    total_tests = len(results["tests"])
    passed_tests = sum(1 for test in results["tests"].values() 
                      if test["status"] == "✓")
    partial_tests = sum(1 for test in results["tests"].values() 
                       if test["status"] == "⚠")
    
    results["summary"] = {
        "total_tests": total_tests,
        "passed": passed_tests,
        "partial": partial_tests,
        "failed": total_tests - passed_tests - partial_tests,
        "overall_status": "✓" if passed_tests == total_tests else 
                         "⚠" if passed_tests + partial_tests >= total_tests // 2 else "✗"
    }
    
    print(f"\nSummary: {passed_tests}/{total_tests} tests passed")
    print(f"Overall Status: {results['summary']['overall_status']}")
    
    return results

async def main():
    """Run schema constraint tests for all databases"""
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
            
            results = await test_schema_constraints(db_name, client)
            all_results.append(results)
            
        except Exception as e:
            print(f"\nError testing {db_name}: {str(e)}")
            all_results.append({
                "database": db_name,
                "error": str(e),
                "test_type": "schema_constraint_validation"
            })
        finally:
            if hasattr(client, 'disconnect'):
                await client.disconnect()
    
    # Save results
    results_file = Path(__file__).parent.parent / "results" / f"consistency_schema_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "test": "consistency_schema_constraints",
            "timestamp": timestamp,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # Generate report
    from utils.report_generator import ReportGenerator
    report_gen = ReportGenerator()
    report_path = report_gen.generate_consistency_report(
        {"consistency_schema": all_results},
        "schema_constraints"
    )
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())