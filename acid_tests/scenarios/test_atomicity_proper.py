#!/usr/bin/env python3
"""
Proper Atomicity Test Implementation according to acid_test_plan.md
Tests true transactional behavior by using valid data throughout
"""

import asyncio
import sys
import os
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.base_client import VectorData
from utils.data_generator import DataGenerator
from utils.report_generator import ReportGenerator


class ProperTestDataGenerator(DataGenerator):
    """Generate proper test data without ID format issues"""
    
    def generate_vectors_with_proper_ids(self, count: int) -> list[VectorData]:
        """Generate vectors with database-compatible IDs"""
        vectors = []
        
        for i in range(count):
            # Use pure UUIDs for databases that require them
            vector_id = str(uuid.uuid4())
            
            vectors.append(VectorData(
                id=vector_id,
                vector=self._generate_random_vector(),
                metadata={
                    'index': i,
                    'timestamp': asyncio.get_event_loop().time(),
                    'category': f"cat_{i % 10}",
                    'description': f"Test vector {i}",
                    'score': float(i % 100)
                }
            ))
        
        return vectors
    
    def generate_batch_with_invalid_vector(self, count: int, invalid_position: int) -> list[VectorData]:
        """Generate batch with an invalid vector at specific position"""
        vectors = self.generate_vectors_with_proper_ids(count)
        
        # Make one vector invalid (wrong dimension)
        if 0 <= invalid_position < count:
            # Create vector with wrong dimension
            vectors[invalid_position].vector = self._generate_random_vector()[:-10]  # Remove 10 dimensions
        
        return vectors


async def test_transactional_atomicity_pgvector(client):
    """Test true SQL transaction rollback for pgvector"""
    print("\n" + "="*60)
    print("TRANSACTIONAL ATOMICITY TEST (pgvector)")
    print("="*60)
    
    generator = ProperTestDataGenerator(dimension=384)
    collection_name = "acid_test_transaction"
    
    try:
        await client.drop_collection(collection_name)
        await client.create_collection(collection_name, dimension=384)
        
        initial_count = await client.count(collection_name)
        print(f"\n1. Initial state: {initial_count} vectors")
        
        # Test explicit transaction with deliberate failure
        print("\n2. Testing SQL transaction with mid-transaction failure...")
        
        try:
            async with client.connection.transaction():
                # Insert some valid vectors
                batch1 = generator.generate_vectors_with_proper_ids(50)
                for vec in batch1:
                    await client.insert_single(collection_name, vec)
                print("   - Inserted 50 vectors within transaction")
                
                # Insert more vectors
                batch2 = generator.generate_vectors_with_proper_ids(50)
                for vec in batch2:
                    await client.insert_single(collection_name, vec)
                print("   - Inserted another 50 vectors within transaction")
                
                # Force failure
                print("   - Forcing transaction rollback...")
                raise Exception("Deliberate failure to test rollback")
                
        except Exception as e:
            print(f"   - Transaction failed as expected: {str(e)}")
        
        # Check if rollback worked
        final_count = await client.count(collection_name)
        vectors_added = final_count - initial_count
        
        print(f"\n3. Results:")
        print(f"   - Vectors inserted in transaction: 100")
        print(f"   - Vectors after rollback: {vectors_added}")
        
        if vectors_added == 0:
            print("\n✅ TRANSACTION ROLLBACK SUCCESSFUL")
            print("   → All inserts within transaction were rolled back")
            print("   → True ACID compliance demonstrated")
            return {'rollback_successful': True, 'vectors_rolled_back': 100}
        else:
            print("\n❌ TRANSACTION ROLLBACK FAILED")
            print(f"   → {vectors_added} vectors persisted despite rollback")
            return {'rollback_successful': False, 'vectors_persisted': vectors_added}
            
    finally:
        await client.drop_collection(collection_name)


async def test_best_effort_insert_behavior(db_name: str, client):
    """Test best-effort insert behavior for non-transactional databases"""
    print(f"\n" + "="*60)
    print(f"BEST-EFFORT INSERT BEHAVIOR TEST ({db_name})")
    print("="*60)
    
    generator = ProperTestDataGenerator(dimension=384)
    collection_name = "acid_test_best_effort"
    
    try:
        await client.drop_collection(collection_name)
        await client.create_collection(collection_name, dimension=384)
        
        initial_count = await client.count(collection_name)
        print(f"\n1. Initial state: {initial_count} vectors")
        
        # Create batch with invalid vector in the middle
        batch_size = 100
        invalid_position = 50
        print(f"\n2. Creating batch of {batch_size} vectors with invalid vector at position {invalid_position}...")
        
        batch = generator.generate_batch_with_invalid_vector(batch_size, invalid_position)
        print(f"   - Invalid vector has dimension: {len(batch[invalid_position].vector)} (should be 384)")
        
        # Attempt batch insert
        print("\n3. Attempting batch insert...")
        success = False
        error_msg = None
        
        try:
            success = await client.insert_batch(collection_name, batch)
            if success:
                print("   - Batch insert returned success")
        except Exception as e:
            error_msg = str(e)
            print(f"   - Batch insert failed: {error_msg[:100]}...")
        
        # Check final count
        await asyncio.sleep(1)
        final_count = await client.count(collection_name)
        vectors_inserted = final_count - initial_count
        
        print(f"\n4. Results:")
        print(f"   - Vectors attempted: {batch_size}")
        print(f"   - Vectors successfully inserted: {vectors_inserted}")
        print(f"   - Success status: {success}")
        
        # Analyze behavior
        print(f"\n5. BEHAVIOR ANALYSIS:")
        
        if vectors_inserted == 0:
            print(f"   ❌ ALL-OR-NOTHING BEHAVIOR")
            print(f"   → Entire batch rejected due to invalid vector")
            print(f"   → Not true atomicity, but strict validation")
            behavior = "all_or_nothing_validation"
        elif 0 < vectors_inserted < batch_size:
            print(f"   ✅ BEST-EFFORT BEHAVIOR (EXPECTED)")
            print(f"   → Partial success: {vectors_inserted}/{batch_size} vectors inserted")
            print(f"   → No transaction support - this is normal BASE behavior")
            print(f"   → Invalid vector at position {invalid_position} was skipped")
            behavior = "best_effort_partial_success"
        else:
            print(f"   ⚠️ UNEXPECTED BEHAVIOR")
            print(f"   → All vectors inserted despite invalid one")
            behavior = "unexpected_full_success"
        
        return {
            'behavior': behavior,
            'vectors_attempted': batch_size,
            'vectors_inserted': vectors_inserted,
            'invalid_position': invalid_position,
            'error_message': error_msg
        }
        
    finally:
        await client.drop_collection(collection_name)


async def test_partial_failure_detection(db_name: str, client):
    """Test behavior when failure occurs mid-batch"""
    print(f"\n" + "="*60)
    print(f"PARTIAL FAILURE DETECTION TEST ({db_name})")
    print("="*60)
    
    generator = ProperTestDataGenerator(dimension=384)
    collection_name = "acid_test_partial_failure"
    
    try:
        await client.drop_collection(collection_name)
        await client.create_collection(collection_name, dimension=384)
        
        # First, insert some vectors to create potential conflicts
        print("\n1. Setting up initial data...")
        initial_batch = generator.generate_vectors_with_proper_ids(20)
        await client.insert_batch(collection_name, initial_batch)
        initial_count = await client.count(collection_name)
        print(f"   - Initial vectors: {initial_count}")
        
        # Create batch with duplicate ID in the middle
        print("\n2. Creating batch with duplicate ID...")
        new_batch = generator.generate_vectors_with_proper_ids(100)
        
        # Insert duplicate ID from initial batch
        duplicate_position = 50
        new_batch[duplicate_position].id = initial_batch[10].id
        print(f"   - Duplicate ID at position {duplicate_position}")
        
        # Attempt insert
        print("\n3. Attempting batch insert with duplicate...")
        success = False
        error_msg = None
        
        try:
            success = await client.insert_batch(collection_name, new_batch)
        except Exception as e:
            error_msg = str(e)
            print(f"   - Error: {error_msg[:100]}...")
        
        # Check results
        await asyncio.sleep(1)
        final_count = await client.count(collection_name)
        vectors_added = final_count - initial_count
        
        print(f"\n4. Results:")
        print(f"   - Vectors before duplicate: {duplicate_position}")
        print(f"   - Vectors after duplicate: {100 - duplicate_position - 1}")
        print(f"   - Total vectors added: {vectors_added}")
        
        print(f"\n5. PARTIAL FAILURE ANALYSIS:")
        
        if vectors_added == 0:
            print(f"   → No vectors added - batch fully rejected")
            partial_success = False
        elif vectors_added < 100:
            print(f"   → PARTIAL SUCCESS DETECTED")
            print(f"   → {vectors_added} vectors inserted before/after failure")
            print(f"   → Confirms lack of transactional atomicity")
            partial_success = True
        else:
            print(f"   → All vectors added (duplicate was handled)")
            partial_success = False
        
        return {
            'partial_success': partial_success,
            'vectors_added': vectors_added,
            'duplicate_position': duplicate_position
        }
        
    finally:
        await client.drop_collection(collection_name)


async def main():
    """Run proper atomicity tests according to plan"""
    
    from clients.pgvector_client import PgVectorClient
    from clients.qdrant_client import QdrantDBClient
    from clients.milvus_client import MilvusDBClient
    from clients.chroma_client import ChromaDBClient
    
    configs = {
        'pgvector': {
            'client_class': PgVectorClient,
            'config': {
                'host': 'localhost',
                'port': 15432,
                'user': 'postgres',
                'password': 'postgres',
                'database': 'vectordb'
            }
        },
        'qdrant': {
            'client_class': QdrantDBClient,
            'config': {
                'host': 'localhost',
                'port': 16333
            }
        },
        'milvus': {
            'client_class': MilvusDBClient,
            'config': {
                'host': 'localhost',
                'port': 19530
            }
        },
        'chroma': {
            'client_class': ChromaDBClient,
            'config': {
                'host': 'localhost',
                'port': 18000
            }
        }
    }
    
    print("="*80)
    print("PROPER ATOMICITY TESTS FOR VECTOR DATABASES")
    print("According to acid_test_plan.md specifications")
    print("="*80)
    
    all_results = {}
    
    for db_name, db_config in configs.items():
        print(f"\n\n{'#'*80}")
        print(f"# TESTING {db_name.upper()}")
        print(f"{'#'*80}")
        
        client = db_config['client_class'](db_config['config'])
        
        try:
            await client.connect()
            await client.wait_for_ready(timeout=30)
            
            if db_name == 'pgvector':
                # Test 1: Transactional atomicity (pgvector only)
                transaction_result = await test_transactional_atomicity_pgvector(client)
                all_results[db_name] = {
                    'transactional_atomicity': transaction_result,
                    'supports_transactions': True
                }
            else:
                # Test 2: Best-effort behavior (others)
                best_effort_result = await test_best_effort_insert_behavior(db_name, client)
                
                # Test 3: Partial failure detection
                partial_failure_result = await test_partial_failure_detection(db_name, client)
                
                all_results[db_name] = {
                    'best_effort_behavior': best_effort_result,
                    'partial_failure': partial_failure_result,
                    'supports_transactions': False
                }
                
        except Exception as e:
            print(f"ERROR testing {db_name}: {e}")
            all_results[db_name] = {'error': str(e)}
        finally:
            await client.disconnect()
    
    # Summary according to plan
    print("\n\n" + "="*80)
    print("TEST SUMMARY (According to acid_test_plan.md)")
    print("="*80)
    
    print("\n" + "-"*80)
    print(f"{'Database':<15} {'Transaction Support':<25} {'Behavior':<40}")
    print("-"*80)
    
    for db, results in all_results.items():
        if 'error' not in results:
            if results.get('supports_transactions'):
                # pgvector
                trans = results.get('transactional_atomicity', {})
                if trans.get('rollback_successful'):
                    status = "✓ Rollback supported"
                    behavior = "Full transaction rollback on failure"
                else:
                    status = "✗ Rollback failed"
                    behavior = "Transaction support claimed but not working"
            else:
                # Others
                best_effort = results.get('best_effort_behavior', {})
                partial = results.get('partial_failure', {})
                
                if best_effort.get('behavior') == 'best_effort_partial_success' or partial.get('partial_success'):
                    status = "✗ No rollback (expected)"
                    behavior = "Best-effort insert, partial success allowed"
                else:
                    status = "N/A"
                    behavior = "Validation prevents partial insert"
            
            print(f"{db:<15} {status:<25} {behavior:<40}")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("\n✓ = Transactional rollback supported (ACID)")
    print("✗ = Rollback unsupported (BASE) - This is EXPECTED for Qdrant/Milvus/Chroma")
    print("N/A = Not applicable (validation prevents testing atomicity)")
    
    # Generate reports
    report_gen = ReportGenerator()
    report_gen.save_json_results("proper_atomicity_test", all_results)
    report_gen.generate_markdown_report("proper_atomicity_test", all_results)
    report_gen.generate_html_report("proper_atomicity_test", all_results)


if __name__ == "__main__":
    asyncio.run(main())