import numpy as np
from typing import List, Dict, Any, Optional
import uuid
import time
import random
from dataclasses import dataclass
from db_clients.base_client import VectorData


class DataGenerator:
    """Generate test data for ACID testing"""
    
    def __init__(self, dimension: int = 384, seed: Optional[int] = None):
        self.dimension = dimension
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def generate_vectors(self, count: int, 
                        prefix: str = "vec",
                        start_index: int = 0) -> List[VectorData]:
        """Generate test vectors with metadata"""
        vectors = []
        
        for i in range(count):
            idx = start_index + i
            vectors.append(VectorData(
                id=f"{prefix}-{str(uuid.uuid4())}",
                vector=self._generate_random_vector(),
                metadata={
                    'index': idx,
                    'timestamp': time.time(),
                    'category': f"cat_{idx % 10}",
                    'description': f"Test vector {idx}",
                    'tags': self._generate_tags(idx),
                    'score': random.uniform(0, 100)
                }
            ))
        
        return vectors
    
    def generate_conflict_vectors(self, base_id: str, count: int = 10) -> List[VectorData]:
        """Generate vectors with same ID to test conflict handling"""
        vectors = []
        
        for i in range(count):
            vectors.append(VectorData(
                id=base_id,
                vector=self._generate_random_vector(),
                metadata={
                    'version': i,
                    'timestamp': time.time() + i * 0.001,
                    'conflict_test': True
                }
            ))
        
        return vectors
    
    def generate_invalid_vectors(self, count: int = 5) -> List[Dict[str, Any]]:
        """Generate invalid vectors for error testing"""
        invalid_vectors = []
        
        # Wrong dimension
        invalid_vectors.append({
            'id': str(uuid.uuid4()),
            'vector': np.random.randn(self.dimension - 1).tolist(),
            'metadata': {'error_type': 'wrong_dimension'}
        })
        
        # Invalid vector values
        invalid_vectors.append({
            'id': str(uuid.uuid4()),
            'vector': [float('inf')] * self.dimension,
            'metadata': {'error_type': 'invalid_values'}
        })
        
        # Missing vector
        invalid_vectors.append({
            'id': str(uuid.uuid4()),
            'metadata': {'error_type': 'missing_vector'}
        })
        
        # Invalid ID
        invalid_vectors.append({
            'id': '',
            'vector': self._generate_random_vector(),
            'metadata': {'error_type': 'invalid_id'}
        })
        
        # Oversized metadata
        invalid_vectors.append({
            'id': str(uuid.uuid4()),
            'vector': self._generate_random_vector(),
            'metadata': {
                'error_type': 'oversized_metadata',
                'large_field': 'x' * 1000000  # 1MB string
            }
        })
        
        return invalid_vectors[:count]
    
    def generate_batch_with_failure(self, batch_size: int, 
                                  failure_index: int,
                                  failure_type: str = 'duplicate') -> List[VectorData]:
        """Generate a batch with a deliberate failure at specific index"""
        vectors = self.generate_vectors(batch_size)
        
        if failure_type == 'duplicate' and failure_index > 0:
            # Make the failure_index item a duplicate of previous
            vectors[failure_index].id = vectors[failure_index - 1].id
        elif failure_type == 'invalid_dimension':
            # Wrong dimension vector
            vectors[failure_index].vector = vectors[failure_index].vector[:-1]
        elif failure_type == 'null_vector':
            vectors[failure_index].vector = None
        
        return vectors
    
    def generate_query_vectors(self, count: int) -> List[List[float]]:
        """Generate query vectors for search testing"""
        return [self._generate_random_vector() for _ in range(count)]
    
    def _generate_random_vector(self) -> List[float]:
        """Generate a random normalized vector"""
        vector = np.random.randn(self.dimension)
        # Normalize to unit length for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
    
    def _generate_tags(self, index: int) -> List[str]:
        """Generate random tags for metadata"""
        tag_pool = ['important', 'test', 'sample', 'vector', 'data', 
                   'experiment', 'benchmark', 'acid', 'database']
        num_tags = random.randint(1, 4)
        return random.sample(tag_pool, num_tags)
    
    def generate_update_scenarios(self) -> List[Dict[str, Any]]:
        """Generate various update scenarios for testing"""
        scenarios = []
        
        # Update vector only
        scenarios.append({
            'type': 'vector_only',
            'vector': self._generate_random_vector(),
            'metadata': None
        })
        
        # Update metadata only
        scenarios.append({
            'type': 'metadata_only',
            'vector': None,
            'metadata': {
                'updated': True,
                'update_time': time.time(),
                'version': 2
            }
        })
        
        # Update both
        scenarios.append({
            'type': 'both',
            'vector': self._generate_random_vector(),
            'metadata': {
                'updated': True,
                'update_time': time.time(),
                'version': 3,
                'tags': ['updated', 'modified']
            }
        })
        
        return scenarios


class WorkloadGenerator:
    """Generate realistic workloads for ACID testing"""
    
    def __init__(self, data_generator: DataGenerator):
        self.data_generator = data_generator
    
    def generate_mixed_workload(self, duration_seconds: int = 60) -> List[Dict[str, Any]]:
        """Generate a mixed workload of operations"""
        operations = []
        operation_types = ['insert', 'update', 'delete', 'search']
        weights = [0.4, 0.2, 0.1, 0.3]  # Operation distribution
        
        start_time = time.time()
        op_count = 0
        
        while time.time() - start_time < duration_seconds:
            op_type = random.choices(operation_types, weights=weights)[0]
            
            if op_type == 'insert':
                operations.append({
                    'type': 'insert',
                    'data': self.data_generator.generate_vectors(
                        random.randint(1, 10)
                    ),
                    'timestamp': time.time()
                })
            elif op_type == 'update':
                operations.append({
                    'type': 'update',
                    'id': f"vec-{random.randint(0, 1000)}",
                    'updates': random.choice(
                        self.data_generator.generate_update_scenarios()
                    ),
                    'timestamp': time.time()
                })
            elif op_type == 'delete':
                operations.append({
                    'type': 'delete',
                    'id': f"vec-{random.randint(0, 1000)}",
                    'timestamp': time.time()
                })
            elif op_type == 'search':
                operations.append({
                    'type': 'search',
                    'vector': self.data_generator._generate_random_vector(),
                    'limit': random.randint(5, 20),
                    'timestamp': time.time()
                })
            
            op_count += 1
            # Small delay to simulate realistic workload
            time.sleep(random.uniform(0.001, 0.01))
        
        return operations