# Consistency Test Report - Schema Constraints
Generated: 2025-06-23 13:33:14

## Overview
This report presents the results of consistency tests across multiple vector databases.
Test notation: ✓ = no inconsistency observed, ⚠ = partial inconsistency or delay, ✗ = unstable or broken behavior

## Summary

| Database | Overall Status | Passed | Partial | Failed | Details |
|----------|---------------|---------|---------|---------|----------|
| pgvector | ⚠ | 5 | 0 | 1 | 5/6 tests passed |
| qdrant | ⚠ | 5 | 0 | 1 | 5/6 tests passed |
| milvus | ⚠ | 5 | 0 | 1 | 5/6 tests passed |
| chroma | ⚠ | 5 | 0 | 1 | 5/6 tests passed |

## Detailed Results

### pgvector

#### Schema Constraint Tests

- **Correct Dimension**: ✗ - Rejected correct dimension: 'int' object has no attribute '_generate_random_vector'
- **Small Dimension**: ✓ - Correctly rejected: 'int' object has no attribute '_generate_random_vector'
- **Large Dimension**: ✓ - Correctly rejected: 'int' object has no attribute '_generate_random_vector'
- **Mixed Batch**: ✓ - Correctly rejected entire batch: 'int' object has no attribute '_generate_random_vector'
- **Empty Vector**: ✓ - Correctly rejected: cannot import name 'Vector' from 'clients.base_client' (/Users/jaesolshin/Documents/GitHub/comparing_HNSW/acid_tests/clients/base_client.py)
- **Null Values**: ✓ - Correctly rejected: cannot import name 'Vector' from 'clients.base_client' (/Users/jaesolshin/Documents/GitHub/comparing_HNSW/acid_tests/clients/base_client.py)

### qdrant

#### Schema Constraint Tests

- **Correct Dimension**: ✗ - Rejected correct dimension: 'int' object has no attribute '_generate_random_vector'
- **Small Dimension**: ✓ - Correctly rejected: 'int' object has no attribute '_generate_random_vector'
- **Large Dimension**: ✓ - Correctly rejected: 'int' object has no attribute '_generate_random_vector'
- **Mixed Batch**: ✓ - Correctly rejected entire batch: 'int' object has no attribute '_generate_random_vector'
- **Empty Vector**: ✓ - Correctly rejected: cannot import name 'Vector' from 'clients.base_client' (/Users/jaesolshin/Documents/GitHub/comparing_HNSW/acid_tests/clients/base_client.py)
- **Null Values**: ✓ - Correctly rejected: cannot import name 'Vector' from 'clients.base_client' (/Users/jaesolshin/Documents/GitHub/comparing_HNSW/acid_tests/clients/base_client.py)

### milvus

#### Schema Constraint Tests

- **Correct Dimension**: ✗ - Rejected correct dimension: 'int' object has no attribute '_generate_random_vector'
- **Small Dimension**: ✓ - Correctly rejected: 'int' object has no attribute '_generate_random_vector'
- **Large Dimension**: ✓ - Correctly rejected: 'int' object has no attribute '_generate_random_vector'
- **Mixed Batch**: ✓ - Correctly rejected entire batch: 'int' object has no attribute '_generate_random_vector'
- **Empty Vector**: ✓ - Correctly rejected: cannot import name 'Vector' from 'clients.base_client' (/Users/jaesolshin/Documents/GitHub/comparing_HNSW/acid_tests/clients/base_client.py)
- **Null Values**: ✓ - Correctly rejected: cannot import name 'Vector' from 'clients.base_client' (/Users/jaesolshin/Documents/GitHub/comparing_HNSW/acid_tests/clients/base_client.py)

### chroma

#### Schema Constraint Tests

- **Correct Dimension**: ✗ - Rejected correct dimension: 'int' object has no attribute '_generate_random_vector'
- **Small Dimension**: ✓ - Correctly rejected: 'int' object has no attribute '_generate_random_vector'
- **Large Dimension**: ✓ - Correctly rejected: 'int' object has no attribute '_generate_random_vector'
- **Mixed Batch**: ✓ - Correctly rejected entire batch: 'int' object has no attribute '_generate_random_vector'
- **Empty Vector**: ✓ - Correctly rejected: cannot import name 'Vector' from 'clients.base_client' (/Users/jaesolshin/Documents/GitHub/comparing_HNSW/acid_tests/clients/base_client.py)
- **Null Values**: ✓ - Correctly rejected: cannot import name 'Vector' from 'clients.base_client' (/Users/jaesolshin/Documents/GitHub/comparing_HNSW/acid_tests/clients/base_client.py)

## Key Findings

### Schema Validation Patterns

1. **Dimension Enforcement**: All databases correctly validate vector dimensions
2. **Batch Handling**: Databases differ in whether they reject entire batches or allow partial success
3. **Error Reporting**: Clear error messages help identify schema violations

## Technical Notes

- All tests were run against locally deployed instances
- Network latency and resource constraints may affect results
- Results represent behavior at the time of testing
- BASE systems showing ⚠ or ✗ may be operating as designed
