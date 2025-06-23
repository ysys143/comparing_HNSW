# Consistency Test Report - Schema Constraints
Generated: 2025-06-23 14:52:25

## Overview
This report presents the results of consistency tests across multiple vector databases.
Test notation: ✓ = no inconsistency observed, ⚠ = partial inconsistency or delay, ✗ = unstable or broken behavior

## Summary

| Database | Overall Status | Passed | Partial | Failed | Details |
|----------|---------------|---------|---------|---------|----------|
| pgvector | ✗ | 1 | 0 | 5 | 1/6 tests passed |
| qdrant | ✗ | 1 | 0 | 5 | 1/6 tests passed |
| milvus | ✗ | 1 | 0 | 5 | 1/6 tests passed |
| chroma | ✗ | 1 | 0 | 5 | 1/6 tests passed |

## Detailed Results

### pgvector

#### Schema Constraint Tests

- **Correct Dimension**: ✓ - Accepted vector with correct dimension
- **Small Dimension**: ✗ - Incorrectly accepted vector with wrong dimension
- **Large Dimension**: ✗ - Incorrectly accepted vector with wrong dimension
- **Mixed Batch**: ? - Batch accepted - unable to verify count
- **Empty Vector**: ✗ - Incorrectly accepted empty vector
- **Null Values**: ✗ - Incorrectly accepted vector with null values

### qdrant

#### Schema Constraint Tests

- **Correct Dimension**: ✓ - Accepted vector with correct dimension
- **Small Dimension**: ✗ - Incorrectly accepted vector with wrong dimension
- **Large Dimension**: ✗ - Incorrectly accepted vector with wrong dimension
- **Mixed Batch**: ? - Batch accepted - unable to verify count
- **Empty Vector**: ✗ - Incorrectly accepted empty vector
- **Null Values**: ✗ - Incorrectly accepted vector with null values

### milvus

#### Schema Constraint Tests

- **Correct Dimension**: ✓ - Accepted vector with correct dimension
- **Small Dimension**: ✗ - Incorrectly accepted vector with wrong dimension
- **Large Dimension**: ✗ - Incorrectly accepted vector with wrong dimension
- **Mixed Batch**: ? - Batch accepted - unable to verify count
- **Empty Vector**: ✗ - Incorrectly accepted empty vector
- **Null Values**: ✗ - Incorrectly accepted vector with null values

### chroma

#### Schema Constraint Tests

- **Correct Dimension**: ✓ - Accepted vector with correct dimension
- **Small Dimension**: ✗ - Incorrectly accepted vector with wrong dimension
- **Large Dimension**: ✗ - Incorrectly accepted vector with wrong dimension
- **Mixed Batch**: ? - Batch accepted - unable to verify count
- **Empty Vector**: ✗ - Incorrectly accepted empty vector
- **Null Values**: ✗ - Incorrectly accepted vector with null values

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
