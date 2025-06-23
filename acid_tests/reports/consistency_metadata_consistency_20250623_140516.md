# Consistency Test Report - Metadata Consistency
Generated: 2025-06-23 14:05:16

## Overview
This report presents the results of consistency tests across multiple vector databases.
Test notation: ✓ = no inconsistency observed, ⚠ = partial inconsistency or delay, ✗ = unstable or broken behavior

## Summary

| Database | Overall Status | Passed | Partial | Failed | Details |
|----------|---------------|---------|---------|---------|----------|
| pgvector | ✗ | 0 | 0 | 4 | 0/4 tests passed |
| qdrant | ✗ | 0 | 0 | 4 | 0/4 tests passed |
| milvus | ⚠ | 3 | 0 | 1 | 3/4 tests passed |
| chroma | ✗ | 1 | 0 | 3 | 1/4 tests passed |

## Detailed Results

### pgvector

#### Metadata Consistency Tests

- **Single Update**: ✗ - Metadata update failed or inconsistent
- **Concurrent Updates**: ✗ - Only 0/10 vectors have consistent metadata
- **Search Metadata**: ✗ - Metadata not preserved in search results
- **Type Consistency**: ✗ - Only 0/5 metadata types preserved
  - Type preservation:
    - string: ✗
    - integer: ✗
    - float: ✗
    - boolean: ✗
    - nested: ✗

### qdrant

#### Metadata Consistency Tests

- **Single Update**: ✗ - Metadata update failed or inconsistent
- **Concurrent Updates**: ✗ - Only 0/10 vectors have consistent metadata
- **Search Metadata**: ✗ - Metadata not preserved in search results
- **Type Consistency**: ✗ - Only 0/5 metadata types preserved
  - Type preservation:
    - string: ✗
    - integer: ✗
    - float: ✗
    - boolean: ✗
    - nested: ✗

### milvus

#### Metadata Consistency Tests

- **Single Update**: ✓ - Metadata update successful and consistent
- **Concurrent Updates**: ✗ - Only 1/10 vectors have consistent metadata
- **Search Metadata**: ✓ - All metadata preserved in search results
- **Type Consistency**: ✓ - All metadata types preserved correctly
  - Type preservation:
    - string: ✓
    - integer: ✓
    - float: ✓
    - boolean: ✓
    - nested: ✓

### chroma

#### Metadata Consistency Tests

- **Single Update**: ✗ - Metadata update failed or inconsistent
- **Concurrent Updates**: ✗ - Only 0/10 vectors have consistent metadata
- **Search Metadata**: ✓ - All metadata preserved in search results
- **Type Consistency**: ✗ - Only 0/5 metadata types preserved
  - Type preservation:
    - string: ✗
    - integer: ✗
    - float: ✗
    - boolean: ✗
    - nested: ✗

## Key Findings

### Metadata Handling Patterns

1. **Update Support**: Not all databases support direct metadata updates
2. **Concurrent Safety**: Concurrent metadata updates may lead to inconsistencies
3. **Type Preservation**: Data type handling varies across databases

## Technical Notes

- All tests were run against locally deployed instances
- Network latency and resource constraints may affect results
- Results represent behavior at the time of testing
- BASE systems showing ⚠ or ✗ may be operating as designed
