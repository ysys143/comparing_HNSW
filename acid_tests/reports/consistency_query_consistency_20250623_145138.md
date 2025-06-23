# Consistency Test Report - Query Consistency
Generated: 2025-06-23 14:51:38

## Overview
This report presents the results of consistency tests across multiple vector databases.
Test notation: ✓ = no inconsistency observed, ⚠ = partial inconsistency or delay, ✗ = unstable or broken behavior

## Summary

| Database | Overall Status | Passed | Partial | Failed | Details |
|----------|---------------|---------|---------|---------|----------|
| pgvector | ⚠ | 2 | 0 | 1 | 2/3 tests passed |
| qdrant | ⚠ | 3 | 1 | 0 | 3/4 tests passed |
| milvus | ✓ | 4 | 0 | 0 | 4/4 tests passed |
| chroma | ⚠ | 3 | 1 | 0 | 3/4 tests passed |

## Detailed Results

### pgvector

#### Query Consistency Tests

- **Read After Write**: ✓ - Perfect read-after-write consistency
- **Query During Updates**: ✗ - Unstable results (avg 1.0/10 overlap)
- **Stale Reads**: ✓ - No stale reads detected after deletion

### qdrant

#### Query Consistency Tests

- **Read After Write**: ✓ - Perfect read-after-write consistency
- **Query During Updates**: ⚠ - Moderate stability (avg 7.3/10 overlap, 1 order changes)
- **Concurrent Reads**: ✓ - All concurrent reads returned identical results
- **Stale Reads**: ✓ - No stale reads detected after deletion

### milvus

#### Query Consistency Tests

- **Read After Write**: ✓ - Perfect read-after-write consistency
- **Query During Updates**: ✓ - Stable results during updates (avg 9.1/10 overlap)
- **Concurrent Reads**: ✓ - All concurrent reads returned identical results
- **Stale Reads**: ✓ - No stale reads detected after deletion

### chroma

#### Query Consistency Tests

- **Read After Write**: ✓ - Perfect read-after-write consistency
- **Query During Updates**: ⚠ - Moderate stability (avg 7.1/10 overlap, 3 order changes)
- **Concurrent Reads**: ✓ - All concurrent reads returned identical results
- **Stale Reads**: ✓ - No stale reads detected after deletion

## Key Findings

### Query Consistency Patterns

1. **Read-After-Write**: Most databases show good read-after-write consistency
2. **Update Impact**: Ongoing updates can cause transient inconsistencies
3. **Concurrent Reads**: Some variation in concurrent read results observed

## Technical Notes

- All tests were run against locally deployed instances
- Network latency and resource constraints may affect results
- Results represent behavior at the time of testing
- BASE systems showing ⚠ or ✗ may be operating as designed
