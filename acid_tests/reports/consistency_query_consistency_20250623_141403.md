# Consistency Test Report - Query Consistency
Generated: 2025-06-23 14:14:03

## Overview
This report presents the results of consistency tests across multiple vector databases.
Test notation: ✓ = no inconsistency observed, ⚠ = partial inconsistency or delay, ✗ = unstable or broken behavior

## Summary

| Database | Overall Status | Passed | Partial | Failed | Details |
|----------|---------------|---------|---------|---------|----------|
| pgvector | ⚠ | 1 | 0 | 1 | 1/2 tests passed |
| qdrant | ⚠ | 1 | 0 | 1 | 1/2 tests passed |
| milvus | ✓ | 4 | 0 | 0 | 4/4 tests passed |
| chroma | ⚠ | 3 | 1 | 0 | 3/4 tests passed |

## Detailed Results

### pgvector

#### Query Consistency Tests

- **Read After Write**: ✗ - Only 0/10 exact matches
- **Stale Reads**: ✓ - No stale reads detected after deletion

### qdrant

#### Query Consistency Tests

- **Read After Write**: ✗ - Only 0/10 exact matches
- **Stale Reads**: ✓ - No stale reads detected after deletion

### milvus

#### Query Consistency Tests

- **Read After Write**: ✓ - Perfect read-after-write consistency
- **Query During Updates**: ✓ - Stable results during updates (avg 8.6/10 overlap)
- **Concurrent Reads**: ✓ - All concurrent reads returned identical results
- **Stale Reads**: ✓ - No stale reads detected after deletion

### chroma

#### Query Consistency Tests

- **Read After Write**: ✓ - Perfect read-after-write consistency
- **Query During Updates**: ⚠ - Moderate stability (avg 7.7/10 overlap, 2 order changes)
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
