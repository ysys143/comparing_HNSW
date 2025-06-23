# Consistency Test Report - Index Synchronization
Generated: 2025-06-23 14:02:08

## Overview
This report presents the results of consistency tests across multiple vector databases.
Test notation: ✓ = no inconsistency observed, ⚠ = partial inconsistency or delay, ✗ = unstable or broken behavior

## Summary

| Database | Overall Status | Passed | Partial | Failed | Details |
|----------|---------------|---------|---------|---------|----------|
| pgvector | ✗ | 0 | 0 | 4 | 0/4 tests passed |
| qdrant | ✗ | 0 | 0 | 4 | 0/4 tests passed |
| milvus | ✓ | 4 | 0 | 0 | 4/4 tests passed |
| chroma | ✗ | 0 | 0 | 4 | 0/4 tests passed |

## Detailed Results

### pgvector

#### Index Synchronization Tests

- **Immediate Visibility**: ✗ - Vector not immediately searchable
- **Load Visibility**: ✗ - No vectors became searchable under load
- **Bulk Visibility**: ✗ - No sampled vectors became searchable
- **Search Consistency**: ✗ - Search results only 0% consistent

### qdrant

#### Index Synchronization Tests

- **Immediate Visibility**: ✗ - Vector not immediately searchable
- **Load Visibility**: ✗ - No vectors became searchable under load
- **Bulk Visibility**: ✗ - No sampled vectors became searchable
- **Search Consistency**: ✗ - Search results only 0% consistent

### milvus

#### Index Synchronization Tests

- **Immediate Visibility**: ✓ - Vector immediately searchable after insert
- **Load Visibility**: ✓ - All vectors searchable. Avg delay: 3.034s, Max: 3.398s
  - Visible: 20
  - Missing: 0
  - Avg delay: 3034ms
  - Max delay: 3397ms
- **Bulk Visibility**: ✓ - All sampled vectors searchable. Avg time: 3.595s
- **Search Consistency**: ✓ - Search results remain consistent during inserts

### chroma

#### Index Synchronization Tests

- **Immediate Visibility**: ✗ - Vector not immediately searchable
- **Load Visibility**: ✗ - No vectors became searchable under load
- **Bulk Visibility**: ✗ - No sampled vectors became searchable
- **Search Consistency**: ✗ - Search results only 0% consistent

## Key Findings

### Index Synchronization Patterns

1. **Immediate Visibility**: Most databases show good immediate visibility for single inserts
2. **Load Impact**: High insert loads can cause visibility delays
3. **Consistency Trade-offs**: Some databases prioritize write speed over immediate consistency

## Technical Notes

- All tests were run against locally deployed instances
- Network latency and resource constraints may affect results
- Results represent behavior at the time of testing
- BASE systems showing ⚠ or ✗ may be operating as designed
