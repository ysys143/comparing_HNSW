# Consistency Test Report - Index Synchronization
Generated: 2025-06-23 14:59:30

## Overview
This report presents the results of consistency tests across multiple vector databases.
Test notation: ✓ = no inconsistency observed, ⚠ = partial inconsistency or delay, ✗ = unstable or broken behavior

## Summary

| Database | Overall Status | Passed | Partial | Failed | Details |
|----------|---------------|---------|---------|---------|----------|
| pgvector | ✗ | 1 | 0 | 3 | 1/4 tests passed |
| qdrant | ⚠ | 2 | 0 | 2 | 2/4 tests passed |
| milvus | ✓ | 4 | 0 | 0 | 4/4 tests passed |
| chroma | ✓ | 4 | 0 | 0 | 4/4 tests passed |

## Detailed Results

### pgvector

#### Index Synchronization Tests

- **Immediate Visibility**: ✗ - Vector not immediately searchable
- **Load Visibility**: ✗ - 19/20 vectors not searchable
  - Visible: 1
  - Missing: 19
  - Avg delay: 3ms
  - Max delay: 3ms
- **Bulk Visibility**: ✓ - All sampled vectors searchable. Avg time: 0.062s
- **Search Consistency**: ✗ - Search results only 0% consistent

### qdrant

#### Index Synchronization Tests

- **Immediate Visibility**: ✗ - Vector not immediately searchable
- **Load Visibility**: ✓ - All vectors searchable. Avg delay: 0.002s, Max: 0.003s
  - Visible: 20
  - Missing: 0
  - Avg delay: 1ms
  - Max delay: 3ms
- **Bulk Visibility**: ✓ - All sampled vectors searchable. Avg time: 0.035s
- **Search Consistency**: ✗ - Search results only 0% consistent

### milvus

#### Index Synchronization Tests

- **Immediate Visibility**: ✓ - Vector immediately searchable after insert
- **Load Visibility**: ✓ - All vectors searchable. Avg delay: 3.040s, Max: 3.402s
  - Visible: 20
  - Missing: 0
  - Avg delay: 3039ms
  - Max delay: 3402ms
- **Bulk Visibility**: ✓ - All sampled vectors searchable. Avg time: 3.202s
- **Search Consistency**: ✓ - Search results remain consistent during inserts

### chroma

#### Index Synchronization Tests

- **Immediate Visibility**: ✓ - Vector immediately searchable after insert
- **Load Visibility**: ✓ - All vectors searchable. Avg delay: 0.006s, Max: 0.009s
  - Visible: 20
  - Missing: 0
  - Avg delay: 6ms
  - Max delay: 8ms
- **Bulk Visibility**: ✓ - All sampled vectors searchable. Avg time: 0.026s
- **Search Consistency**: ✓ - Search results remain consistent during inserts

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
