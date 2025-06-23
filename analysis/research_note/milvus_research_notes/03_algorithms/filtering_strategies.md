# Milvus Filtering Strategies Analysis

## Overview
Milvus implements a sophisticated filtering system that combines scalar filtering with vector search, supporting both attribute filtering and advanced indexing strategies.

## Filtering Approach

### 1. **Scalar Index Integration**
- Builds dedicated scalar indices for filterable fields
- Supports multiple index types: Inverted Index, Bitmap Index, STL sort
- Scalar indices are used to pre-filter candidates before vector search

### 2. **Expression-Based Filtering**
```python
# Milvus filter expression example
expr = "category == 'science' and year >= 2020 and price < 100.0"
results = collection.search(
    data=[[1.0, 2.0, 3.0]], 
    anns_field="embedding",
    param=search_params,
    limit=10,
    expr=expr
)
```

### 3. **Implementation Architecture**

#### Segmented Storage
- Data is organized into segments
- Each segment maintains both vector and scalar indices
- Parallel filtering across segments

#### Filter Execution Pipeline
```go
// internal/querynode/search.go
type searchTask struct {
    vectors     []float32
    expressions string
    topK        int
    // Filter expressions are parsed and optimized
}
```

### 4. **Index Structures for Filtering**

#### Inverted Index
- Used for equality and range queries
- Efficient for high-cardinality fields
- Location: `internal/core/src/index/InvertedIndex.cpp`

#### Bitmap Index
- Optimized for low-cardinality fields
- Supports fast boolean operations
- Memory-efficient for categorical data

#### Hybrid Approach
```cpp
// Segment-level filtering
class SegmentInterface {
    // Combines scalar and vector indices
    virtual Status Search(const query::Plan* plan,
                         const PlaceholderGroup& ph_group,
                         Timestamp timestamp,
                         SearchResult& results) = 0;
};
```

### 5. **Advanced Features**

#### Dynamic Filtering
- Supports time-travel queries with timestamp filtering
- Partition-based filtering for data organization
- Dynamic schema with flexible field addition

#### Expression Optimization
```python
# Complex expressions with optimization
expr = """
    (category in ['science', 'technology']) and 
    (year between 2020 and 2023) and
    (tags array_contains 'AI')
"""
```

### 6. **Performance Optimizations**

#### Segment Pruning
- Skip entire segments based on metadata
- Min/max statistics for range queries
- Bloom filters for existence checks

#### Parallel Execution
- Concurrent filtering across segments
- SIMD optimization for bitmap operations
- Cache-friendly data layouts

#### Adaptive Strategy
```go
// Chooses between different strategies based on selectivity
func (s *searchTask) ChooseFilterStrategy() FilterStrategy {
    if s.estimateSelectivity() < 0.1 {
        return PreFilterStrategy
    }
    return PostFilterStrategy
}
```

### 7. **Configuration and Tuning**

#### Index Configuration
```yaml
# milvus.yaml
queryNode:
  segcore:
    chunkRows: 1024
    enableDisk: true
  
# Collection schema with indices
fields:
  - name: category
    type: VARCHAR
    index:
      type: INVERTED
  - name: year
    type: INT64
    index:
      type: STL_SORT
```

#### Search Parameters
```python
search_params = {
    "metric_type": "L2",
    "params": {
        "nprobe": 10,
        "ef": 100,
        # Filter-aware parameters
        "round_decimal": -1,
        "search_list": 100
    }
}
```

### 8. **Performance Characteristics**

#### Advantages
- Excellent performance for complex filtering scenarios
- Scalable across distributed deployments
- Rich expression language with optimization

#### Trade-offs
- Additional storage overhead for scalar indices
- Index building time for large datasets
- Memory consumption for maintaining multiple indices

## Code References

### Core Filtering Logic
- `internal/core/src/query/PlanNode.h` - Query planning
- `internal/core/src/segcore/segment_c.cpp` - Segment interface
- `internal/querynode/segment.go` - Go bindings

### Expression Parsing
- `internal/parser/planparserv2/` - Expression parser
- `internal/core/src/query/Expr.h` - Expression representation

## Comparison Notes
- More comprehensive than simple metadata filtering
- Designed for large-scale deployments with billions of vectors
- Better suited for complex analytical queries than pure similarity search
- Trade-off: Higher complexity but superior performance at scale