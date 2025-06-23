# Vespa Filtering Strategies Analysis

## Overview
Vespa implements a sophisticated filtering system that combines its powerful search engine capabilities with vector search, offering multiple strategies for different use cases.

## Filtering Approach

### 1. **Multi-Strategy System**
- **Pre-filtering**: Build allow-list before HNSW search
- **Post-filtering**: Filter after retrieving candidates
- **Hybrid approach**: Combination based on filter selectivity
- Automatic strategy selection based on query analysis

### 2. **YQL Query Language**
```yql
# Vespa Query Language example
select * from vectors where {
    targetHits: 10,
    nearestNeighbor: {
        field: "embedding",
        queryVector: [1.0, 2.0, 3.0],
        hnsw.exploreAdditionalHits: 100
    }
} and category contains "science" and price < 100.0
```

### 3. **Implementation Architecture**

#### Filter Evaluation Pipeline
```java
// searchlib/src/vespa/searchlib/queryeval/nearest_neighbor_iterator.cpp
class NearestNeighborIterator : public SearchIterator {
    void seek(uint32_t docId) override {
        if (_filter && !_filter->testBit(docId)) {
            return; // Skip filtered documents
        }
        // Continue with distance calculation
    }
};
```

#### Strategy Selection
```cpp
// Automatically choose strategy based on filter cost
FilterStrategy chooseStrategy(const Filter& filter, size_t totalDocs) {
    double selectivity = filter.estimate() / totalDocs;
    
    if (selectivity < 0.01) {
        return FilterStrategy::PRE_FILTER;
    } else if (selectivity > 0.5) {
        return FilterStrategy::POST_FILTER;
    } else {
        return FilterStrategy::HYBRID;
    }
}
```

### 4. **Filter Types and Indices**

#### Attribute-Based Filtering
```java
// Fast memory-based attribute filtering
public class AttributeFilter {
    // In-memory attribute storage for fast access
    private final AttributeVector attributeVector;
    
    public BitVector filter(String operator, Object value) {
        // Direct memory access for filtering
        return attributeVector.evaluate(operator, value);
    }
}
```

#### B-tree and Hash Indices
```xml
<!-- Schema definition with indices -->
<field name="category" type="string">
    <indexing>attribute | index</indexing>
    <attribute>
        <fast-search/>  <!-- Creates B-tree index -->
    </attribute>
</field>

<field name="tags" type="array<string>">
    <indexing>attribute</indexing>
    <attribute>
        <fast-search/>  <!-- Hash-based index for array -->
    </attribute>
</field>
```

### 5. **Advanced Features**

#### Weak AND Integration
```yql
# Combines filtering with Vespa's weakAnd operator
select * from vectors where {
    nearestNeighbor: {
        field: "embedding",
        queryVector: [1.0, 2.0, 3.0]
    }
} and (
    weakAnd(
        category contains "tech",
        tags contains "AI",
        description contains "machine learning"
    )
)
```

#### Grouping and Aggregation
```yql
# Filter with grouping
select * from vectors where {
    nearestNeighbor: {
        field: "embedding",
        queryVector: query_vector
    }
} and status = "active" |
all(
    group(category) each(
        max(10) output(count())
    )
)
```

### 6. **Performance Optimizations**

#### Bitvector Caching
```cpp
class BitvectorCache {
    // Cache frequently used filter results
    std::unordered_map<FilterKey, std::unique_ptr<BitVector>> cache;
    
    BitVector* getOrCompute(const Filter& filter) {
        auto key = filter.hash();
        if (cache.find(key) != cache.end()) {
            return cache[key].get();
        }
        // Compute and cache
        auto bitvector = computeBitvector(filter);
        cache[key] = std::move(bitvector);
        return cache[key].get();
    }
};
```

#### Approximate Filtering
```java
// When exact filtering is too expensive
public class ApproximateFilter {
    // Use Bloom filters for quick rejection
    private final BloomFilter bloomFilter;
    
    public boolean mightMatch(int docId) {
        return bloomFilter.mightContain(docId);
    }
}
```

### 7. **Configuration Options**

#### Search Definitions
```xml
<search>
    <diversity>
        <attribute>category</attribute>
        <min-groups>3</min-groups>
    </diversity>
    
    <rank-profile name="with_filter">
        <first-phase>
            <expression>
                if (attribute(category) == query(target_category), 
                    1.1 * closeness(field, embedding), 
                    closeness(field, embedding))
            </expression>
        </first-phase>
    </rank-profile>
</search>
```

#### Query Parameters
```json
{
    "yql": "select * from vectors where {...}",
    "ranking": {
        "profile": "with_filter",
        "matching": {
            "numThreadsPerSearch": 1,
            "minHitsPerThread": 100
        }
    },
    "approximate": {
        "hnsw.exploreAdditionalHits": 200,
        "targetHits": 100
    }
}
```

### 8. **Performance Characteristics**

#### Advantages
- Flexible strategy selection based on query patterns
- Excellent performance for both selective and non-selective filters
- Rich query language with complex boolean expressions
- Integrated with full-text search capabilities

#### Trade-offs
- Memory overhead for maintaining multiple index types
- Complexity in query planning and optimization
- May require tuning for optimal performance

## Code References

### Core Implementation
- `searchlib/src/vespa/searchlib/tensor/hnsw_index.cpp` - HNSW implementation
- `searchlib/src/vespa/searchlib/queryeval/filter_wrapper.cpp` - Filter wrapping
- `searchlib/src/vespa/searchlib/attribute/` - Attribute filtering

### Key Components
1. **Query Evaluation Tree**: Builds optimal execution plan
2. **Iterator Framework**: Lazy evaluation of filters
3. **Cost Model**: Estimates filter selectivity for planning

## Comparison Notes
- Most comprehensive filtering system among analyzed databases
- Benefits from Vespa's search engine heritage
- Excellent for complex queries combining text and vector search
- Trade-off: Higher complexity but maximum flexibility