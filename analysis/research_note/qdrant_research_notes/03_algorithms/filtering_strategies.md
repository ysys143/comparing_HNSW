# Qdrant Filtering Strategies Analysis

## Overview
Qdrant implements one of the most advanced filtering systems among vector databases, with its custom "Filterable HNSW" algorithm that integrates filtering directly into the graph traversal.

## Filtering Approach

### 1. **Filterable HNSW Algorithm**
- Custom modification of HNSW that evaluates filters during graph traversal
- Avoids the need to over-fetch and post-filter
- Maintains search efficiency even with highly selective filters

### 2. **Filter Types and Syntax**
```rust
// Rust API example
let filter = Filter::must([
    Condition::matches("category", "science".to_string()),
    Condition::range("year", Range {
        gte: Some(2020.0),
        lt: Some(2024.0),
        ..Default::default()
    }),
]);

// JSON API
{
    "filter": {
        "must": [
            {"key": "category", "match": {"value": "science"}},
            {"key": "year", "range": {"gte": 2020, "lt": 2024}}
        ]
    }
}
```

### 3. **Implementation Architecture**

#### Core Algorithm Modification
```rust
// lib/segment/src/index/hnsw_index/hnsw.rs
impl HnswIndex {
    fn search_filtered(
        &self,
        query: &[f32],
        filter: &Filter,
        top: usize,
        ef: usize,
    ) -> Vec<ScoredPoint> {
        // Custom search that checks filter during traversal
        let mut visited = FixedBitSet::with_capacity(self.points_count);
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();
        
        // Entry point selection considers filter
        let entry_point = self.get_filtered_entry_point(filter);
        
        // Modified graph traversal
        self.search_layer_filtered(
            query, 
            entry_point, 
            ef, 
            filter,
            &mut visited,
            &mut candidates,
            &mut w
        );
        
        w.into_sorted_vec()
    }
}
```

#### Filter Evaluation During Traversal
```rust
fn search_layer_filtered(
    &self,
    query: &[f32],
    entry_points: Vec<PointId>,
    ef: usize,
    filter: &Filter,
    visited: &mut FixedBitSet,
    candidates: &mut BinaryHeap<Reverse<ScoredPoint>>,
    w: &mut BinaryHeap<ScoredPoint>,
) {
    for point in entry_points {
        if !filter.check(point) {
            continue; // Skip filtered out points
        }
        // Process valid points
    }
    
    while let Some(current) = candidates.pop() {
        if current.score > w.peek().score {
            break;
        }
        
        for neighbor in self.get_neighbors(current.id) {
            if visited[neighbor] {
                continue;
            }
            visited.set(neighbor, true);
            
            // Check filter before computing distance
            if !filter.check(neighbor) {
                continue;
            }
            
            let distance = self.distance(query, neighbor);
            // Add to candidates...
        }
    }
}
```

### 4. **Filter Index Structures**

#### Payload Indices
```rust
// Multiple index types for different data patterns
pub enum PayloadIndex {
    Plain(PlainIndex),           // Hash-based, good for equality
    Struct(StructIndex),         // Nested field support  
    Integer(IntegerIndex),       // Range queries, numeric
    Keyword(KeywordIndex),       // Full-text search
    Geo(GeoIndex),              // Geographical queries
}
```

#### Optimized Field Storage
- Column-oriented storage for efficient filtering
- Mmap-based indices for large datasets
- Compressed representations for memory efficiency

### 5. **Advanced Filtering Features**

#### Nested Object Filtering
```json
{
    "filter": {
        "must": [{
            "key": "metadata.author.name",
            "match": {"value": "John Doe"}
        }]
    }
}
```

#### Geo-spatial Filtering
```json
{
    "filter": {
        "must": [{
            "key": "location",
            "geo_radius": {
                "center": {"lat": 52.52, "lon": 13.405},
                "radius": 1000.0
            }
        }]
    }
}
```

#### Full-text Search Integration
```json
{
    "filter": {
        "must": [{
            "key": "description",
            "match": {"text": "machine learning"}
        }]
    }
}
```

### 6. **Performance Optimizations**

#### Adaptive Strategy Selection
```rust
// Automatically choose between strategies
pub fn search_with_filter(&self, request: SearchRequest) -> SearchResult {
    let filter_cardinality = self.estimate_filter_cardinality(&request.filter);
    
    match filter_cardinality {
        n if n < 0.001 => self.exact_search_filtered(request),
        n if n < 0.1 => self.filtered_hnsw_search(request),
        _ => self.standard_hnsw_search(request),
    }
}
```

#### Query Planning
- Cardinality estimation for filters
- Cost-based optimization for complex filters
- Index selection based on query patterns

### 7. **Configuration and Tuning**

#### Collection Configuration
```json
{
    "vectors": {
        "size": 768,
        "distance": "Cosine"
    },
    "hnsw_config": {
        "m": 16,
        "ef_construct": 200,
        "full_scan_threshold": 10000,
        "max_indexing_threads": 0,
        "on_disk": false,
        "payload_m": 16  // Separate parameter for payload graph
    }
}
```

#### Index Configuration
```json
{
    "create_index": {
        "field_name": "category",
        "field_schema": "keyword",
        "options": {
            "type": "plain",
            "on_disk": false
        }
    }
}
```

### 8. **Performance Characteristics**

#### Advantages
- No performance degradation with selective filters
- Efficient memory usage through specialized indices
- Scales well with both data size and filter complexity

#### Trade-offs
- Additional memory for payload indices
- Slightly longer insertion time due to index maintenance
- Complexity in query planning and optimization

## Code References

### Core Implementation
- `lib/segment/src/index/hnsw_index/` - HNSW implementation
- `lib/segment/src/index/field_index/` - Field indices
- `lib/segment/src/types/` - Filter types and evaluation

### Key Innovations
1. **Filtered Entry Point Selection**: Starts traversal from points matching the filter
2. **Early Termination**: Stops exploring paths with no valid points
3. **Payload Graph**: Separate graph structure for efficient filtering

## Comparison Notes
- Most advanced filtering among analyzed databases
- Purpose-built for filtered vector search scenarios
- Better performance than post-filtering approaches
- Trade-off: Higher implementation complexity but superior query performance