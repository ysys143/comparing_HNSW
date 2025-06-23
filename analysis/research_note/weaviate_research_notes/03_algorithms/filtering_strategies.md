# Weaviate Filtering Strategies Analysis

## Overview
Weaviate implements a sophisticated filtering system that uses roaring bitmaps and an adaptive approach to balance between pre-filtering and post-filtering based on filter selectivity.

## Filtering Approach

### 1. **Adaptive Filtering Strategy**
- Dynamically chooses between pre-filtering and post-filtering
- Uses filter cost estimation to make decisions
- Leverages roaring bitmaps for efficient set operations

### 2. **Where Filter Syntax**
```graphql
# GraphQL query with filters
{
  Get {
    Article(
      where: {
        operator: And
        operands: [{
          path: ["category"]
          operator: Equal
          valueText: "science"
        }, {
          path: ["wordCount"]
          operator: GreaterThan
          valueInt: 1000
        }]
      }
      nearVector: {
        vector: [1.0, 2.0, 3.0]
        certainty: 0.7
      }
      limit: 10
    ) {
      title
      _additional {
        distance
      }
    }
  }
}
```

### 3. **Implementation Architecture**

#### Filter Resolution Pipeline
```go
// adapters/repos/db/vector/hnsw/search.go
func (h *hnsw) searchWithFilter(ctx context.Context, 
    vector []float32, limit int, filter *filters.LocalFilter) ([]uint64, error) {
    
    // Resolve filter to bitmap
    allowList, err := h.resolver.resolve(ctx, filter)
    if err != nil {
        return nil, err
    }
    
    // Choose strategy based on selectivity
    selectivity := float64(allowList.GetCardinality()) / float64(h.getNodeCount())
    
    if selectivity < 0.01 {
        // Use exact search for very selective filters
        return h.exactSearchWithFilter(vector, limit, allowList)
    }
    
    // Use HNSW with filtering
    return h.hnswSearchWithFilter(vector, limit, allowList)
}
```

#### Roaring Bitmap Usage
```go
// entities/filters/filters.go
type AllowList struct {
    bitmap *roaring.Bitmap
}

func (al *AllowList) Contains(id uint64) bool {
    return al.bitmap.Contains(uint32(id))
}

func (al *AllowList) Intersection(other *AllowList) *AllowList {
    return &AllowList{
        bitmap: roaring.And(al.bitmap, other.bitmap),
    }
}
```

### 4. **Filter Types and Operators**

#### Supported Operators
```go
const (
    OperatorEqual            Operator = "Equal"
    OperatorNotEqual         Operator = "NotEqual"
    OperatorGreaterThan      Operator = "GreaterThan"
    OperatorGreaterThanEqual Operator = "GreaterThanEqual"
    OperatorLessThan         Operator = "LessThan"
    OperatorLessThanEqual    Operator = "LessThanEqual"
    OperatorAnd              Operator = "And"
    OperatorOr               Operator = "Or"
    OperatorNot              Operator = "Not"
    OperatorWithinGeoRange   Operator = "WithinGeoRange"
    OperatorContainsAny      Operator = "ContainsAny"
    OperatorContainsAll      Operator = "ContainsAll"
)
```

#### Complex Filter Example
```graphql
where: {
  operator: And
  operands: [
    {
      operator: Or
      operands: [
        {
          path: ["category"]
          operator: Equal
          valueText: "technology"
        },
        {
          path: ["category"]
          operator: Equal
          valueText: "science"
        }
      ]
    },
    {
      path: ["publicationDate"]
      operator: GreaterThan
      valueDate: "2023-01-01T00:00:00Z"
    },
    {
      operator: Not
      operands: [{
        path: ["status"]
        operator: Equal
        valueText: "draft"
      }]
    }
  ]
}
```

### 5. **Indexing for Filters**

#### Inverted Index
```go
// adapters/repos/db/inverted/index.go
type Index struct {
    Property string
    Type     IndexType
    Storage  Storage
}

// Different index types for different data types
type IndexType int
const (
    IndexTypeString IndexType = iota
    IndexTypeStringArray
    IndexTypeInt
    IndexTypeIntArray
    IndexTypeNumber
    IndexTypeNumberArray
    IndexTypeBoolean
    IndexTypeGeo
)
```

#### Property-Specific Indices
```go
// Schema configuration for indexing
{
    "class": "Article",
    "properties": [{
        "name": "category",
        "dataType": ["string"],
        "indexFilterable": true,
        "indexSearchable": false
    }, {
        "name": "publicationDate",
        "dataType": ["date"],
        "indexFilterable": true,
        "indexRangeFilters": true
    }]
}
```

### 6. **Performance Optimizations**

#### Lazy Evaluation
```go
// Filters are evaluated lazily during HNSW traversal
func (h *hnsw) greedy_search_layer(query []float32, ep *vertex, 
    ef int, layer int, allowList *roaring.Bitmap) *distanceHeap {
    
    candidates := &distanceHeap{}
    w := &distanceHeap{}
    visited := make(map[uint64]bool)
    
    for candidates.Len() > 0 {
        current := heap.Pop(candidates).(*distanceNode)
        
        if current.distance > w.Peek().distance {
            break
        }
        
        for _, neighborID := range h.getNeighbors(current.id, layer) {
            if visited[neighborID] {
                continue
            }
            visited[neighborID] = true
            
            // Check filter before distance calculation
            if allowList != nil && !allowList.Contains(neighborID) {
                continue
            }
            
            distance := h.distanceFunc(query, h.getVector(neighborID))
            // Process valid neighbor...
        }
    }
    
    return w
}
```

#### Parallel Filter Resolution
```go
// Resolve complex filters in parallel
func (r *Resolver) resolveCompound(ctx context.Context, 
    filter *filters.LocalFilter) (*roaring.Bitmap, error) {
    
    results := make(chan *roaring.Bitmap, len(filter.Children))
    errors := make(chan error, len(filter.Children))
    
    for _, child := range filter.Children {
        go func(f *filters.LocalFilter) {
            bitmap, err := r.resolve(ctx, f)
            if err != nil {
                errors <- err
                return
            }
            results <- bitmap
        }(child)
    }
    
    // Combine results based on operator
    return r.combineResults(filter.Operator, results, errors)
}
```

### 7. **Configuration and Tuning**

#### Vector Index Configuration
```json
{
  "vectorIndexConfig": {
    "ef": 100,
    "efConstruction": 128,
    "maxConnections": 32,
    "vectorCacheMaxObjects": 1000000,
    "flatSearchCutoff": 40000,
    "dynamicEfMin": 100,
    "dynamicEfMax": 500,
    "dynamicEfFactor": 8,
    "skip": false,
    "pq": {
      "enabled": false,
      "segments": 0,
      "centroids": 256
    }
  }
}
```

#### Filter-Specific Settings
```json
{
  "invertedIndexConfig": {
    "bm25": {
      "b": 0.75,
      "k1": 1.2
    },
    "stopwords": {
      "preset": "en",
      "additions": [],
      "removals": []
    },
    "indexTimestamps": true,
    "indexNullState": true,
    "indexPropertyLength": true
  }
}
```

### 8. **Performance Characteristics**

#### Advantages
- Efficient bitmap operations for complex boolean filters
- Adaptive strategy prevents worst-case scenarios
- Good balance between memory usage and query speed
- Supports diverse data types and operators

#### Trade-offs
- Bitmap maintenance overhead during updates
- Memory usage for roaring bitmaps can be significant
- Filter resolution time for complex queries

## Code References

### Core Implementation
- `adapters/repos/db/vector/hnsw/` - HNSW with filtering
- `adapters/repos/db/filters/` - Filter resolution logic
- `adapters/repos/db/inverted/` - Inverted index implementation
- `entities/filters/` - Filter types and definitions

### Key Innovations
1. **Roaring Bitmaps**: Efficient compressed bitmap implementation
2. **Adaptive Strategy**: Dynamic selection based on selectivity
3. **Lazy Evaluation**: Filters checked only when necessary during traversal

## Comparison Notes
- More sophisticated than simple post-filtering approaches
- Less complex than Qdrant's filterable HNSW but still efficient
- Good balance between implementation complexity and performance
- Well-suited for applications with diverse filtering requirements