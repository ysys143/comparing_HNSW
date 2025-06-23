# Vector Database Filtering Strategies Comparison

## Overview

This document analyzes filtering strategies implemented across 7 vector databases, examining how each system handles the challenge of combining vector similarity search with metadata filtering. The approaches range from simple post-filtering to sophisticated adaptive strategies.

## Filtering Approaches Classification

### 1. Pre-filtering (Filter-then-Search)
- Apply filters first, then search within filtered set
- Efficient for highly selective filters
- Risk of disconnected graph regions

### 2. Post-filtering (Search-then-Filter)
- Perform vector search first, then apply filters
- Simple implementation but potentially inefficient
- May require oversampling

### 3. Hybrid/Adaptive Approaches
- Dynamic selection between strategies
- Based on filter selectivity estimation
- Optimal performance across different scenarios

## System-by-System Analysis

### Qdrant: Advanced Hybrid Filtering

**Architecture**:
```rust
pub struct FilteredScorer<'a> {
    raw_scorer: Box<dyn RawScorer + 'a>,
    filter_context: Option<BoxCow<'a, dyn FilterContext + 'a>>,
    point_deleted: &'a BitSlice,
    vec_deleted: &'a BitSlice,
}
```

**Dynamic Strategy Selection**:
```rust
fn search_with_filter(&self, filter: &Filter) -> SearchResult {
    let cardinality = self.estimate_cardinality(filter);
    
    if cardinality.max < self.config.full_scan_threshold {
        // High selectivity: use plain search (pre-filtering)
        self.search_plain_filtered(...)
    } else if cardinality.min > self.config.full_scan_threshold {
        // Low selectivity: use HNSW with filtering (post-filtering)
        self.search_hnsw_filtered(...)
    } else {
        // Uncertain: use sampling to decide
        if self.sample_check_cardinality(filter) {
            self.search_hnsw_filtered(...)
        } else {
            self.search_plain_filtered(...)
        }
    }
}
```

**Cardinality Estimation**:
```rust
pub struct CardinalityEstimation {
    pub primary_clauses: Vec<PrimaryCondition>,
    pub min: usize,  // Best case minimum
    pub exp: usize,  // Expected value
    pub max: usize,  // Worst case maximum
}

// Complex boolean query estimation
fn estimate_cardinality(&self, filter: &Filter) -> CardinalityEstimation {
    match filter {
        Filter::And(filters) => {
            // Multiply selectivities
        }
        Filter::Or(filters) => {
            // Add with overlap estimation
        }
        Filter::Not(filter) => {
            // Inverse selectivity
        }
    }
}
```

**Key Features**:
- Statistical sampling for uncertain cases
- Payload-based subgraph building for common filters
- Integrated filtering during graph traversal

### Weaviate: Multi-Strategy Approach

**Three Filtering Strategies**:

1. **SWEEPING** (Default):
```go
func (h *hnsw) searchLayerByVectorWithSweeping(
    vector []float32,
    entryPointID uint64,
    limit int,
    layer int,
    filter helpers.AllowList,
) ([]*priorityqueue.Item, error) {
    candidates := h.searchLayer(vector, entryPointID, limit, layer)
    
    // Post-filter results
    filtered := make([]*priorityqueue.Item, 0, len(candidates))
    for _, candidate := range candidates {
        if filter.Contains(candidate.ID) {
            filtered = append(filtered, candidate)
        }
    }
    return filtered, nil
}
```

2. **ACORN** (Adaptive Cost-Optimized Refined Navigation):
```go
func (h *hnsw) searchLayerByVectorWithAcorn(
    vector []float32,
    entryPointID uint64,
    limit int,
    layer int,
    filter helpers.AllowList,
) ([]*priorityqueue.Item, error) {
    // Multi-hop neighborhood expansion
    maxHops := 2
    if filter.Ratio() < 0.1 {
        maxHops = 3  // More aggressive expansion for selective filters
    }
    
    candidates := h.expandNeighborhood(vector, entryPointID, maxHops, filter)
    return h.selectBest(candidates, limit), nil
}
```

3. **RRE** (Reduced Redundant Expansion):
```go
// Only apply filter at layer 0
if layer == 0 && filter != nil {
    // Apply filter
} else {
    // Skip filtering at higher layers
}
```

**Strategy Selection**:
```go
func (h *hnsw) selectFilterStrategy(filter helpers.AllowList) FilterStrategy {
    if filter == nil {
        return NoFilter
    }
    
    ratio := filter.Ratio()
    if ratio < h.acornAlpha {
        return Acorn  // Very selective filter
    } else if ratio > h.acornBeta {
        return Sweeping  // Non-selective filter
    } else {
        return RRE  // Medium selectivity
    }
}
```

### pgvector: PostgreSQL Integration

**Simple Post-filtering**:
```c
// HNSW scan returns candidates
static bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
    HnswScanState *scanstate = (HnswScanState *) scan->opaque;
    
    // Get next candidate from HNSW
    if (scanstate->currentCandidate < scanstate->numCandidates) {
        ItemPointer tid = &scanstate->candidates[scanstate->currentCandidate++];
        scan->xs_heaptid = *tid;
        
        // PostgreSQL handles filtering via WHERE clause
        return true;
    }
    
    return false;
}
```

**Integration with Query Planner**:
```sql
-- PostgreSQL automatically handles filtering
SELECT * FROM items 
WHERE category = 'electronics' 
  AND price < 1000
ORDER BY embedding <-> '[1,2,3]'
LIMIT 10;
```

### Vespa: Global Filter Architecture

**Pre-computed Global Filter**:
```cpp
class GlobalFilter {
    BitVector::UP _bit_vector;
    uint32_t _doc_count;
    
public:
    bool check(uint32_t docid) const {
        return _bit_vector && _bit_vector->testBit(docid);
    }
    
    double hit_ratio() const {
        return static_cast<double>(_bit_vector->countTrueBits()) / _doc_count;
    }
};
```

**Adaptive Search Strategy**:
```cpp
void NearestNeighborBlueprint::set_global_filter(const GlobalFilter& filter) {
    double hit_ratio = filter.hit_ratio();
    
    if (hit_ratio < _global_filter_lower_limit) {
        // Very selective: use exact search
        _algorithm = ExactSearch;
    } else if (hit_ratio > _global_filter_upper_limit) {
        // Non-selective: use index with post-filtering
        _algorithm = ApproximateSearch;
    } else {
        // Medium selectivity: adjust target hits
        _adjusted_target_hits = _target_hits / hit_ratio;
        _algorithm = ApproximateSearch;
    }
}
```

### Elasticsearch: Lucene-based Filtering

**Query-time Integration**:
```java
// Integrated with Lucene's query framework
public class KnnVectorQuery extends Query {
    private final Query filterQuery;
    
    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode) {
        Weight filterWeight = filterQuery.createWeight(searcher, ScoreMode.COMPLETE_NO_SCORES);
        return new KnnVectorWeight(this, filterWeight);
    }
}
```

### Milvus: Segment-based Filtering

**Bitmap-based Filtering**:
```go
func (s *Segment) Search(searchReq *SearchRequest) (*SearchResult, error) {
    // Build filter bitmap
    filterBitmap := s.buildFilterBitmap(searchReq.Filter)
    
    // Pass to Knowhere
    searchParams["filter_bitmap"] = filterBitmap
    
    return s.vectorIndex.Search(searchReq.Vectors, searchParams)
}
```

### Chroma: Adaptive Filtering

Chroma implements an adaptive filtering strategy that switches between pre-filtering and post-filtering based on the selectivity of the metadata filter.

**Adaptive Strategy**:
- **Pre-filtering**: For highly selective filters, Chroma first resolves the allowed document IDs from its metadata store (SQLite/DuckDB) and passes this allow-list directly to the underlying `hnswlib` for an efficient, filtered vector search.
- **Post-filtering**: For less selective filters, it performs a broader vector search with over-fetching and then filters the results.

```rust
// Simplified conceptual implementation from research notes
impl HnswIndex {
    pub fn search_with_prefilter(&self, query: &[f32], k: usize, allowed_ids: &[usize]) -> Result<Vec<(usize, f32)>> {
        // Heuristic to decide strategy
        if allowed_ids.len() < k * 2 {
            // Selective filter: Use hnswlib's built-in filtering
            self.index.query(query, k, allowed_ids, &[])
        } else {
            // Non-selective: Oversample and post-filter
            let candidates = self.index.query(query, k * 3, &[], &[])?;
            let allowed_set: HashSet<_> = allowed_ids.iter().cloned().collect();
            
            Ok(candidates.into_iter()
                .filter(|(id, _)| allowed_set.contains(id))
                .take(k)
                .collect())
        }
    }
}
```

## Cardinality Estimation Comparison

| System | Estimation Method | Accuracy | Overhead |
|--------|------------------|----------|----------|
| **Qdrant** | Statistical with sampling | High | Medium |
| **Weaviate** | Simple ratio calculation | Medium | Low |
| **Vespa** | Exact pre-computation (BitVector) | Perfect | High |
| **Elasticsearch** | Exact pre-computation (BitSet) | Perfect | High |
| **pgvector** | PostgreSQL statistics | High | Low |
| **Chroma** | Exact pre-computation (ID set size heuristic) | Perfect | Medium |
| **Milvus** | Basic (Bitmap stats) | Low | Low |

## Performance Characteristics

### Pre-filtering Performance
```