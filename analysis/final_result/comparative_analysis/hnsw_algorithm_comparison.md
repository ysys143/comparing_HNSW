# HNSW Algorithm Implementation Comparison

## Executive Summary

This document provides a comprehensive comparison of HNSW implementations across 7 vector databases, analyzing their algorithmic choices, optimizations, and unique features. Each system has adapted the core HNSW algorithm to fit their specific architecture and use cases.

## Core Algorithm Variations

### Graph Construction Strategies

| System | Construction Approach | Unique Features | Implementation Details |
|--------|---------------------|-----------------|----------------------|
| **pgvector** | Two-phase (memory → disk) | Parallel build with shared memory coordination | InitBuildState → FlushPages, maintenance_work_mem aware |
| **Qdrant** | Incremental with GPU support | Payload-based subgraphs, graph healing | GraphLayersBuilder, points_count > SINGLE_THREADED_BUILD_THRESHOLD |
| **Vespa** | Two-phase prepare-commit | RCU for lock-free reads during construction | PreparedAddNode → complete_add_document |
| **Weaviate** | Single-phase with commit log | Batch operations, compression during build | commitlog.Log for durability, vertex-based storage |
| **Chroma** | Library wrapper (hnswlib) | Provider pattern with caching | HnswIndexProvider, thread pool management |
| **Elasticsearch** | Lucene-based | Segment-based incremental building | HnswGraphBuilder, OnHeapHnswGraph |
| **Milvus** | Knowhere library | CPU/GPU abstraction layer | hnswlib::HierarchicalNSW with OpenMP |

### Neighbor Selection Algorithms

#### Heuristic Selection Implementation

**pgvector**:
```c
// Simple pruning when connections exceed limit
neighbors = HnswPruneConnections(neighbors, m);
```

**Qdrant**:
```rust
// Sophisticated heuristic with distance-based pruning
fn select_neighbors_heuristic(&self, candidates: &[ScoredPoint], m: usize) {
    // Check if candidate improves connectivity
    for &existing in &result {
        if distance_to_existing < current.score {
            good = false;
        }
    }
}
```

**Vespa**:
```cpp
// Template-based with configurable heuristic
if (_cfg.heuristic_select_neighbors()) {
    return select_neighbors_heuristic(neighbors, max_links);
}
```

**Weaviate**:
```go
// Heuristic with candidate extension
if h.extendCandidates {
    // Consider neighbors of neighbors
}
```

### Search Algorithm Variations

#### Filter Integration Strategies

| System | Filter Strategy | Implementation |
|--------|----------------|----------------|
| **Chroma** | Adaptive pre/post | Selectivity-based strategy |
| **Elasticsearch** | Pre-filtering | Lucene BitSet integration |
| **Milvus** | Pre-filtering | Boolean expression pushdown |
| **pgvector** | Post-filtering | Standard PostgreSQL WHERE clause |
| **Qdrant** | Dynamic (pre/post/hybrid) | Cardinality estimation with sampling |
| **Vespa** | Global filter pushdown | Size-based clamping |
| **Weaviate** | Adaptive with flat fallback | Filter ratio threshold |

#### Early Termination Conditions

**pgvector**:
```c
if (lc->distance > GetFurthestDistance(w))
    break;
```

**Qdrant**:
```rust
if current.score > w.peek().unwrap().score {
    break;
}
```

**Vespa**:
```cpp
// Doom-based cancellation
if (doom.is_doomed()) {
    return partial_results;
}
```

## Memory Management Strategies

### Storage Layouts

| System | Node Storage | Link Storage | Memory Model | Specific Implementation |
|--------|-------------|--------------|--------------|------------------------|
| **pgvector** | PostgreSQL pages | Inline with nodes | Buffer cache | HnswElementData with FLEXIBLE_ARRAY_MEMBER |
| **Qdrant** | Separate vectors | Compressed/plain links | Arena allocator | Delta encoding, SmallMultiMap<PointOffsetType> |
| **Vespa** | RCU-protected | Array store | Generation-based | GenerationHandler with AtomicEntryRef |
| **Weaviate** | Slice-based | Per-layer arrays | GC-managed | [][]uint64 connections per layer |
| **Elasticsearch** | Lucene segments | BytesRef storage | Off-heap option | OffHeapVectorValues with IndexInput |
| **Milvus** | Segment-based | Graph serialization | Memory pool | Block allocation with alignment |

### Memory Optimization Techniques

#### Link Compression (Qdrant)
```rust
pub enum GraphLinksType {
    Plain(PlainGraphLinks),
    Compressed(CompressedGraphLinks),  // Delta encoding
}
```

#### Generation-Based Management (Vespa)
```cpp
_gen_handler.scheduleDestroy(old_data);
_gen_handler.reclaim_memory();
```

#### Page-Based Storage (pgvector)
```c
typedef struct HnswElementData {
    ItemPointerData heaptids[HNSW_HEAPTIDS];
    HnswNeighborArray neighbors[FLEXIBLE_ARRAY_MEMBER];
} HnswElementData;
```

## Concurrency Models

### Lock Strategies

| System | Concurrency Model | Lock Granularity | Implementation Details |
|--------|------------------|------------------|----------------------|
| **pgvector** | PostgreSQL MVCC | Buffer-level locks | LockBuffer(BUFFER_LOCK_SHARE/EXCLUSIVE), START_CRIT_SECTION |
| **Qdrant** | RwLock with parking_lot | Graph-level + visited pools | FxHashSet<PointOffsetType> pool per thread |
| **Vespa** | RCU (Read-Copy-Update) | Lock-free reads | vespalib::GenerationHandler, std::memory_order_release |
| **Weaviate** | Multiple specialized locks | Operation-specific | sync.RWMutex for cache/nodes/tombstones |
| **Elasticsearch** | Java synchronization | Segment-level | synchronized blocks, ReentrantReadWriteLock |
| **Milvus** | std::shared_mutex | Component-level | Reader-writer locks for concurrent search |

### Concurrent Operation Examples

**Vespa (RCU)**:
```cpp
PreparedAddNode prepare_add_document(uint32_t docid, VectorBundle vectors) {
    auto guard = _graph.node_refs_size.getGuard();
    // Prepare without locks
}

void complete_add_document(PreparedAddNode prepared) {
    // Atomic commit
}
```

**Weaviate (Fine-grained)**:
```go
deleteLock      sync.Mutex      // For deletions
tombstoneLock   sync.RWMutex    // For tombstone access
insertLock      sync.RWMutex    // For insertions
```

## Performance Optimizations

### SIMD Implementations

| System | SIMD Support | Platforms | Implementation | Distance Functions |
|--------|-------------|-----------|----------------|-------------------|
| **pgvector** | Runtime detection | AVX2, AVX512, NEON | pg_popcount_available() dispatch | L2, IP, Cosine, L1, Hamming |
| **Qdrant** | Manual SIMD | AVX2, AVX512, NEON | #[cfg] + unsafe Rust | Dot, L2, Cosine with scores |
| **Vespa** | Hardware accelerator | AVX2, AVX512 | vespalib::hwaccelrated | Configurable via TypedCells |
| **Weaviate** | Assembly functions | AVX2, AVX512 | //go:noescape with CPU detection | l2/dot with platform variants |
| **Elasticsearch** | Java Vector API | Platform agnostic | VectorSpecies<Float> SPECIES_256 | Lucene's SimdOps |
| **Milvus** | Knowhere SIMD | AVX2, AVX512 | Faiss backend | All standard metrics |

### Distance Calculation Examples

**Weaviate (Assembly)**:
```go
//go:noescape
func l2_avx2(a, b []float32) float32

if cpu.X86.HasAVX2 {
    return l2_avx2(a, b)
}
```

**Qdrant (Rust SIMD)**:
```rust
#[cfg(target_arch = "x86_64")]
pub fn dot_product_avx(a: &[f32], b: &[f32]) -> f32 {
    unsafe { /* AVX implementation */ }
}
```

## Unique Features by System

### pgvector
1. **Two-Phase Build**: Optimizes for large datasets
2. **WAL Integration**: Full crash recovery
3. **PostgreSQL Integration**: Leverages existing infrastructure

### Qdrant
1. **Dynamic Filtering**: Adaptive strategy selection
2. **Payload Subgraphs**: Pre-built indices for common filters
3. **GPU Acceleration**: Optional Vulkan-based GPU acceleration
4. **Quantization**: Multiple compression methods

### Vespa
1. **RCU Concurrency**: Lock-free reads
2. **MIPS Optimization**: Distance transform for maximum inner product
3. **Template Design**: Compile-time optimizations
4. **Generation Management**: Safe memory reclamation

### Weaviate
1. **Commit Log**: Durability and recovery
2. **Compression Variety**: PQ, BQ, SQ support
3. **Multi-Vector**: Late interaction and Muvera
4. **Adaptive Parameters**: Dynamic ef tuning

### Chroma
1. **Library Wrapper**: Leverages proven hnswlib
2. **Provider Pattern**: Efficient index caching
3. **Hybrid Architecture**: Python/Rust split

### Elasticsearch
1. **Lucene Integration**: Codec-based extensibility
2. **Quantization by Default**: int8_hnsw standard
3. **Segment-Based**: Incremental index building

### Milvus
1. **Knowhere Library**: Unified index interface
2. **Distributed Native**: Built for cloud scale
3. **Multi-Index Support**: Beyond just HNSW

## Parameter Handling

### Configuration Comparison

| Parameter | pgvector | Qdrant | Vespa | Weaviate | Elasticsearch | Milvus |
|-----------|----------|---------|--------|-----------|---------------|---------|
| M | 4-100 (default 16) | 4-128 | 2-1024 | 4-64 | 4-512 (default 16) | 4-64 (default 16) |
| ef_construction | 4-1000 (default 64) | 4-4096 | 10-2000 | 8-2048 | 32-512 (default 100) | 8-512 (default 200) |
| ef_search | 1-1000 (GUC) | Dynamic | 10-10000 | Dynamic/Auto | num_candidates param | Configurable |
| Max dimensions | 2000 | 65536 | No limit | 65536 | Up to 4096 | Up to 32768 |
| Build threads | maintenance_max_workers | max_threads | Single/Multi | Go routines | Segment parallel | OMP threads |

### Dynamic Parameter Adjustment

**Qdrant**:
```rust
// Automatic ef based on result quality
ef = max(k * 2, min_ef);
```

**Weaviate**:
```go
// Auto-tuning ef
ef = h.autoEfMin + int(float32(k-h.autoEfMin)*h.autoEfFactor)
```

## Build Strategies

### Parallel Building

**pgvector**:
- Shared memory coordination
- Worker synchronization via condition variables
- Parallel page writes

**Qdrant**:
- Configurable thread count
- Single-threaded start (avoid disconnection)
- GPU acceleration option

### Incremental Building

**Elasticsearch/Lucene**:
- Segment-based incremental updates
- Merge on read
- Background optimization

**Qdrant**:
- Graph healing for incremental updates
- Reuse existing connections
- Maintains graph quality

## Search Strategies

### Filter Handling Innovation

**Qdrant's Cardinality Estimation**:
```rust
if cardinality.max < threshold {
    plain_search()  // Pre-filter
} else if cardinality.min > threshold {
    hnsw_search()   // Post-filter
} else {
    sample_and_decide()  // Hybrid
}
```

**Weaviate's Flat Search Cutoff**:
```go
filterRatio := float32(allowList.Len()) / float32(len(h.nodes))
if filterRatio < h.flatSearchCutoff {
    return h.flatSearch(vector, k, allowList)
}
```

## Persistence and Recovery

### Durability Mechanisms

| System | Persistence Method | Recovery Support | Crash Safety |
|--------|-------------------|------------------|---------------|
| **pgvector** | PostgreSQL WAL | Full ACID compliance | XLogInsert with LSN tracking |
| **Qdrant** | Binary format + mmap | Version checking | State file with CRC validation |
| **Vespa** | Memory-mapped files | Generation-based | Attribute flush with fsync |
| **Weaviate** | Commit log | Log replay | commitlog.AddNode operations |
| **Elasticsearch** | Lucene segments | Translog replay | Immutable segments + translog |
| **Milvus** | Object storage | Binlog replay | S3/MinIO with segment sealing |

## Performance Trade-offs

### Memory vs Speed

1. **Compression**: Qdrant, Weaviate offer multiple options
2. **Link Storage**: Compressed (Qdrant) vs Plain (others)
3. **Caching**: Weaviate's vector cache vs direct access

### Accuracy vs Performance

1. **Quantization**: Elasticsearch (default), Qdrant (optional)
2. **Early Termination**: All systems implement variants
3. **Approximate Filtering**: Qdrant's sampling approach

## Architectural Integration Insights

### Native vs Library Implementations

| Approach | Systems | Benefits | Trade-offs |
|----------|---------|----------|------------|
| **Native Implementation** | pgvector, Qdrant, Vespa, Weaviate | Deep integration, custom optimizations | Higher maintenance, more complex |
| **Library Wrapper** | Chroma (hnswlib), Milvus (Knowhere) | Proven algorithms, faster development | Less flexibility, dependency management |
| **Framework Integration** | Elasticsearch (Lucene) | Ecosystem benefits, codec extensibility | Framework constraints |

### Key Architectural Decisions

1. **Storage Integration**:
   - pgvector: PostgreSQL pages → ACID compliance
   - Qdrant: RocksDB + mmap → Persistence + performance
   - Milvus: Object storage → Cloud scalability
   - Vespa: Custom memory management → Real-time updates

2. **Concurrency Models**:
   - Vespa: RCU → Lock-free reads
   - pgvector: MVCC → Transaction safety
   - Weaviate: Fine-grained locks → Operation isolation
   - Qdrant: Thread-local pools → Reduced contention

3. **Build Strategies**:
   - pgvector: Two-phase → Memory efficiency
   - Elasticsearch: Segment-based → Incremental updates
   - Milvus: Distributed → Horizontal scaling
   - Qdrant: Graph healing → Quality maintenance

## Conclusions

Each HNSW implementation reflects its system's design philosophy:

1. **pgvector**: Deep PostgreSQL integration, reliability-first
2. **Qdrant**: Innovation in filtering, production-focused
3. **Vespa**: Concurrent access optimization, enterprise-scale
4. **Weaviate**: Flexibility and adaptability, operational features
5. **Chroma**: Simplicity through proven libraries
6. **Elasticsearch**: Leverage existing infrastructure
7. **Milvus**: Cloud-native distributed design

The architectural analysis reveals that successful HNSW implementation requires:
- Careful adaptation to existing system architecture
- Balance between performance and operational requirements
- Strategic choices in memory management and persistence
- Integration with system-specific features (transactions, distribution, etc.)