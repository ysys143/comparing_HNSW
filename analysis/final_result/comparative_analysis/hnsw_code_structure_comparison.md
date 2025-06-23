# HNSW Code Structure Comparison

## Overview

This document provides a comprehensive comparison of HNSW implementation code structures across 7 vector databases, highlighting architectural decisions, design patterns, and implementation strategies.

## Implementation Approaches

### 1. Library-Based vs Native Implementation

**Library Wrappers:**
- **Chroma**: Uses `hnswlib` (C++) via a Rust FFI layer. This **"wrapper-first"** approach prioritizes developer experience and stability by building on a proven, battle-tested library, rather than implementing core algorithms from scratch.
- **Elasticsearch**: Leverages Apache Lucene's HNSW implementation. However, this is not a simple wrapper. Elasticsearch features **deep integration with hardware acceleration** through the **Panama Vector API**, making it a cutting-edge SIMD optimization engine that extends Lucene's capabilities.
- **Milvus**: Uses the `Knowhere` library, an abstracted vector search library that provides unified access to multiple index types (like Faiss, HNSWlib).

**Native Implementations:**
- **pgvector**: Custom C implementation for PostgreSQL
- **Qdrant**: Full Rust implementation
- **Vespa**: Template-based C++ implementation
- **Weaviate**: Go implementation with custom optimizations

### 2. Language Distribution

| Language | Systems | Key Characteristics |
|----------|---------|-------------------|
| C++ | Vespa, Milvus (Knowhere) | Template metaprogramming, SIMD optimizations |
| Rust | Chroma (wrapper), Qdrant | Memory safety, zero-cost abstractions |
| Go | Weaviate | Goroutines, GC-managed memory |
| C | pgvector | PostgreSQL integration, manual memory management |
| Java | Elasticsearch | JVM-based, Lucene integration |

## Core Data Structure Comparison

### Graph Representation

**pgvector**:
```c
struct HnswElementData {
    HnswElementPtr next;
    ItemPointerData heaptids[HNSW_HEAPTIDS];
    uint8 level;
    uint8 deleted;
    HnswNeighborsPtr neighbors;
}
```

**Qdrant**:
```rust
struct HNSWIndex {
    graph: GraphLayers,
    config: HnswGraphConfig,
    searches_telemetry: HNSWSearchesTelemetry,
}
```

**Vespa**:
```cpp
template <HnswIndexType type>
class HnswIndex : public NearestNeighborIndex {
    GraphType graph;
    // RCU for lock-free reads
}
```

**Weaviate**:
```go
type hnsw struct {
    nodes []*vertex
    entryPointID uint64
    currentMaximumLayer int
    tombstones map[uint64]struct{}
}
```

### Key Design Differences

1. **Memory Layout**:
   - **Contiguous**: pgvector (PostgreSQL pages), Vespa (arrays)
   - **Pointer-based**: Weaviate (Go slices), Qdrant (Vec)
   - **External**: Chroma (hnswlib), Elasticsearch (Lucene segments)

2. **Node Representation**:
   - **Inline Data**: pgvector (heaptids in node)
   - **ID-based**: Qdrant, Weaviate (separate vector storage)
   - **Template-based**: Vespa (compile-time optimization)

## Concurrency Models

### 1. Lock-Free Approaches

**Vespa**: RCU (Read-Copy-Update)
- Generation-based memory management
- Lock-free reads during updates
- Copy-on-write for graph modifications

### 2. Fine-Grained Locking

**Weaviate**: Multiple specialized locks
```go
deleteLock      sync.Mutex
tombstoneLock   sync.RWMutex
resetLock       sync.Mutex
insertLock      sync.RWMutex
```

**pgvector**: PostgreSQL LWLocks
- Buffer-level locking
- WAL integration for consistency

### 3. Single Writer Models

**Chroma**: Atomic reference counting
- Thread-safe through Arc<RwLock>
- External library handles internal concurrency

**Qdrant**: Atomic operations with parking_lot
- RwLock for graph access
- Visited pool for search efficiency

## Persistence Strategies

### 1. Write-Ahead Logging

**pgvector**:
- PostgreSQL WAL integration
- Crash recovery support
- Two-phase build process

**Weaviate**:
- Custom commit logger
- Condensed log for compaction
- Async persistence

### 2. Binary Formats

**Chroma**:
```
header.bin
data_level0.bin
length.bin
link_lists.bin
```

**Qdrant**:
- Custom binary serialization
- Version compatibility checks
- Mmap support for large graphs

### 3. Integrated Storage

**Elasticsearch**:
- Lucene segment files
- Codec-based extensibility
- Direct I/O optimizations

**Vespa**:
- Memory-mapped files
- Custom allocators
- Generation-based cleanup

## Build Strategies

### 1. Two-Phase Construction

**pgvector**:
```c
// Phase 1: In-memory build
if (buildstate->heap)
    InitBuildState(buildstate, index);

// Phase 2: Write to disk
if (!buildstate->heap)
    WriteTuplesInOrder(buildstate);
```

### 2. Parallel Building

**Qdrant**:
- Configurable thread count
- Single-threaded start to avoid disconnection
- GPU acceleration option

**Weaviate**:
- Background maintenance cycles
- Incremental optimization
- Concurrent insertions

### 3. Streaming Construction

**Elasticsearch**:
- Segment-based incremental build
- Merge policies
- Background optimization

## Search Implementation Patterns

### 1. Filter Integration

| System | Filter Strategy | Implementation Details |
| :--- | :--- | :--- |
| **Chroma** | Adaptive (Pre/Post) | Uses `hnswlib`'s allow-list for selective filters; otherwise, post-filters. |
| **Elasticsearch** | True Pre-filtering | Leverages Lucene's powerful query engine to create a `BitSet` of allowed documents before search. |
| **Milvus** | Pre-filtering | Pushes boolean expression filters down to storage/index nodes before the vector search. |
| **pgvector** | Post-filtering | Relies on standard PostgreSQL `WHERE` clauses applied after the approximate search. |
| **Qdrant** | Dynamic Pre-filtering | Uses payload indexes and cardinality estimation to dynamically select the best filtering strategy. |
| **Vespa** | True Pre-filtering | Natively integrates filters into the query execution, applying them before or during the HNSW search. |
| **Weaviate** | Adaptive Pre-filtering | Uses a pre-filter for restrictive filters and can fall back to a flat search if the filtered set is small. |


### 2. Distance Calculation

**SIMD Optimizations**:
- **Vespa**: Template specialization and runtime dispatch to hardware-accelerated functions.
- **Qdrant**: Platform-specific implementations using Rust's portable SIMD.
- **Weaviate**: Assembly-level optimizations for `amd64`.
- **Elasticsearch**: Deep integration with Java's Panama Vector API for hardware acceleration.
- **pgvector**: Runtime dispatch to functions compiled with `target_clones` for different CPU features (AVX, F16C).

**Quantization Support**:

| System | Supported Techniques | Key Features |
| :--- | :--- | :--- |
| **Chroma** | None (External) | Relies on application-level quantization before ingestion. |
| **Elasticsearch** | Scalar (int8, int4), Binary (BBQ) | Advanced OSQ optimization, confidence intervals, and default `int8_hnsw`. |
| **Milvus** | Scalar, Product (PQ), Binary | Flexible quantization options provided via the Knowhere engine. |
| **pgvector** | Scalar (halfvec), Binary (bit) | `halfvec` (16-bit float) and `bit` types with SIMD-optimized operations. |
| **Qdrant** | Scalar (uint8), Binary, Product (PQ) | Automatic rescoring, optimized for memory and performance trade-offs. |
| **Vespa** | Scalar (int8), Binary | Native `Int8Float` cell type with hardware-accelerated distance functions. |
| **Weaviate** | Product (PQ), Binary (BQ) | Built-in PQ and BQ with background training and optimization. |


### 3. Result Management

**pgvector**: PostgreSQL heap integration
- Returns ItemPointers
- Integrates with query executor

**Vespa**: Multi-best neighbors
- Supports multi-vector queries
- Global filter pushdown

## Unique Implementation Features

### Chroma
- **Rust Safety Layer**: Provides thread-safe, memory-safe wrappers around C++ components.
- **Developer Experience Focus**: Prioritizes simple APIs and pragmatic design over algorithmic novelty.
- **Provider Pattern**: Centralizes index lifecycle management for efficient caching and resource handling.

### Elasticsearch
- **Panama Vector API**: Leverages Java's modern vector API for aggressive, cross-platform SIMD optimizations.
- **Hardware-Adaptive Algorithms**: Automatically detects and utilizes CPU features like AVX-512 for optimal performance.
- **Quantization by Default**: Employs `int8_hnsw` as a default to balance performance, accuracy, and memory usage.

### Milvus
- **Knowhere Abstraction**: Utilizes the `Knowhere` library, which provides a unified API over various indexing libraries like Faiss and HNSWlib, abstracting away the specific implementation details.
- **Segment-based Architecture**: Data is partitioned into `growing` segments (for real-time data) and `sealed` segments (immutable, for bulk data). Searches are performed on both, and results are merged, enabling real-time indexing without compromising search performance.
- **Decoupled Compute Nodes**: Index building (IndexNode) and searching (QueryNode) are handled by separate, independently scalable microservices, allowing for fine-tuned resource allocation in a distributed environment.

### pgvector
- **Runtime CPU Dispatch**: Detects CPU features (e.g., AVX, F16C) at startup and dispatches to the most optimized function.
- **Hardware-Gated SIMD**: Uses C `target_clones` for instruction-set-specific implementations (e.g., F16C for `halfvec`).
- **Deep PostgreSQL Integration**: Leverages native WAL, memory management, and parallel processing.

### Qdrant
- Payload-based subgraphs
- Link compression
- Telemetry integration

### Vespa
- Identity mapping optimization
- MIPS distance transform
- Tensor framework integration

### Weaviate
- Commit log durability
- Vector cache prefilling
- Module system integration

## Performance Optimization Strategies

### Memory Management
1. **Arena Allocators**: Vespa
2. **Memory Pools**: Qdrant (visited pools)
3. **Context-based**: pgvector
4. **GC Pressure Mitigation**: Weaviate

### Caching
1. **Vector Caching**: Weaviate (with prefilling)
2. **Index Caching**: Chroma (provider cache)
3. **Buffer Management**: pgvector (PostgreSQL buffers)

### Batch Operations
1. **Bulk Insert**: All systems support batch operations
2. **Parallel Processing**: Qdrant, pgvector
3. **Streaming**: Elasticsearch, Vespa

## Integration Patterns

### Database Integration
- **pgvector**: Deep PostgreSQL integration
- **Elasticsearch**: Lucene codec system
- **Vespa**: Attribute system integration

### API Abstraction
- **Trait-based**: Qdrant, Chroma (Rust traits)
- **Interface-based**: Weaviate (Go interfaces)
- **Template-based**: Vespa (C++ templates)

### Storage Backend
- **Custom**: Weaviate (LSMKV), Vespa (custom stores)
- **Library**: pgvector (PostgreSQL), Elasticsearch (Lucene)
- **Flexible**: Qdrant (mmap/memory options)

## Conclusions

The HNSW implementations across these systems reflect different design philosophies:

1. **Performance-First**: Vespa, Qdrant (native implementations with SIMD)
2. **Integration-First**: pgvector, Elasticsearch (leverage existing infrastructure)
3. **Flexibility-First**: Weaviate, Milvus (modular architecture)
4. **Simplicity-First**: Chroma (library wrapper approach)

Each approach has trade-offs between performance, maintainability, and feature richness, ultimately serving different use cases and deployment scenarios.