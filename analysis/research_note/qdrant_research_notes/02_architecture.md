# Qdrant Architecture Analysis

## Overview

Qdrant is built entirely in Rust with a focus on production-ready vector search. It features a sophisticated filterable HNSW implementation with dynamic strategy selection, comprehensive quantization support, and distributed capabilities.

## System Architecture

### Layer Structure

```
┌─────────────────────────────────────────────┐
│            API Layer                        │
│    (REST/Actix-web, gRPC/Tonic)           │
├─────────────────────────────────────────────┤
│         Collection Management               │
│    (Sharding, Replication, Routing)        │
├─────────────────────────────────────────────┤
│          Segment Layer                      │
│   (Vector Storage, Payload Indexing)       │
├─────────────────────────────────────────────┤
│         Index Layer                         │
│  (HNSW, Plain, Sparse, Quantization)      │
├─────────────────────────────────────────────┤
│         Storage Layer                       │
│    (RocksDB, Mmap, In-memory)             │
└─────────────────────────────────────────────┘
```

## Core Components

### 1. API Layer (`/src/`)

**REST API** (Actix-web):
```rust
pub struct CollectionApi;
impl CollectionApi {
    pub async fn create_collection() -> Result<HttpResponse>
    pub async fn search_points() -> Result<HttpResponse>
    pub async fn upsert_points() -> Result<HttpResponse>
}
```

**gRPC API** (Tonic):
- Protocol buffer definitions
- Streaming support
- Binary efficiency

### 2. Segment Architecture (`/lib/segment/`)

**Core Structure**:
```rust
pub struct Segment {
    version: SeqNumberType,
    persisted_version: Arc<Mutex<SeqNumberType>>,
    current_path: PathBuf,
    vector_data: HashMap<String, VectorStorage>,
    payload_index: StructPayloadIndex,
    id_tracker: IdTracker,
}
```

**Key Features**:
- Multiple vector fields per segment
- Payload indexing for filtering
- Version tracking for consistency
- Concurrent read/write support

### 3. HNSW Implementation (`/lib/segment/src/index/hnsw_index/`)

**Dynamic Strategy Selection**:
```rust
pub struct HNSWIndex<TGraphLinks> {
    graph: GraphLayers<TGraphLinks>,
    config: HnswGraphConfig,
    searches_telemetry: HNSWSearchesTelemetry,
    visited_pool: PooledVisitedList,
}

// Dynamic filtering strategy
fn search_with_filter(&self, filter: &Filter) -> SearchResult {
    let cardinality = self.estimate_cardinality(filter);
    
    if cardinality.max < self.config.full_scan_threshold {
        self.search_plain_filtered(...)
    } else if cardinality.min > self.config.full_scan_threshold {
        self.search_hnsw_filtered(...)
    } else {
        // Use sampling to decide
        self.sample_and_search(...)
    }
}
```

**Graph Structure**:
```rust
pub struct GraphLayers<TGraphLinks> {
    layers: Vec<GraphLayer<TGraphLinks>>,
    entry_points: EntryPoints,
    visited_pool: PooledVisitedList,
}

pub struct GraphLayer<TGraphLinks> {
    nodes: Vec<Node>,
    links: TGraphLinks, // Compressed or uncompressed
}
```

### 4. Quantization Support (`/lib/quantization/`)

**Quantization Types**:
- Scalar Quantization (SQ)
- Product Quantization (PQ)
- Binary Quantization (BQ)

**Integration**:
```rust
pub trait Quantized {
    fn encode(&self, vector: &[f32]) -> QuantizedVector;
    fn score(&self, query: &[f32], stored: &QuantizedVector) -> f32;
    fn rescore(&self, query: &[f32], candidates: Vec<ScoredPoint>) -> Vec<ScoredPoint>;
}
```

### 5. Storage Architecture

**Storage Options**:
```rust
pub enum StorageType {
    InMemory(InMemoryStorage),
    Mmap(MmapStorage),
    OnDisk(RocksDbStorage),
}
```

**RocksDB Integration**:
- Key-value storage for metadata
- Column families for different data types
- Write-ahead logging
- Compression support

## Key Design Patterns

### 1. Trait-Based Abstraction

```rust
pub trait VectorIndex {
    fn search(&self, vectors: &[&QueryVector], filter: Option<&Filter>, top: usize) -> Vec<Vec<ScoredPoint>>;
    fn build(vectors: &dyn VectorStorage, config: &IndexConfig) -> Result<Self>;
}

// Multiple implementations
impl VectorIndex for HNSWIndex { ... }
impl VectorIndex for PlainIndex { ... }
impl VectorIndex for SparseIndex { ... }
```

### 2. Builder Pattern

```rust
pub struct SegmentBuilder {
    vector_configs: HashMap<String, VectorConfig>,
    payload_indexes: HashMap<String, PayloadIndex>,
}

impl SegmentBuilder {
    pub fn with_vector_field(mut self, name: &str, config: VectorConfig) -> Self { ... }
    pub fn with_payload_index(mut self, field: &str, index_type: PayloadIndexType) -> Self { ... }
    pub fn build(self) -> Result<Segment> { ... }
}
```

### 3. Visitor Pattern for Queries

```rust
pub trait QueryVisitor {
    fn visit_nearest(&mut self, query: &NearestQuery);
    fn visit_discovery(&mut self, query: &DiscoveryQuery);
    fn visit_context(&mut self, query: &ContextQuery);
}
```

## Data Flow

### 1. Insert/Update Flow

```
API Request
    ↓
Collection Router
    ↓
Shard Selection
    ↓
Segment Selection
    ↓
Concurrent Operations:
├── Vector Storage Update
├── Payload Index Update
└── ID Tracker Update
    ↓
WAL Write
    ↓
Response
```

### 2. Search Flow

```
Search Request
    ↓
Query Parsing
    ↓
Filter Cardinality Estimation
    ↓
Strategy Selection:
├── Plain Search (low cardinality)
├── HNSW Search (high cardinality)
└── Sampling Decision (uncertain)
    ↓
Score Computation
    ↓
Quantization Rescoring (if needed)
    ↓
Result Aggregation
    ↓
Response
```

## Concurrency Model

### 1. Lock Strategy

```rust
// Multiple RwLocks for different operations
pub struct ConcurrentSegment {
    vectors: RwLock<VectorStorage>,
    payload: RwLock<PayloadIndex>,
    id_tracker: RwLock<IdTracker>,
}
```

### 2. Visited Pool

```rust
pub struct PooledVisitedList {
    pool: Mutex<Vec<VisitedList>>,
}

// Efficient reuse of visited tracking
impl PooledVisitedList {
    pub fn get(&self) -> VisitedList {
        self.pool.lock().pop().unwrap_or_else(VisitedList::new)
    }
}
```

### 3. Parallel Operations

- Parallel index building with configurable threads
- Concurrent search operations
- Atomic version counters

## Memory Management

### 1. Vector Storage Options

```rust
pub enum VectorStorageEnum {
    Simple(SimpleVectorStorage),       // In-memory
    Memmap(MemmapVectorStorage),       // Memory-mapped
    Multi(MultiVectorStorage),         // Multi-vector support
    Quantized(QuantizedVectorStorage), // Compressed vectors
}
```

### 2. Memory Optimization

- Lazy loading for mmap storage
- Vector compression with quantization
- Efficient serialization formats
- Pooled allocations for search

## Performance Optimizations

### 1. SIMD Distance Calculations

```rust
#[cfg(target_arch = "x86_64")]
pub fn dot_product_avx(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        // AVX implementation
    }
}

#[cfg(target_arch = "aarch64")]
pub fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        // NEON implementation
    }
}
```

### 2. Link Compression

```rust
pub enum GraphLinksType {
    Plain(PlainGraphLinks),
    Compressed(CompressedGraphLinks),
}

// Compressed representation saves memory
pub struct CompressedGraphLinks {
    compressed_data: Vec<u8>,
    offsets: Vec<u32>,
}
```

### 3. GPU Acceleration

```rust
#[cfg(feature = "gpu")]
pub struct GpuIndex {
    device: CudaDevice,
    vectors: CudaBuffer<f32>,
}
```

## Distributed Features

### 1. Consensus (Raft)

```rust
pub struct RaftNode {
    id: NodeId,
    state: RaftState,
    log: RaftLog,
    peers: Vec<NodeId>,
}
```

### 2. Sharding

```rust
pub struct ShardDistribution {
    shard_count: u32,
    replication_factor: u32,
    placement: HashMap<ShardId, Vec<NodeId>>,
}
```

### 3. Replication

- Synchronous replication for consistency
- Asynchronous replication for performance
- Read replicas for scaling

## Monitoring and Telemetry

### 1. Search Telemetry

```rust
pub struct HNSWSearchesTelemetry {
    unfiltered_plain: AtomicU64,
    unfiltered_hnsw: AtomicU64,
    small_cardinality: AtomicU64,
    large_cardinality: AtomicU64,
    exact_filtered: AtomicU64,
}
```

### 2. Metrics Collection

- Prometheus integration
- Custom metrics for vector operations
- Performance profiling hooks

## Error Handling

### Comprehensive Error Types

```rust
#[derive(Error, Debug)]
pub enum CollectionError {
    #[error("Collection {0} not found")]
    NotFound(String),
    #[error("Invalid vector dimension: expected {expected}, got {got}")]
    InconsistentDimension { expected: usize, got: usize },
    // ... more variants
}
```

## Configuration

### Index Configuration

```rust
pub struct HnswConfig {
    pub m: usize,                    // Number of connections
    pub ef_construct: usize,         // Build-time search width
    pub ef: usize,                   // Search-time width
    pub full_scan_threshold: usize,  // Strategy switching threshold
    pub max_indexing_threads: usize, // Parallelism control
    pub on_disk: Option<bool>,       // Storage option
    pub payload_m: Option<usize>,    // Payload subgraph connections
}
```

## Summary

Qdrant's architecture demonstrates:
1. **Production Focus**: Comprehensive error handling, monitoring, and distributed features
2. **Performance Innovation**: Dynamic strategy selection, quantization, GPU support
3. **Rust Benefits**: Memory safety, zero-cost abstractions, excellent concurrency
4. **Flexibility**: Multiple storage options, index types, and deployment modes
5. **Advanced Features**: Sophisticated filtering, discovery queries, multi-vector support

The architecture is designed for high-performance production deployments with careful attention to memory efficiency, query performance, and operational concerns.