# Qdrant Performance & Scalability Analysis

## Overview

Qdrant's Rust-based architecture provides zero-cost abstractions, memory safety without garbage collection, and fine-grained concurrency control. The system employs sophisticated memory management techniques including object pooling, memory-mapped files, graph link compression, and SIMD-optimized distance calculations with optional GPU acceleration.

## Memory Management

### 1. **Visited List Pool**

```rust
// lib/segment/src/index/visited_pool.rs
pub struct VisitedPool {
    pool: RwLock<Vec<VisitedList>>,
}

struct VisitedList {
    current_iter: u8,
    visit_counters: Vec<u8>,
}

impl VisitedPool {
    pub fn get(&self, num_points: usize) -> VisitedListHandle {
        match self.pool.write().pop() {
            None => VisitedListHandle::new(self, VisitedList::new(num_points)),
            Some(data) => {
                let mut visited_list = VisitedListHandle::new(self, data);
                visited_list.resize(num_points);
                visited_list.next_iteration();
                visited_list
            }
        }
    }
    
    fn return_back(&self, data: VisitedList) {
        let mut pool = self.pool.write();
        if pool.len() < *POOL_KEEP_LIMIT {
            pool.push(data);
        }
    }
}

// Efficient visited tracking without memory allocation
impl VisitedList {
    fn next_iteration(&mut self) {
        self.current_iter = self.current_iter.wrapping_add(1);
        if self.current_iter == 0 {
            self.current_iter = 1;
            self.visit_counters.fill(0);
        }
    }
}
```

### 2. **Graph Links Compression**

```rust
// lib/segment/src/index/hnsw_index/graph_links.rs
#[derive(Debug)]
enum GraphLinksEnum {
    Ram(Vec<u8>),
    Mmap(Arc<Mmap>),
}

pub struct GraphLinks {
    owner: GraphLinksEnum,
    dependent: GraphLinksView,
}

impl GraphLinks {
    pub fn load_from_file(
        path: &Path,
        on_disk: bool,
        format: GraphLinksFormat,
    ) -> OperationResult<Self> {
        let populate = !on_disk;
        let mmap = open_read_mmap(path, AdviceSetting::Advice(Advice::Random), populate)?;
        Self::try_new(GraphLinksEnum::Mmap(Arc::new(mmap)), |x| {
            x.load_view(format)
        })
    }
}

// Delta encoding + variable length encoding for link compression
pub struct CompressedGraphLinks {
    compressed_data: Vec<u8>,
    offsets: Vec<u32>,
    decompressor: LinkDecompressor,
}
```

### 3. **Memory-Mapped Storage**

```rust
// Memory advice for optimal access patterns
use memory::madvise::{Advice, AdviceSetting, Madviseable};

impl GraphLinksEnum {
    fn load_view(&self, format: GraphLinksFormat) -> OperationResult<GraphLinksView> {
        let data = match self {
            GraphLinksEnum::Ram(data) => data.as_slice(),
            GraphLinksEnum::Mmap(mmap) => &mmap[..],
        };
        GraphLinksView::load(data, format)
    }
}

// Populate disk cache on demand
pub fn populate(&self) -> OperationResult<()> {
    if let GraphLinksEnum::Mmap(mmap) = self.borrow_owner() {
        mmap.advise(Advice::WillNeed)?;
    }
    Ok(())
}
```

### 4. **Hardware-Aware Memory Allocation**

```rust
// CPU cache-aware structures
const SINGLE_THREADED_HNSW_BUILD_THRESHOLD: usize = 256;

// Prevent false sharing in concurrent structures
#[repr(align(64))] // Cache line alignment
struct AlignedCounter {
    value: AtomicU64,
}

// Memory pool limits based on hardware
pub fn calculate_pool_size() -> usize {
    let num_cpus = num_cpus::get();
    let base_pool_size = 32;
    base_pool_size * num_cpus
}
```

## Concurrency Model

### 1. **Lock-Free and Fine-Grained Locking**

```rust
// lib/segment/src/index/hnsw_index/hnsw.rs
pub struct HNSWIndex {
    id_tracker: Arc<AtomicRefCell<IdTrackerSS>>,
    vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
    quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,
    payload_index: Arc<AtomicRefCell<StructPayloadIndex>>,
    graph: GraphLayers,
    searches_telemetry: HNSWSearchesTelemetry,
}

// Atomic reference cells for lock-free reads
impl HNSWIndex {
    pub fn search(&self, query: &[f32], ef: usize) -> Vec<ScoredPointOffset> {
        // Lock-free access to vector storage
        let vector_storage = self.vector_storage.borrow();
        let quantized_vectors = self.quantized_vectors.borrow();
        
        // No locking required for graph traversal
        self.graph.search(query, ef, &vector_storage, &quantized_vectors)
    }
}
```

### 2. **Rayon-Based Parallel Construction**

```rust
// Parallel index building with thread pool
impl HNSWIndex {
    pub fn build_index(&mut self, permit: &CpuPermit) -> OperationResult<()> {
        let pool = rayon::ThreadPoolBuilder::new()
            .thread_name(|idx| format!("hnsw-build-{idx}"))
            .num_threads(permit.num_cpus as usize)
            .spawn_handler(|thread| {
                let mut b = thread::Builder::new();
                if let Some(name) = thread.name() {
                    b = b.name(name.to_owned());
                }
                b.spawn(|| {
                    // On Linux, use lower thread priority
                    #[cfg(target_os = "linux")]
                    if let Err(err) = linux_low_thread_priority() {
                        log::debug!("Failed to set low thread priority: {err}");
                    }
                    thread.run()
                })?;
                Ok(())
            })
            .build()?;
            
        // Parallel insertion
        pool.install(|| {
            ids.into_par_iter().try_for_each(|id| {
                self.insert_point(id)
            })
        })?;
        
        Ok(())
    }
}
```

### 3. **Parking Lot RwLocks**

```rust
use parking_lot::{RwLock, Mutex};

// Faster, more predictable locks than std
pub struct VisitedPool {
    pool: RwLock<Vec<VisitedList>>,  // parking_lot RwLock
}

// Optimized telemetry aggregation
struct HNSWSearchesTelemetry {
    unfiltered_plain: Arc<Mutex<OperationDurationsAggregator>>,
    filtered_plain: Arc<Mutex<OperationDurationsAggregator>>,
    unfiltered_hnsw: Arc<Mutex<OperationDurationsAggregator>>,
    small_cardinality: Arc<Mutex<OperationDurationsAggregator>>,
    large_cardinality: Arc<Mutex<OperationDurationsAggregator>>,
}
```

### 4. **Atomic Operations and Ordering**

```rust
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// Lock-free cancellation
pub fn check_process_stopped(is_stopped: &AtomicBool) -> OperationResult<()> {
    if is_stopped.load(Ordering::Relaxed) {
        return Err(OperationError::Cancelled);
    }
    Ok(())
}

// Relaxed ordering for statistics
impl HNSWSearchesTelemetry {
    fn record_search(&self, search_type: SearchType) {
        match search_type {
            SearchType::UnfilteredPlain => {
                self.unfiltered_plain.fetch_add(1, Ordering::Relaxed);
            }
            SearchType::FilteredHNSW => {
                self.large_cardinality.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}
```

## I/O Optimization

### 1. **Memory-Mapped Files with Advice**

```rust
use memory::mmap_ops::open_read_mmap;
use memory::madvise::{Advice, AdviceSetting};

impl GraphLinks {
    pub fn load_from_file(
        path: &Path,
        on_disk: bool,
        format: GraphLinksFormat,
    ) -> OperationResult<Self> {
        let populate = !on_disk;
        
        // Advise kernel about access pattern
        let mmap = open_read_mmap(
            path, 
            AdviceSetting::Advice(Advice::Random),  // Random access pattern
            populate  // Pre-populate if in-memory
        )?;
        
        Self::try_new(GraphLinksEnum::Mmap(Arc::new(mmap)), |x| {
            x.load_view(format)
        })
    }
}

// Clear disk cache for benchmarking
#[cfg(target_os = "linux")]
pub fn clear_cache(&self) -> OperationResult<()> {
    if let GraphLinksEnum::Mmap(mmap) = self.borrow_owner() {
        clear_disk_cache(mmap.as_ptr(), mmap.len())?;
    }
    Ok(())
}
```

### 2. **Compressed Storage Format**

```rust
// Hierarchical graph structure with compression
const LINK_COMPRESSION_FORMAT: GraphLinksFormat = GraphLinksFormat::Compressed;

impl GraphLayersBuilder {
    pub fn write_to_disk(&self, path: &Path) -> OperationResult<()> {
        // Serialize graph structure
        let graph_data = GraphLayerData {
            m: self.m,
            m0: self.m0,
            ef_construct: self.ef_construct,
            entry_points: Cow::Borrowed(&self.entry_points),
        };
        
        // Write compressed links
        let links_path = if LINK_COMPRESSION_FORMAT == GraphLinksFormat::Compressed {
            path.join(COMPRESSED_HNSW_LINKS_FILE)
        } else {
            path.join(HNSW_LINKS_FILE)
        };
        
        GraphLinksSerializer::new(self.links_map.iter())
            .write_to_file(&links_path)?;
            
        Ok(())
    }
}
```

### 3. **Batch I/O Operations**

```rust
// Efficient batch vector loading
impl VectorStorage {
    pub fn prefetch_vectors(&self, ids: &[PointOffsetType]) -> OperationResult<()> {
        match self {
            VectorStorage::Mmap(storage) => {
                // Group sequential IDs for efficient prefetching
                let mut ranges = Vec::new();
                let mut start = ids[0];
                let mut end = ids[0];
                
                for &id in &ids[1..] {
                    if id == end + 1 {
                        end = id;
                    } else {
                        ranges.push((start, end));
                        start = id;
                        end = id;
                    }
                }
                ranges.push((start, end));
                
                // Prefetch ranges
                for (start, end) in ranges {
                    storage.prefetch_range(start, end)?;
                }
            }
            _ => {} // In-memory storage doesn't need prefetching
        }
        Ok(())
    }
}
```

### 4. **Asynchronous I/O Support**

```rust
// Async snapshot creation
impl HNSWIndex {
    pub async fn create_snapshot_async(&self, path: &Path) -> OperationResult<()> {
        // Clone necessary data
        let graph_data = self.graph.clone();
        let path = path.to_owned();
        
        // Spawn blocking task for I/O
        tokio::task::spawn_blocking(move || {
            graph_data.write_to_disk(&path)
        }).await??;
        
        Ok(())
    }
}
```

## Performance Monitoring and Tuning

### 1. **Detailed Telemetry Collection**

```rust
// lib/segment/src/telemetry.rs
#[derive(Serialize, Clone, Debug, JsonSchema)]
pub struct VectorIndexSearchesTelemetry {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_name: Option<VectorNameBuf>,
    
    #[serde(skip_serializing_if = "OperationDurationStatistics::is_empty")]
    pub unfiltered_plain: OperationDurationStatistics,
    
    #[serde(skip_serializing_if = "OperationDurationStatistics::is_empty")]
    pub unfiltered_hnsw: OperationDurationStatistics,
    
    #[serde(skip_serializing_if = "OperationDurationStatistics::is_empty")]
    pub filtered_small_cardinality: OperationDurationStatistics,
    
    #[serde(skip_serializing_if = "OperationDurationStatistics::is_empty")]
    pub filtered_large_cardinality: OperationDurationStatistics,
}

// Duration tracking with percentiles
pub struct OperationDurationStatistics {
    pub count: usize,
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p99_ms: f64,
}
```

### 2. **Hardware Counter Integration**

```rust
use common::counter::hardware_counter::HardwareCounterCell;
use common::counter::hardware_accumulator::HwMeasurementAcc;

impl HNSWIndex {
    pub fn search_with_hw_counter(
        &self,
        query: &[f32],
        filter: Option<&Filter>,
        hw_counter: &HardwareCounterCell,
    ) -> Vec<ScoredPointOffset> {
        let _hw_measurement = HwMeasurementAcc::new(hw_counter);
        
        // Hardware counters track:
        // - CPU cycles
        // - Cache misses
        // - Branch mispredictions
        // - Memory bandwidth
        
        self.search_internal(query, filter)
    }
}
```

### 3. **Dynamic Strategy Selection Metrics**

```rust
impl HNSWIndex {
    fn select_search_strategy(&self, filter: &Filter) -> SearchStrategy {
        let cardinality = self.estimate_cardinality(filter);
        
        // Record strategy selection
        match cardinality {
            CardinalityEstimate::Small(n) if n < self.config.full_scan_threshold => {
                self.telemetry.strategy_selections.small_cardinality.inc();
                SearchStrategy::PlainFiltered
            }
            CardinalityEstimate::Large(n) if n > self.config.full_scan_threshold * 10 => {
                self.telemetry.strategy_selections.large_cardinality.inc();
                SearchStrategy::HnswFiltered
            }
            _ => {
                self.telemetry.strategy_selections.sampling_needed.inc();
                self.sample_and_decide(filter)
            }
        }
    }
}
```

## Scalability Characteristics

### 1. **GPU Acceleration**

```rust
#[cfg(feature = "gpu")]
impl HNSWIndex {
    pub fn build_with_gpu(
        &mut self,
        gpu_device: &LockedGpuDevice,
        vectors: &VectorStorage,
    ) -> OperationResult<()> {
        let gpu_vectors = GpuVectorStorage::from_cpu_vectors(
            gpu_device,
            vectors,
            self.config.gpu_batch_size,
        )?;
        
        let gpu_constructed_graph = build_hnsw_on_gpu(
            &self.id_tracker.borrow(),
            &gpu_vectors,
            &self.graph_layers_builder,
            self.config.ef_construct,
            GPU_MAX_VISITED_FLAGS_FACTOR,
        )?;
        
        self.graph = gpu_constructed_graph.to_cpu()?;
        Ok(())
    }
}

// GPU kernel configuration
const GPU_MAX_VISITED_FLAGS_FACTOR: usize = 4;
const GPU_BLOCK_SIZE: usize = 256;
const GPU_GRID_SIZE: usize = 1024;
```

### 2. **Distributed Sharding**

```rust
// Shard-aware operations
pub struct ShardedHNSW {
    shards: Vec<HNSWIndex>,
    shard_selector: Arc<dyn ShardSelector>,
}

impl ShardedHNSW {
    pub fn search_distributed(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&Filter>,
    ) -> Vec<ScoredPointOffset> {
        // Parallel search across shards
        let shard_results: Vec<_> = self.shards
            .par_iter()
            .map(|shard| shard.search(query, limit * 2, filter))
            .collect();
            
        // Merge results
        let mut all_results = Vec::with_capacity(shard_results.len() * limit);
        for results in shard_results {
            all_results.extend(results);
        }
        
        // Sort and take top-k
        all_results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        all_results.truncate(limit);
        all_results
    }
}
```

### 3. **Memory Pressure Handling**

```rust
// Adaptive memory management
impl HNSWIndex {
    pub fn handle_memory_pressure(&mut self) -> OperationResult<()> {
        // Check available memory
        let available_memory = sys_info::mem_info()?.avail;
        let index_memory = self.estimate_memory_usage();
        
        if index_memory > available_memory * 8 / 10 {
            // Switch to on-disk mode
            self.convert_to_on_disk()?;
            
            // Enable more aggressive compression
            self.enable_link_compression()?;
            
            // Reduce cache sizes
            self.visited_pool.resize(self.visited_pool.size() / 2);
        }
        
        Ok(())
    }
    
    fn estimate_memory_usage(&self) -> u64 {
        let graph_size = self.graph.memory_usage();
        let vectors_size = self.vector_storage.borrow().memory_usage();
        let quantized_size = self.quantized_vectors.borrow()
            .as_ref()
            .map(|q| q.memory_usage())
            .unwrap_or(0);
            
        graph_size + vectors_size + quantized_size
    }
}
```

### 4. **Elastic Resource Management**

```rust
// CPU permit system for resource control
pub struct CpuPermit {
    pub num_cpus: u32,
    _guard: PermitGuard,
}

impl HNSWIndex {
    pub fn build_with_cpu_budget(
        &mut self,
        cpu_budget: &CpuBudget,
    ) -> OperationResult<()> {
        // Acquire CPU permit
        let permit = cpu_budget.acquire(self.estimate_build_cpus())?;
        
        // Build with allocated resources
        self.build_index(&permit)?;
        
        // Permit automatically released on drop
        Ok(())
    }
    
    fn estimate_build_cpus(&self) -> u32 {
        let num_vectors = self.id_tracker.borrow().num_indexed_points();
        let cpus_needed = (num_vectors / 100_000).max(1).min(32);
        cpus_needed as u32
    }
}
```

## Configuration and Tuning Parameters

### 1. **HNSW Configuration**

```rust
pub struct HnswConfig {
    pub m: usize,                           // 16
    pub ef_construct: usize,                // 100
    pub ef: usize,                          // Dynamic at search
    pub full_scan_threshold: usize,         // 10000
    pub max_indexing_threads: usize,        // 0 (auto)
    pub on_disk: Option<bool>,              // false
    pub payload_m: Option<usize>,           // None
    pub heuristic: bool,                    // true
    pub indexing_threshold: usize,          // 20000
}

impl HnswConfig {
    pub fn optimize_for_recall(&mut self) {
        self.m = 32;
        self.ef_construct = 200;
        self.heuristic = true;
    }
    
    pub fn optimize_for_speed(&mut self) {
        self.m = 16;
        self.ef_construct = 100;
        self.heuristic = false;
    }
}
```

### 2. **Runtime Optimization**

```rust
// Dynamic parameter adjustment
impl HNSWIndex {
    pub fn auto_tune(&mut self, workload: &WorkloadProfile) {
        match workload.search_type {
            SearchType::HighRecall => {
                self.config.ef = (workload.avg_k * 2).max(64);
            }
            SearchType::LowLatency => {
                self.config.ef = workload.avg_k.max(32);
            }
        }
        
        // Adjust full scan threshold based on filter selectivity
        if workload.avg_filter_selectivity < 0.01 {
            self.config.full_scan_threshold = 1000;
        } else if workload.avg_filter_selectivity > 0.1 {
            self.config.full_scan_threshold = 20000;
        }
    }
}
```

## Best Practices Summary

### 1. **Memory Management**
- Efficient object pooling for visited lists
- Memory-mapped files with appropriate advice
- Compression for graph links
- Hardware-aware memory alignment

### 2. **Concurrency**
- Lock-free data structures where possible
- Fine-grained locking with parking_lot
- Rayon for CPU-bound parallelism
- Atomic operations with relaxed ordering

### 3. **I/O Optimization**
- Memory-mapped files for large datasets
- Compressed storage formats
- Batch I/O operations
- Asynchronous snapshots

### 4. **Scalability**
- GPU acceleration support
- Distributed sharding
- Adaptive memory management
- Elastic resource allocation

## Code References

- `lib/segment/src/index/hnsw_index/hnsw.rs` - Main HNSW implementation
- `lib/segment/src/index/visited_pool.rs` - Visited list pooling
- `lib/segment/src/index/hnsw_index/graph_links.rs` - Graph storage and compression
- `lib/segment/src/telemetry.rs` - Performance monitoring
- `lib/segment/src/index/hnsw_index/gpu/` - GPU acceleration

## Comparison Notes

- **Advantages**: Zero-cost abstractions, no GC overhead, advanced filtering strategies, production-ready monitoring
- **Trade-offs**: Compilation time, learning curve for Rust
- **Scalability**: Excellent vertical and horizontal scaling, GPU support, adaptive strategies