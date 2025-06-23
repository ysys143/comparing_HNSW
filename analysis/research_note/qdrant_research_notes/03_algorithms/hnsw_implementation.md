# Qdrant HNSW Implementation Analysis

## Overview

Qdrant's HNSW implementation is notable for its sophisticated filterable search capabilities, dynamic strategy selection, and production-focused optimizations. It features GPU support, multiple quantization methods, and innovative filtering strategies.

## Graph Construction

### Core Graph Structure

```rust
pub struct GraphLayers<TGraphLinks: GraphLinks> {
    layers: Vec<GraphLayer<TGraphLinks>>,
    backlinks: BackLinksLayer,
    links_layers: LinkLayersArray<TGraphLinks>,
}

pub struct GraphLayer<TGraphLinks: GraphLinks> {
    nodes: Vec<PointOffsetType>,
    links: TGraphLinks,  // Can be Plain or Compressed
}

pub enum GraphLinksType {
    Plain(PlainGraphLinks),
    Compressed(CompressedGraphLinks),
}
```

### Node Insertion Algorithm

```rust
impl HNSWIndex {
    pub fn insert(&mut self, point_id: PointOffsetType, vector: &[f32]) -> Result<()> {
        let level = self.get_random_layer(&mut self.rng);
        let mut nearest = self.search_for_neighbors(vector, self.ef_construct, level);
        
        // Insert at each layer from top to current
        for layer in (0..=level).rev() {
            let m = if layer == 0 { self.m * 2 } else { self.m };
            
            // Select neighbors using heuristic or simple strategy
            let neighbors = if HNSW_USE_HEURISTIC {
                self.select_neighbors_heuristic(&nearest, m)
            } else {
                self.select_neighbors_simple(&nearest, m)
            };
            
            // Add bidirectional links
            for &neighbor in &neighbors {
                self.graph.add_link(point_id, neighbor, layer);
                self.graph.add_link(neighbor, point_id, layer);
                
                // Prune neighbor's connections if needed
                self.prune_connections(neighbor, layer, m);
            }
            
            // Update nearest for next layer
            nearest = self.search_layer(vector, nearest, layer - 1);
        }
        
        // Update entry point if necessary
        if level > self.entry_point_level {
            self.entry_point = point_id;
            self.entry_point_level = level;
        }
    }
}
```

### Heuristic Neighbor Selection

```rust
fn select_neighbors_heuristic(
    &self,
    candidates: &[ScoredPoint],
    m: usize,
) -> Vec<PointOffsetType> {
    let mut result = Vec::with_capacity(m);
    let mut w = candidates.to_vec();
    
    while result.len() < m && !w.is_empty() {
        // Get closest candidate
        let (idx, _) = w.iter().enumerate()
            .min_by(|(_, a), (_, b)| a.score.partial_cmp(&b.score).unwrap())
            .unwrap();
        let current = w.swap_remove(idx);
        
        // Check if it improves connectivity
        let mut good = true;
        for &existing in &result {
            let dist_to_existing = self.distance(current.idx, existing);
            if dist_to_existing < current.score {
                good = false;
                break;
            }
        }
        
        if good {
            result.push(current.idx);
        }
    }
    
    result
}
```

### GPU Acceleration

```rust
#[cfg(feature = "gpu")]
pub fn build_with_gpu(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
    let gpu_builder = GpuGraphBuilder::new(self.config.clone())?;
    
    // Transfer vectors to GPU
    gpu_builder.upload_vectors(vectors)?;
    
    // Build graph on GPU
    let gpu_graph = gpu_builder.build()?;
    
    // Transfer back to CPU
    self.graph = gpu_graph.to_cpu()?;
    
    Ok(())
}
```

## Search Algorithm

### Dynamic Strategy Selection

Qdrant's key innovation is dynamic selection between different search strategies based on filter selectivity:

```rust
pub fn search(&self, query: &[f32], filter: Option<&Filter>, top: usize) -> Vec<ScoredPoint> {
    match filter {
        None => self.search_without_filter(query, top),
        Some(filter) => {
            // Estimate filter cardinality
            let cardinality = self.estimate_cardinality(filter);
            
            // Dynamic strategy selection
            if cardinality.max < self.config.full_scan_threshold {
                // Low selectivity: use plain search
                self.search_plain_filtered(query, filter, top)
            } else if cardinality.min > self.config.full_scan_threshold {
                // High selectivity: use HNSW with filtering
                self.search_hnsw_filtered(query, filter, top)
            } else {
                // Uncertain: use sampling to decide
                if self.sample_check_cardinality(filter) {
                    self.search_hnsw_filtered(query, filter, top)
                } else {
                    self.search_plain_filtered(query, filter, top)
                }
            }
        }
    }
}
```

### Cardinality Estimation

```rust
pub struct CardinalityEstimation {
    pub primary_clauses: Vec<PrimaryCondition>,
    pub min: usize,  // Best case minimum
    pub exp: usize,  // Expected value
    pub max: usize,  // Worst case maximum
}

impl StructPayloadIndex {
    pub fn estimate_cardinality(&self, filter: &Filter) -> CardinalityEstimation {
        match filter {
            Filter::Condition(field_condition) => {
                // Use field index statistics
                self.estimate_field_condition(field_condition)
            }
            Filter::And(filters) => {
                // Multiply selectivities
                filters.iter()
                    .map(|f| self.estimate_cardinality(f))
                    .reduce(|a, b| CardinalityEstimation {
                        min: a.min * b.min / self.total_points,
                        exp: a.exp * b.exp / self.total_points,
                        max: (a.max * b.max / self.total_points).min(self.total_points),
                    })
                    .unwrap_or_default()
            }
            Filter::Or(filters) => {
                // Add selectivities with overlap estimation
                // ...
            }
        }
    }
}
```

### Filtered Search Implementation

```rust
pub fn search_hnsw_filtered(
    &self,
    query: &[f32],
    filter: &Filter,
    top: usize,
    ef: usize,
) -> Vec<ScoredPoint> {
    let mut visited = FixedBitSet::with_capacity(self.points_count);
    let mut candidates = BinaryHeap::new();
    let mut w = BinaryHeap::new();
    
    // Start from entry point
    let entry_point = self.entry_point;
    let entry_dist = self.distance(query, entry_point);
    candidates.push(Reverse(ScoredPoint::new(entry_point, entry_dist)));
    w.push(ScoredPoint::new(entry_point, entry_dist));
    visited.set(entry_point, true);
    
    // Search with filter checking
    while let Some(Reverse(current)) = candidates.pop() {
        if current.score > w.peek().unwrap().score {
            break;
        }
        
        // Get neighbors
        let neighbors = self.graph.get_neighbors(current.idx, 0);
        
        for &neighbor in neighbors {
            if !visited[neighbor] {
                visited.set(neighbor, true);
                
                // Check filter BEFORE distance calculation (optimization)
                if !filter.check(neighbor) {
                    continue;
                }
                
                let distance = self.distance(query, neighbor);
                
                if distance < w.peek().unwrap().score || w.len() < ef {
                    candidates.push(Reverse(ScoredPoint::new(neighbor, distance)));
                    w.push(ScoredPoint::new(neighbor, distance));
                    
                    if w.len() > ef {
                        w.pop();
                    }
                }
            }
        }
    }
    
    // Return top results
    w.into_sorted_vec().into_iter().take(top).collect()
}
```

### Payload-Based Subgraph Building

```rust
pub fn build_additional_filters(&mut self, field_indexes: &PayloadIndex) -> Result<()> {
    for (field, index) in field_indexes {
        // Get common filter conditions
        let payload_blocks = index.payload_blocks(field, self.config.full_scan_threshold);
        
        for block in payload_blocks {
            if block.cardinality > self.percolation_threshold() {
                continue; // Skip to avoid disconnected graph
            }
            
            // Build separate HNSW for this filter
            let filtered_points: Vec<_> = self.points.iter()
                .filter(|p| block.condition.check(p))
                .collect();
            
            let mut subgraph = HNSWIndex::new(self.config.clone());
            subgraph.build_from_points(&filtered_points)?;
            
            // Merge into main graph
            self.graph.merge_from_other(subgraph.graph, &block.condition)?;
        }
    }
    Ok(())
}
```

## Memory Optimization

### Link Compression

```rust
pub struct CompressedGraphLinks {
    compressed_data: Vec<u8>,
    offsets: Vec<u32>,
    decompressor: LinkDecompressor,
}

impl CompressedGraphLinks {
    pub fn compress(plain_links: &PlainGraphLinks) -> Self {
        let mut compressed = Vec::new();
        let mut offsets = Vec::new();
        
        for links in &plain_links.links {
            offsets.push(compressed.len() as u32);
            // Delta encoding + variable length encoding
            let compressed_links = compress_links(links);
            compressed.extend(&compressed_links);
        }
        
        Self {
            compressed_data: compressed,
            offsets,
            decompressor: LinkDecompressor::new(),
        }
    }
}
```

### Visited List Pool

```rust
pub struct PooledVisitedList {
    pool: Mutex<Vec<FixedBitSet>>,
    capacity: usize,
}

impl PooledVisitedList {
    pub fn get(&self) -> VisitedList {
        let mut pool = self.pool.lock();
        pool.pop().unwrap_or_else(|| FixedBitSet::with_capacity(self.capacity))
    }
    
    pub fn return_to_pool(&self, mut visited: FixedBitSet) {
        visited.clear();
        let mut pool = self.pool.lock();
        if pool.len() < MAX_POOL_SIZE {
            pool.push(visited);
        }
    }
}
```

## Quantization Support

```rust
pub enum QuantizedVectors {
    Scalar(ScalarQuantizedVectors),
    Product(ProductQuantizedVectors),
    Binary(BinaryQuantizedVectors),
}

impl HNSWIndex {
    pub fn search_quantized(
        &self,
        query: &[f32],
        quantized_storage: &QuantizedVectors,
        top: usize,
    ) -> Vec<ScoredPoint> {
        // First pass with quantized vectors
        let candidates = self.search_with_scorer(
            query,
            top * self.config.rescore_multiplier,
            |idx| quantized_storage.score(query, idx),
        );
        
        // Rescore top candidates with full vectors
        let mut rescored = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let exact_score = self.distance(query, candidate.idx);
            rescored.push(ScoredPoint::new(candidate.idx, exact_score));
        }
        
        rescored.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        rescored.truncate(top);
        rescored
    }
}
```

## Performance Monitoring

```rust
pub struct HNSWSearchesTelemetry {
    pub unfiltered_plain: AtomicU64,
    pub unfiltered_hnsw: AtomicU64,
    pub small_cardinality: AtomicU64,
    pub large_cardinality: AtomicU64,
    pub exact_filtered: AtomicU64,
    pub exact_unfiltered: AtomicU64,
}

impl HNSWIndex {
    fn record_search(&self, search_type: SearchType) {
        match search_type {
            SearchType::UnfilteredPlain => {
                self.searches_telemetry.unfiltered_plain.fetch_add(1, Ordering::Relaxed);
            }
            SearchType::FilteredHNSW => {
                self.searches_telemetry.large_cardinality.fetch_add(1, Ordering::Relaxed);
            }
            // ... other cases
        }
    }
}
```

## Configuration

```rust
pub struct HnswConfig {
    pub m: usize,                           // 16
    pub ef_construct: usize,                // 100
    pub ef: usize,                          // Dynamic at search time
    pub full_scan_threshold: usize,         // 10000
    pub max_indexing_threads: usize,        // 0 (auto)
    pub on_disk: Option<bool>,              // false
    pub payload_m: Option<usize>,           // None
    pub heuristic_ef: bool,                 // true
    pub indexing_threshold: usize,          // 20000
}
```

## Unique Features

### 1. Dynamic Filtering Strategy
- Automatic selection between pre/post filtering
- Statistical cardinality estimation
- Sampling for uncertain cases

### 2. Payload-Based Subgraphs
- Pre-built indices for common filters
- Avoids graph disconnection with percolation threshold
- Merges multiple graph structures

### 3. Comprehensive Quantization
- Multiple quantization methods
- Integrated rescoring
- Configurable accuracy/speed trade-offs

### 4. Production Features
- Telemetry and monitoring
- GPU acceleration
- Link compression
- Memory pooling

## Summary

Qdrant's HNSW implementation represents a significant advancement in filtered vector search, with its dynamic strategy selection and payload-aware indexing. The production-focused design with comprehensive monitoring, quantization support, and GPU acceleration makes it suitable for demanding real-world applications.