# Qdrant Quantization Techniques Analysis

## Overview
Qdrant implements state-of-the-art quantization techniques including Scalar Quantization (SQ) and Product Quantization (PQ), with a focus on maintaining high recall while reducing memory usage and improving search speed.

## Quantization Methods

### 1. **Scalar Quantization**
```rust
// lib/segment/src/vector_storage/quantized/quantized_vectors.rs
pub struct ScalarQuantizedVectorStorage {
    quantized_data: MmapQuantizedVectors,
    quantization_config: ScalarQuantizationConfig,
    original_distance: Distance,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScalarQuantizationConfig {
    pub r#type: ScalarType,
    pub quantile: Option<f32>,  // e.g., 0.99 for outlier handling
    pub always_ram: Option<bool>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum ScalarType {
    Int8,
}
```

#### Implementation
```rust
impl ScalarQuantization {
    // Training phase - determine optimal scaling factors
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> OperationResult<()> {
        let num_vectors = vectors.len();
        let dim = vectors[0].len();
        
        // Collect statistics per dimension
        let mut min_values = vec![f32::MAX; dim];
        let mut max_values = vec![f32::MIN; dim];
        
        for vector in vectors {
            for (i, &value) in vector.iter().enumerate() {
                min_values[i] = min_values[i].min(value);
                max_values[i] = max_values[i].max(value);
            }
        }
        
        // Handle outliers using quantile if specified
        if let Some(quantile) = self.config.quantile {
            self.apply_quantile_clipping(&mut min_values, &mut max_values, vectors, quantile);
        }
        
        // Calculate scale and offset for each dimension
        self.scales = vec![];
        self.offsets = vec![];
        
        for i in 0..dim {
            let range = max_values[i] - min_values[i];
            if range > 0.0 {
                self.scales.push(255.0 / range);
                self.offsets.push(min_values[i]);
            } else {
                self.scales.push(1.0);
                self.offsets.push(0.0);
            }
        }
        
        Ok(())
    }
    
    // Quantize a vector
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        vector.iter()
            .zip(self.scales.iter().zip(self.offsets.iter()))
            .map(|(&value, (&scale, &offset))| {
                let normalized = (value - offset) * scale;
                normalized.round().clamp(0.0, 255.0) as u8
            })
            .collect()
    }
    
    // SIMD-optimized quantization
    #[target_feature(enable = "avx2")]
    unsafe fn quantize_avx2(&self, vector: &[f32], output: &mut [u8]) {
        use std::arch::x86_64::*;
        
        let zero = _mm256_setzero_ps();
        let max_val = _mm256_set1_ps(255.0);
        
        for chunk_idx in (0..vector.len()).step_by(8) {
            let v = _mm256_loadu_ps(vector.as_ptr().add(chunk_idx));
            let scale = _mm256_loadu_ps(self.scales.as_ptr().add(chunk_idx));
            let offset = _mm256_loadu_ps(self.offsets.as_ptr().add(chunk_idx));
            
            // Quantize: clamp((v - offset) * scale, 0, 255)
            let shifted = _mm256_sub_ps(v, offset);
            let scaled = _mm256_mul_ps(shifted, scale);
            let clamped = _mm256_min_ps(_mm256_max_ps(scaled, zero), max_val);
            
            // Convert to integers
            let rounded = _mm256_round_ps(clamped, _MM_FROUND_TO_NEAREST_INT);
            let ints = _mm256_cvtps_epi32(rounded);
            
            // Pack to bytes
            let packed = _mm256_packus_epi32(ints, ints);
            let packed_low = _mm256_extracti128_si256(packed, 0);
            
            _mm_storel_epi64(
                output.as_mut_ptr().add(chunk_idx) as *mut __m128i,
                _mm_packus_epi16(packed_low, packed_low)
            );
        }
    }
}
```

### 2. **Product Quantization**
```rust
// lib/segment/src/vector_storage/quantized/product_quantization.rs
pub struct ProductQuantization {
    pub num_subvectors: usize,      // M: number of subquantizers
    pub bits_per_subvector: usize,  // typically 8 (256 centroids)
    pub codebooks: Vec<Vec<Vec<f32>>>, // [M][K][D/M] where K = 2^bits
}

impl ProductQuantization {
    // Training using k-means per subspace
    pub fn train(&mut self, vectors: &[Vec<f32>], num_iterations: usize) -> OperationResult<()> {
        let dim = vectors[0].len();
        let subvector_dim = dim / self.num_subvectors;
        let num_centroids = 1 << self.bits_per_subvector;
        
        self.codebooks = vec![vec![vec![0.0; subvector_dim]; num_centroids]; self.num_subvectors];
        
        // Train each subquantizer independently
        vectors.par_chunks(1000)
            .enumerate()
            .for_each(|(m, chunk)| {
                let subvectors: Vec<Vec<f32>> = chunk.iter()
                    .map(|v| v[m * subvector_dim..(m + 1) * subvector_dim].to_vec())
                    .collect();
                
                // Run k-means
                let centroids = self.kmeans(&subvectors, num_centroids, num_iterations);
                self.codebooks[m] = centroids;
            });
        
        Ok(())
    }
    
    // Encode vector to PQ codes
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let subvector_dim = vector.len() / self.num_subvectors;
        let mut codes = vec![0u8; self.num_subvectors];
        
        for m in 0..self.num_subvectors {
            let start = m * subvector_dim;
            let end = (m + 1) * subvector_dim;
            let subvector = &vector[start..end];
            
            // Find nearest centroid
            let mut min_dist = f32::MAX;
            let mut best_idx = 0;
            
            for (idx, centroid) in self.codebooks[m].iter().enumerate() {
                let dist = euclidean_distance(subvector, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = idx;
                }
            }
            
            codes[m] = best_idx as u8;
        }
        
        codes
    }
    
    // Asymmetric distance computation (ADC)
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        // Precompute distance table
        let table = self.compute_distance_table(query);
        
        // Lookup and sum distances
        codes.iter()
            .enumerate()
            .map(|(m, &code)| table[m][code as usize])
            .sum()
    }
    
    // Optimized distance table computation
    fn compute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let subvector_dim = query.len() / self.num_subvectors;
        let num_centroids = self.codebooks[0].len();
        
        (0..self.num_subvectors)
            .into_par_iter()
            .map(|m| {
                let start = m * subvector_dim;
                let end = (m + 1) * subvector_dim;
                let subquery = &query[start..end];
                
                self.codebooks[m].iter()
                    .map(|centroid| euclidean_distance(subquery, centroid))
                    .collect()
            })
            .collect()
    }
}
```

### 3. **Hybrid Quantization**
```rust
// Combining scalar and product quantization
pub struct HybridQuantization {
    scalar_quantization: ScalarQuantization,
    product_quantization: Option<ProductQuantization>,
    config: HybridQuantizationConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HybridQuantizationConfig {
    pub mode: HybridMode,
    pub compression_ratio: f32,
    pub accuracy_threshold: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum HybridMode {
    ScalarOnly,
    ProductOnly,
    ScalarThenProduct,  // Apply SQ first, then PQ on quantized vectors
    Adaptive,           // Choose based on vector characteristics
}

impl HybridQuantization {
    pub fn encode(&self, vector: &[f32]) -> QuantizedVector {
        match self.config.mode {
            HybridMode::ScalarOnly => {
                QuantizedVector::Scalar(self.scalar_quantization.quantize(vector))
            }
            HybridMode::ProductOnly => {
                QuantizedVector::Product(self.product_quantization.encode(vector))
            }
            HybridMode::ScalarThenProduct => {
                // First apply scalar quantization
                let sq_vector = self.scalar_quantization.quantize(vector);
                // Then apply PQ on the quantized vector
                let dequantized = self.scalar_quantization.dequantize(&sq_vector);
                let pq_codes = self.product_quantization.encode(&dequantized);
                QuantizedVector::Hybrid(sq_vector, pq_codes)
            }
            HybridMode::Adaptive => {
                // Choose based on vector statistics
                self.adaptive_encode(vector)
            }
        }
    }
}
```

### 4. **Oversampling and Rescoring**
```rust
// Configuration for search with quantization
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct QuantizationSearchParams {
    pub ignore: bool,  // Bypass quantization for high-precision needs
    pub rescore: Option<bool>,  // Use original vectors for final ranking
    pub oversampling: Option<f32>,  // e.g., 2.0 means fetch 2x more candidates
}

impl QuantizedHnswIndex {
    pub fn search_with_oversampling(
        &self,
        query: &[f32],
        limit: usize,
        params: &QuantizationSearchParams,
    ) -> Vec<ScoredPoint> {
        let oversampling_factor = params.oversampling.unwrap_or(1.0);
        let fetch_limit = (limit as f32 * oversampling_factor) as usize;
        
        // Search using quantized vectors
        let mut candidates = self.search_quantized(query, fetch_limit);
        
        // Optionally rescore with original vectors
        if params.rescore.unwrap_or(true) {
            candidates = self.rescore_with_original(query, candidates, limit);
        }
        
        candidates.truncate(limit);
        candidates
    }
    
    fn rescore_with_original(
        &self,
        query: &[f32],
        mut candidates: Vec<ScoredPoint>,
        limit: usize,
    ) -> Vec<ScoredPoint> {
        // Fetch original vectors for top candidates
        let top_k = candidates.len().min(limit * 2);
        
        for candidate in candidates.iter_mut().take(top_k) {
            let original_vector = self.get_original_vector(candidate.id);
            candidate.score = self.distance.similarity(query, &original_vector);
        }
        
        // Re-sort by new scores
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        candidates
    }
}
```

### 5. **On-Disk Quantization**
```rust
// Memory-mapped quantized storage
pub struct MmapQuantizedVectors {
    mmap: Mmap,
    num_vectors: usize,
    quantized_vector_size: usize,
    metadata: QuantizationMetadata,
}

impl MmapQuantizedVectors {
    pub fn create(path: &Path, config: &QuantizationConfig) -> OperationResult<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        
        // Layout: [metadata][scales][offsets][quantized_vectors]
        let metadata_size = std::mem::size_of::<QuantizationMetadata>();
        let scales_size = config.dim * std::mem::size_of::<f32>();
        let offsets_size = config.dim * std::mem::size_of::<f32>();
        let vectors_size = config.num_vectors * config.dim; // 1 byte per dimension
        
        let total_size = metadata_size + scales_size + offsets_size + vectors_size;
        file.set_len(total_size as u64)?;
        
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        
        Ok(Self {
            mmap: mmap.make_read_only()?,
            num_vectors: config.num_vectors,
            quantized_vector_size: config.dim,
            metadata: config.metadata.clone(),
        })
    }
    
    pub fn get_quantized_vector(&self, idx: usize) -> &[u8] {
        let offset = self.metadata_size() + self.scales_size() + self.offsets_size() +
                    idx * self.quantized_vector_size;
        &self.mmap[offset..offset + self.quantized_vector_size]
    }
}
```

### 6. **Binary Quantization**
```rust
// Binary quantization for maximum compression
pub struct BinaryQuantization {
    thresholds: Vec<f32>,  // Per-dimension thresholds
}

impl BinaryQuantization {
    pub fn train(&mut self, vectors: &[Vec<f32>]) {
        let dim = vectors[0].len();
        self.thresholds = vec![0.0; dim];
        
        // Calculate median per dimension
        for d in 0..dim {
            let mut values: Vec<f32> = vectors.iter()
                .map(|v| v[d])
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            self.thresholds[d] = values[values.len() / 2];
        }
    }
    
    pub fn encode(&self, vector: &[f32]) -> BitVec {
        let mut bits = BitVec::with_capacity(vector.len());
        
        for (value, threshold) in vector.iter().zip(self.thresholds.iter()) {
            bits.push(*value > *threshold);
        }
        
        bits
    }
    
    // Optimized Hamming distance
    pub fn hamming_distance(&self, a: &BitVec, b: &BitVec) -> u32 {
        a.blocks().zip(b.blocks())
            .map(|(block_a, block_b)| (block_a ^ block_b).count_ones())
            .sum()
    }
}
```

### 7. **Performance Monitoring**
```rust
// Quantization performance metrics
#[derive(Debug, Serialize)]
pub struct QuantizationStats {
    pub compression_ratio: f32,
    pub memory_usage_bytes: usize,
    pub average_quantization_error: f32,
    pub search_speed_improvement: f32,
    pub recall_at_10: f32,
}

impl QuantizedCollection {
    pub fn calculate_stats(&self) -> QuantizationStats {
        let original_size = self.num_vectors * self.dim * 4; // f32
        let quantized_size = match &self.quantization {
            Quantization::Scalar(_) => self.num_vectors * self.dim,
            Quantization::Product(pq) => self.num_vectors * pq.num_subvectors,
            _ => original_size,
        };
        
        QuantizationStats {
            compression_ratio: original_size as f32 / quantized_size as f32,
            memory_usage_bytes: quantized_size,
            average_quantization_error: self.calculate_average_error(),
            search_speed_improvement: self.benchmark_search_speed(),
            recall_at_10: self.calculate_recall(10),
        }
    }
}
```

## Configuration Examples

### Collection Configuration
```json
{
    "vectors": {
        "size": 768,
        "distance": "Cosine"
    },
    "quantization_config": {
        "scalar": {
            "type": "int8",
            "quantile": 0.99,
            "always_ram": true
        }
    },
    "hnsw_config": {
        "m": 16,
        "ef_construct": 200,
        "on_disk": false
    }
}
```

### Search Configuration
```json
{
    "vector": [0.1, 0.2, ...],
    "limit": 10,
    "params": {
        "hnsw_ef": 128,
        "quantization": {
            "ignore": false,
            "rescore": true,
            "oversampling": 1.5
        }
    }
}
```

## Performance Characteristics

### Advantages
- Up to 32x compression with binary quantization
- 4x compression with scalar quantization
- Minimal accuracy loss with proper configuration
- Fast SIMD-optimized operations

### Trade-offs
- Training time for product quantization
- Slight accuracy reduction
- Memory overhead for rescoring

## Code References

### Core Implementation
- `lib/segment/src/vector_storage/quantized/` - Quantization implementations
- `lib/segment/src/index/hnsw_index/` - Integration with HNSW
- `lib/collection/src/operations/` - Collection-level quantization

## Comparison Notes
- Most flexible quantization options
- Excellent balance of compression and accuracy
- Production-ready with extensive configuration
- Trade-off: Complexity vs. maximum efficiency