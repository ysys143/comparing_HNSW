# Weaviate Quantization Techniques Analysis

## Overview
Weaviate implements Product Quantization (PQ) as its primary quantization technique, focusing on memory efficiency and search performance while maintaining good recall rates.

## Quantization Methods

### 1. **Product Quantization (PQ)**
```go
// adapters/repos/db/vector/compressionhelpers/pq/pq.go
type ProductQuantizer struct {
    config         PQConfig
    centroids      [][]float32      // [M][K*D/M] - M subspaces, K centroids each
    distances      DistanceFunc
    encodedData    []byte           // Compressed vectors
    originalDim    int
    subspaceDim    int
    trainingData   [][]float32
}

type PQConfig struct {
    Enabled        bool    `json:"enabled"`
    BitCompression int     `json:"bitCompression"` // Typically 8 (256 centroids)
    Segments       int     `json:"segments"`       // Number of segments (M)
    Centroids      int     `json:"centroids"`      // K centroids per segment
    TrainingLimit  int     `json:"trainingLimit"`  // Max vectors for training
    Encoder        string  `json:"encoder"`        // "kmeans" or "random"
}

// Training implementation
func (pq *ProductQuantizer) Train(vectors [][]float32) error {
    if len(vectors) == 0 {
        return errors.New("no vectors to train on")
    }
    
    pq.originalDim = len(vectors[0])
    pq.subspaceDim = pq.originalDim / pq.config.Segments
    
    // Ensure we don't exceed training limit
    if len(vectors) > pq.config.TrainingLimit {
        vectors = pq.sampleVectors(vectors, pq.config.TrainingLimit)
    }
    
    // Initialize centroids
    pq.centroids = make([][]float32, pq.config.Segments)
    
    // Train each subquantizer in parallel
    var wg sync.WaitGroup
    errors := make(chan error, pq.config.Segments)
    
    for m := 0; m < pq.config.Segments; m++ {
        wg.Add(1)
        go func(segment int) {
            defer wg.Done()
            
            // Extract subvectors for this segment
            subvectors := pq.extractSubvectors(vectors, segment)
            
            // Run k-means clustering
            centroids, err := pq.kmeansCluster(
                subvectors,
                pq.config.Centroids,
                100, // max iterations
            )
            if err != nil {
                errors <- err
                return
            }
            
            pq.centroids[segment] = centroids
        }(m)
    }
    
    wg.Wait()
    close(errors)
    
    // Check for errors
    for err := range errors {
        if err != nil {
            return err
        }
    }
    
    return nil
}
```

### 2. **K-means Clustering for PQ**
```go
// K-means implementation optimized for PQ training
func (pq *ProductQuantizer) kmeansCluster(vectors [][]float32, k int, maxIter int) ([]float32, error) {
    n := len(vectors)
    d := len(vectors[0])
    
    // Initialize centroids using k-means++
    centroids := pq.initializeCentroidsKMeansPlusPlus(vectors, k)
    
    // Flatten centroids for efficient storage
    flatCentroids := make([]float32, k*d)
    for i := 0; i < k; i++ {
        copy(flatCentroids[i*d:], centroids[i])
    }
    
    assignments := make([]int, n)
    prevAssignments := make([]int, n)
    
    for iter := 0; iter < maxIter; iter++ {
        // Assignment step - assign each vector to nearest centroid
        changed := false
        
        for i := 0; i < n; i++ {
            minDist := float32(math.MaxFloat32)
            bestCentroid := 0
            
            for j := 0; j < k; j++ {
                dist := pq.distances.Distance(
                    vectors[i],
                    flatCentroids[j*d:(j+1)*d],
                )
                if dist < minDist {
                    minDist = dist
                    bestCentroid = j
                }
            }
            
            if assignments[i] != bestCentroid {
                changed = true
            }
            assignments[i] = bestCentroid
        }
        
        if !changed {
            break // Converged
        }
        
        // Update step - recompute centroids
        pq.updateCentroids(vectors, assignments, flatCentroids, k, d)
        
        copy(prevAssignments, assignments)
    }
    
    return flatCentroids, nil
}

// K-means++ initialization for better convergence
func (pq *ProductQuantizer) initializeCentroidsKMeansPlusPlus(vectors [][]float32, k int) [][]float32 {
    n := len(vectors)
    centroids := make([][]float32, k)
    
    // Choose first centroid randomly
    centroids[0] = make([]float32, len(vectors[0]))
    copy(centroids[0], vectors[rand.Intn(n)])
    
    // Choose remaining centroids
    for i := 1; i < k; i++ {
        distances := make([]float32, n)
        totalDist := float32(0)
        
        // Compute distance to nearest centroid for each point
        for j := 0; j < n; j++ {
            minDist := float32(math.MaxFloat32)
            for c := 0; c < i; c++ {
                dist := pq.distances.Distance(vectors[j], centroids[c])
                if dist < minDist {
                    minDist = dist
                }
            }
            distances[j] = minDist * minDist // Square for probability
            totalDist += distances[j]
        }
        
        // Choose next centroid with probability proportional to squared distance
        r := rand.Float32() * totalDist
        cumSum := float32(0)
        
        for j := 0; j < n; j++ {
            cumSum += distances[j]
            if cumSum >= r {
                centroids[i] = make([]float32, len(vectors[j]))
                copy(centroids[i], vectors[j])
                break
            }
        }
    }
    
    return centroids
}
```

### 3. **Vector Encoding and Compression**
```go
// Encode vectors using trained PQ
func (pq *ProductQuantizer) Encode(vectors [][]float32) ([]byte, error) {
    if len(pq.centroids) == 0 {
        return nil, errors.New("quantizer not trained")
    }
    
    n := len(vectors)
    m := pq.config.Segments
    
    // Each vector encoded as M bytes (one per segment)
    encoded := make([]byte, n*m)
    
    // Process vectors in batches for better cache utilization
    batchSize := 1000
    numBatches := (n + batchSize - 1) / batchSize
    
    var wg sync.WaitGroup
    for batch := 0; batch < numBatches; batch++ {
        wg.Add(1)
        go func(b int) {
            defer wg.Done()
            
            start := b * batchSize
            end := min((b+1)*batchSize, n)
            
            for i := start; i < end; i++ {
                pq.encodeVector(vectors[i], encoded[i*m:])
            }
        }(batch)
    }
    
    wg.Wait()
    return encoded, nil
}

// Encode single vector
func (pq *ProductQuantizer) encodeVector(vector []float32, output []byte) {
    for m := 0; m < pq.config.Segments; m++ {
        // Extract subvector
        start := m * pq.subspaceDim
        end := (m + 1) * pq.subspaceDim
        subvector := vector[start:end]
        
        // Find nearest centroid
        minDist := float32(math.MaxFloat32)
        bestCentroid := 0
        
        for k := 0; k < pq.config.Centroids; k++ {
            centroidStart := k * pq.subspaceDim
            centroid := pq.centroids[m][centroidStart : centroidStart+pq.subspaceDim]
            
            dist := pq.distances.Distance(subvector, centroid)
            if dist < minDist {
                minDist = dist
                bestCentroid = k
            }
        }
        
        output[m] = byte(bestCentroid)
    }
}
```

### 4. **Distance Computation with PQ**
```go
// Asymmetric distance computation (ADC)
type PQDistancer struct {
    pq              *ProductQuantizer
    distanceTable   [][]float32  // Precomputed distances [M][K]
    query          []float32
}

// Precompute distance table for query
func (d *PQDistancer) PrecomputeDistanceTable(query []float32) {
    m := d.pq.config.Segments
    k := d.pq.config.Centroids
    
    d.query = query
    d.distanceTable = make([][]float32, m)
    
    // Compute distances in parallel
    var wg sync.WaitGroup
    for segment := 0; segment < m; segment++ {
        wg.Add(1)
        go func(s int) {
            defer wg.Done()
            
            d.distanceTable[s] = make([]float32, k)
            
            // Extract query subvector
            start := s * d.pq.subspaceDim
            end := (s + 1) * d.pq.subspaceDim
            querySubvec := query[start:end]
            
            // Compute distance to each centroid
            for c := 0; c < k; c++ {
                centroidStart := c * d.pq.subspaceDim
                centroid := d.pq.centroids[s][centroidStart : centroidStart+d.pq.subspaceDim]
                
                d.distanceTable[s][c] = d.pq.distances.Distance(querySubvec, centroid)
            }
        }(segment)
    }
    
    wg.Wait()
}

// Fast distance computation using lookup table
func (d *PQDistancer) Distance(encodedVector []byte) float32 {
    distance := float32(0)
    
    // Sum distances from lookup table
    for m := 0; m < d.pq.config.Segments; m++ {
        centroidIdx := int(encodedVector[m])
        distance += d.distanceTable[m][centroidIdx]
    }
    
    return distance
}

// SIMD-optimized batch distance computation
func (d *PQDistancer) BatchDistance(encodedVectors [][]byte, distances []float32) {
    n := len(encodedVectors)
    m := d.pq.config.Segments
    
    // Process multiple vectors in parallel
    numWorkers := runtime.NumCPU()
    chunkSize := (n + numWorkers - 1) / numWorkers
    
    var wg sync.WaitGroup
    for w := 0; w < numWorkers; w++ {
        wg.Add(1)
        go func(worker int) {
            defer wg.Done()
            
            start := worker * chunkSize
            end := min((worker+1)*chunkSize, n)
            
            for i := start; i < end; i++ {
                dist := float32(0)
                for j := 0; j < m; j++ {
                    dist += d.distanceTable[j][encodedVectors[i][j]]
                }
                distances[i] = dist
            }
        }(w)
    }
    
    wg.Wait()
}
```

### 5. **Integration with HNSW**
```go
// adapters/repos/db/vector/hnsw/compressed_vector_index.go
type CompressedVectorIndex struct {
    hnsw           *hnsw
    pq             *ProductQuantizer
    compressedData []byte
    cache          *VectorCache
}

// Build compressed index
func (c *CompressedVectorIndex) Build(vectors [][]float32) error {
    // Train PQ on sample of vectors
    trainingSize := min(len(vectors), c.pq.config.TrainingLimit)
    trainingData := c.sampleVectors(vectors, trainingSize)
    
    if err := c.pq.Train(trainingData); err != nil {
        return fmt.Errorf("failed to train PQ: %w", err)
    }
    
    // Encode all vectors
    encoded, err := c.pq.Encode(vectors)
    if err != nil {
        return fmt.Errorf("failed to encode vectors: %w", err)
    }
    
    c.compressedData = encoded
    
    // Build HNSW on original vectors for graph structure
    // But store compressed vectors for memory efficiency
    return c.hnsw.Build(vectors)
}

// Search with compressed vectors
func (c *CompressedVectorIndex) Search(query []float32, k int) ([]uint64, []float32, error) {
    // Create PQ distancer with precomputed table
    pqDist := &PQDistancer{pq: c.pq}
    pqDist.PrecomputeDistanceTable(query)
    
    // Search using compressed vectors
    ef := c.hnsw.ef
    
    // Over-fetch to compensate for approximation
    overFetchFactor := 1.5
    fetchK := int(float32(k) * overFetchFactor)
    
    candidates := c.searchCompressed(query, fetchK, ef, pqDist)
    
    // Optional: Re-rank using original vectors
    if c.cache != nil {
        candidates = c.rerankWithOriginal(query, candidates, k)
    }
    
    // Extract top-k results
    ids := make([]uint64, k)
    distances := make([]float32, k)
    
    for i := 0; i < k && i < len(candidates); i++ {
        ids[i] = candidates[i].id
        distances[i] = candidates[i].distance
    }
    
    return ids, distances, nil
}
```

### 6. **Memory Management**
```go
// Memory-efficient storage for compressed vectors
type CompressedVectorStorage struct {
    segmentSize    int
    segments       []*CompressedSegment
    totalVectors   int
    bytesPerVector int
}

type CompressedSegment struct {
    data      []byte
    startIdx  int
    endIdx    int
    mmap      bool
    mmapFile  *os.File
}

// Store compressed vectors with option for memory mapping
func (s *CompressedVectorStorage) Store(compressed []byte, useMmap bool) error {
    numVectors := len(compressed) / s.bytesPerVector
    
    if useMmap && numVectors > 100000 {
        // Use memory-mapped file for large datasets
        return s.storeMmap(compressed)
    }
    
    // Store in memory for smaller datasets
    segment := &CompressedSegment{
        data:     compressed,
        startIdx: s.totalVectors,
        endIdx:   s.totalVectors + numVectors,
        mmap:     false,
    }
    
    s.segments = append(s.segments, segment)
    s.totalVectors += numVectors
    
    return nil
}

// Retrieve compressed vector
func (s *CompressedVectorStorage) Get(idx int) []byte {
    // Find segment containing this index
    for _, segment := range s.segments {
        if idx >= segment.startIdx && idx < segment.endIdx {
            offset := (idx - segment.startIdx) * s.bytesPerVector
            return segment.data[offset : offset+s.bytesPerVector]
        }
    }
    
    return nil
}
```

### 7. **Performance Monitoring**
```go
// Metrics for PQ performance
type PQMetrics struct {
    CompressionRatio   float32
    EncodingTime       time.Duration
    SearchSpeedup      float32
    MemoryUsage        int64
    RecallAt10         float32
}

func (pq *ProductQuantizer) CalculateMetrics(testVectors [][]float32) PQMetrics {
    start := time.Now()
    
    // Measure encoding time
    encoded, _ := pq.Encode(testVectors)
    encodingTime := time.Since(start)
    
    // Calculate compression ratio
    originalSize := len(testVectors) * len(testVectors[0]) * 4 // float32
    compressedSize := len(encoded)
    compressionRatio := float32(originalSize) / float32(compressedSize)
    
    // Memory usage
    memoryUsage := int64(len(pq.centroids) * len(pq.centroids[0]) * 4)
    memoryUsage += int64(compressedSize)
    
    return PQMetrics{
        CompressionRatio: compressionRatio,
        EncodingTime:     encodingTime,
        MemoryUsage:      memoryUsage,
    }
}
```

## Configuration

### Schema Configuration
```yaml
# Vector index configuration with PQ
vectorIndexConfig:
  pq:
    enabled: true
    bitCompression: 8      # 256 centroids per segment
    segments: 0            # 0 = auto-determine based on dimensions
    centroids: 256         # Number of centroids per segment
    trainingLimit: 100000  # Max vectors to use for training
    encoder:
      type: "kmeans"
      distribution: "log-normal"
```

### Dynamic Configuration
```go
// Runtime PQ configuration
func ConfigurePQ(dim int) PQConfig {
    config := PQConfig{
        Enabled:       true,
        BitCompression: 8,
        Centroids:     256,
        TrainingLimit: 100000,
        Encoder:       "kmeans",
    }
    
    // Auto-determine number of segments
    if dim <= 128 {
        config.Segments = 8
    } else if dim <= 512 {
        config.Segments = 16
    } else if dim <= 1024 {
        config.Segments = 32
    } else {
        config.Segments = 64
    }
    
    // Ensure each segment has at least 4 dimensions
    maxSegments := dim / 4
    if config.Segments > maxSegments {
        config.Segments = maxSegments
    }
    
    return config
}
```

## Performance Characteristics

### Advantages
- Significant memory reduction (typically 32-64x)
- Fast distance computation with lookup tables
- Good recall with proper configuration
- Seamless integration with HNSW

### Trade-offs
- Training time overhead
- Slight accuracy loss (typically 5-10%)
- Additional CPU usage during search
- Requires representative training data

## Best Practices

### Training Data Selection
```go
func SelectTrainingData(allVectors [][]float32, limit int) [][]float32 {
    if len(allVectors) <= limit {
        return allVectors
    }
    
    // Use stratified sampling for better representation
    // Implementation depends on data characteristics
    return stratifiedSample(allVectors, limit)
}
```

### Optimal Segment Configuration
1. **Small dimensions (< 128)**: 8 segments
2. **Medium dimensions (128-512)**: 16-32 segments
3. **Large dimensions (> 512)**: 32-64 segments
4. **Rule of thumb**: Each segment should have 4-16 dimensions

## Code References

### Core Implementation
- `adapters/repos/db/vector/compressionhelpers/pq/` - PQ implementation
- `adapters/repos/db/vector/hnsw/compressed_vector_index.go` - Integration
- `adapters/repos/db/vector/compressionhelpers/` - Compression utilities

## Comparison Notes
- Focused on PQ as primary quantization method
- Simple configuration and integration
- Good balance between compression and accuracy
- Trade-off: Limited quantization options vs. ease of use