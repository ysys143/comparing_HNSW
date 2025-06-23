# Weaviate HNSW Implementation Analysis

## Overview

Weaviate's HNSW implementation in Go features advanced compression support, multi-vector capabilities, and sophisticated maintenance operations. It includes commit log persistence, tombstone management, and adaptive search strategies.

## Graph Construction

### Core Graph Structure

```go
type hnsw struct {
    sync.RWMutex
    
    nodes               []*vertex
    entryPointID        uint64
    currentMaximumLayer int
    
    // Multiple specialized locks for different operations
    deleteLock      sync.Mutex
    deleteList      *deleteList
    tombstones      map[uint64]struct{}
    tombstoneLock   sync.RWMutex
    
    // Maintenance and compression
    commitlog       *commitLogger
    condensed       bool
    compressor      compressionhelpers.VectorCompressor
    cache           cache.Cache[float32]
    
    // Configuration
    maximumConnections           int
    maximumConnectionsLayerZero  int
    efConstruction              int
    ef                          int
    cleanupIntervalSeconds      int
    flatSearchCutoff            int
}

type vertex struct {
    id         uint64
    level      int
    committed  bool
    markDelete bool
    
    // Connections stored per layer
    connections [][]uint64
    
    // Maintenance tracking
    maintenance bool
    
    // Locks for concurrent access
    sync.Mutex
}
```

### Node Insertion Algorithm

```go
func (h *hnsw) Add(id uint64, vector []float32) error {
    h.Lock()
    defer h.Unlock()
    
    if h.isEmpty() {
        return h.addFirstNode(id, vector)
    }
    
    // Determine layer for new node
    level := h.calculateLevel()
    node := &vertex{
        id:          id,
        level:       level,
        connections: make([][]uint64, level+1),
    }
    
    // Find nearest neighbors at each layer
    entryPointID := h.entryPointID
    targetLevel := h.currentMaximumLayer
    
    for lc := targetLevel; lc >= 0; lc-- {
        candidates := h.searchLayer(vector, entryPointID, 1, lc)
        
        if lc <= level {
            // Connect at this layer
            m := h.maximumConnections
            if lc == 0 {
                m = h.maximumConnectionsLayerZero
            }
            
            neighbors := h.selectNeighborsHeuristic(vector, candidates, m, lc)
            
            for _, neighbor := range neighbors {
                h.connect(id, neighbor, lc)
                h.connect(neighbor, id, lc)
                
                // Prune connections of neighbor if needed
                h.pruneConnections(neighbor, lc)
            }
        }
        
        if len(candidates) > 0 {
            entryPointID = candidates[0].id
        }
    }
    
    // Update entry point if necessary
    if level > h.currentMaximumLayer {
        h.entryPointID = id
        h.currentMaximumLayer = level
    }
    
    // Write to commit log
    if h.commitlog != nil {
        h.commitlog.AddNode(&vertex{id: id, level: level})
    }
    
    return nil
}
```

### Heuristic Neighbor Selection

```go
func (h *hnsw) selectNeighborsHeuristic(
    vector []float32,
    candidates []priorityqueue.Item,
    m int,
    layer int,
) []uint64 {
    // Build candidate set
    w := make([]priorityqueue.Item, 0, len(candidates))
    for _, c := range candidates {
        w = append(w, c)
    }
    
    // Sort by distance
    sort.Slice(w, func(i, j int) bool {
        return w[i].Dist < w[j].Dist
    })
    
    neighbors := make([]uint64, 0, m)
    
    for len(neighbors) < m && len(w) > 0 {
        // Pick closest candidate
        current := w[0]
        w = w[1:]
        
        // Check if it improves connectivity
        good := true
        for _, selected := range neighbors {
            distToSelected := h.distanceBetweenNodes(current.ID, selected)
            if distToSelected < current.Dist {
                good = false
                break
            }
        }
        
        if good {
            neighbors = append(neighbors, current.ID)
        } else if h.extendCandidates {
            // Consider candidates further away
            for _, c := range current.Neighbors {
                if !contains(candidates, c) {
                    dist := h.distanceBetweenVectors(vector, h.getVector(c))
                    w = append(w, priorityqueue.Item{ID: c, Dist: dist})
                }
            }
            
            // Re-sort
            sort.Slice(w, func(i, j int) bool {
                return w[i].Dist < w[j].Dist
            })
        }
    }
    
    return neighbors
}
```

### Compression Support

```go
func (h *hnsw) AddCompressed(id uint64, vector []float32) error {
    if h.compressor == nil {
        return h.Add(id, vector)
    }
    
    // Compress vector
    compressed := h.compressor.Compress(vector)
    
    // Store compressed version
    h.compressedVectorsLock.Lock()
    h.compressedVectors[id] = compressed
    h.compressedVectorsLock.Unlock()
    
    // Use original for graph construction
    return h.Add(id, vector)
}
```

## Search Algorithm

### Adaptive Search Strategy

```go
func (h *hnsw) SearchByVector(
    vector []float32,
    k int,
    allowList helpers.AllowList,
) ([]uint64, []float32, error) {
    h.RLock()
    defer h.RUnlock()
    
    // Adaptive ef parameter
    ef := h.ef
    if ef < 0 {
        ef = h.autoEfFromK(k)
        ef = h.autoEfMin + int(float32(k-h.autoEfMin)*h.autoEfFactor)
        if ef > h.autoEfMax {
            ef = h.autoEfMax
        }
    }
    
    // Choose search strategy based on filter
    if allowList != nil && h.shouldUseFlatSearch(allowList) {
        return h.flatSearch(vector, k, allowList)
    }
    
    // Regular HNSW search
    return h.searchLayerByVector(vector, h.entryPointID, k, ef, 0, allowList)
}

func (h *hnsw) shouldUseFlatSearch(allowList helpers.AllowList) bool {
    // Use flat search for restrictive filters
    filterRatio := float32(allowList.Len()) / float32(len(h.nodes))
    return filterRatio < h.flatSearchCutoff
}
```

### Multi-Vector Search

```go
func (h *hnsw) SearchByMultiVector(
    vectors [][]float32,
    k int,
    allowList helpers.AllowList,
) ([]uint64, []float32, error) {
    switch h.multivectorMode {
    case MultiVectorModeMuvera:
        return h.searchByMultiVectorMuvera(vectors, k, allowList)
    case MultiVectorModeMulti:
        return h.searchByMultiVectorLateInteraction(vectors, k, allowList)
    default:
        return h.SearchByVector(vectors[0], k, allowList)
    }
}

func (h *hnsw) searchByMultiVectorLateInteraction(
    vectors [][]float32,
    k int,
    allowList helpers.AllowList,
) ([]uint64, []float32, error) {
    // Search with each vector
    allCandidates := make(map[uint64]float32)
    
    for _, vector := range vectors {
        candidates := h.searchLayer(vector, h.entryPointID, k*2, 0)
        
        for _, candidate := range candidates {
            if existing, ok := allCandidates[candidate.ID]; !ok || candidate.Dist < existing {
                allCandidates[candidate.ID] = candidate.Dist
            }
        }
    }
    
    // Sort and return top-k
    return h.sortAndReturnTopK(allCandidates, k)
}
```

### Compression-Aware Search

```go
func (h *hnsw) searchWithCompression(
    vector []float32,
    k int,
    ef int,
    allowList helpers.AllowList,
) ([]uint64, []float32, error) {
    // First pass with compressed vectors
    compressedQuery := h.compressor.Compress(vector)
    
    candidates := h.searchLayerCompressed(
        compressedQuery,
        h.entryPointID,
        k*h.rescoringMultiplier,
        ef,
        0,
    )
    
    // Rescore with original vectors
    rescored := make([]priorityqueue.Item, 0, len(candidates))
    for _, candidate := range candidates {
        originalVec := h.cache.Get(candidate.ID)
        exactDist := h.distanceBetweenVectors(vector, originalVec)
        rescored = append(rescored, priorityqueue.Item{
            ID:   candidate.ID,
            Dist: exactDist,
        })
    }
    
    // Sort and return top-k
    sort.Slice(rescored, func(i, j int) bool {
        return rescored[i].Dist < rescored[j].Dist
    })
    
    return extractTopK(rescored, k)
}
```

## Maintenance Operations

### Tombstone Management

```go
func (h *hnsw) Delete(id uint64) error {
    h.deleteLock.Lock()
    defer h.deleteLock.Unlock()
    
    // Mark as tombstone
    h.tombstoneLock.Lock()
    h.tombstones[id] = struct{}{}
    h.tombstoneLock.Unlock()
    
    // Mark node for deletion
    node := h.nodeByID(id)
    if node != nil {
        node.markDelete = true
    }
    
    // Schedule cleanup
    h.deleteList.Add(id)
    
    // Write to commit log
    if h.commitlog != nil {
        h.commitlog.DeleteNode(id)
    }
    
    return nil
}

func (h *hnsw) CleanupTombstones() error {
    h.tombstoneLock.Lock()
    defer h.tombstoneLock.Unlock()
    
    if len(h.tombstones) == 0 {
        return nil
    }
    
    // Remove connections to tombstoned nodes
    for id := range h.tombstones {
        h.removeAllConnections(id)
    }
    
    // Clear tombstones
    h.tombstones = make(map[uint64]struct{})
    
    // Trigger condensor if needed
    if h.shouldCondense() {
        return h.condense()
    }
    
    return nil
}
```

### Commit Log

```go
type commitLogger struct {
    logger       *wal.WAL
    condensed    bool
    condensor    *condensor
}

func (cl *commitLogger) AddNode(node *vertex) error {
    data, err := node.MarshalBinary()
    if err != nil {
        return err
    }
    
    return cl.logger.Write(wal.Entry{
        Type: wal.EntryTypeAddNode,
        Data: data,
    })
}

func (cl *commitLogger) Flush() error {
    if cl.condensed {
        return nil
    }
    
    // Condense log periodically
    if cl.logger.Size() > cl.maxLogSize {
        return cl.condensor.Do()
    }
    
    return cl.logger.Flush()
}
```

### Graph Condensing

```go
type condensor struct {
    hnsw *hnsw
}

func (c *condensor) Do() error {
    // Create new condensed log
    newLog, err := wal.Create(c.hnsw.path + ".condensed")
    if err != nil {
        return err
    }
    
    // Write all active nodes
    for _, node := range c.hnsw.nodes {
        if node != nil && !node.markDelete {
            if err := newLog.WriteNode(node); err != nil {
                return err
            }
        }
    }
    
    // Atomic swap
    oldLog := c.hnsw.commitlog.logger
    c.hnsw.commitlog.logger = newLog
    c.hnsw.condensed = true
    
    // Clean up old log
    return oldLog.Delete()
}
```

## Vector Caching

```go
type vectorCache struct {
    sync.RWMutex
    cache      map[uint64][]float32
    maxSize    int
    lru        *list.List
    lruMap     map[uint64]*list.Element
    prefetcher *prefetcher
}

func (vc *vectorCache) Prefetch(ids []uint64) {
    vc.prefetcher.Prefetch(ids, func(id uint64) {
        // Load vector from storage
        vector := vc.loadVector(id)
        vc.Put(id, vector)
    })
}

func (h *hnsw) prefetchNeighbors(candidates []priorityqueue.Item) {
    // Collect neighbor IDs
    neighborIDs := make([]uint64, 0)
    for _, candidate := range candidates {
        neighbors := h.getNeighbors(candidate.ID, 0)
        neighborIDs = append(neighborIDs, neighbors...)
    }
    
    // Prefetch in background
    h.cache.Prefetch(neighborIDs)
}
```

## Configuration

```go
type Config struct {
    // Basic HNSW parameters
    MaxConnections          int     `json:"maxConnections"`
    EFConstruction         int     `json:"efConstruction"`
    EF                     int     `json:"ef"`
    
    // Dynamic ef tuning
    DynamicEFMin           int     `json:"dynamicEfMin"`
    DynamicEFMax           int     `json:"dynamicEfMax"`
    DynamicEFFactor        float32 `json:"dynamicEfFactor"`
    
    // Search strategy
    FlatSearchCutoff       int     `json:"flatSearchCutoff"`
    
    // Maintenance
    CleanupIntervalSeconds int     `json:"cleanupIntervalSeconds"`
    
    // Compression
    PQ                     *CompressionConfig `json:"pq"`
    BQ                     *CompressionConfig `json:"bq"`
    SQ                     *CompressionConfig `json:"sq"`
    
    // Multi-vector
    MultivectorMode        string  `json:"multivectorMode"`
}
```

## Performance Optimizations

### 1. SIMD Distance Calculations

```go
// Assembly optimized distance functions
//go:noescape
func l2_avx2(a, b []float32) float32

//go:noescape  
func dot_avx2(a, b []float32) float32

func (h *hnsw) distanceBetweenVectors(a, b []float32) float32 {
    switch h.distanceProvider {
    case "l2-squared":
        if cpu.X86.HasAVX2 {
            return l2_avx2(a, b)
        }
        return l2_generic(a, b)
    case "cosine":
        if cpu.X86.HasAVX2 {
            return 1 - dot_avx2(a, b)/(norm_avx2(a)*norm_avx2(b))
        }
        return cosine_generic(a, b)
    }
}
```

### 2. Batch Operations

```go
func (h *hnsw) AddBatch(vectors []VectorWithID) error {
    // Sort by ID for better cache locality
    sort.Slice(vectors, func(i, j int) bool {
        return vectors[i].ID < vectors[j].ID
    })
    
    // Add in batches
    batchSize := 1000
    for i := 0; i < len(vectors); i += batchSize {
        end := i + batchSize
        if end > len(vectors) {
            end = len(vectors)
        }
        
        h.Lock()
        for j := i; j < end; j++ {
            h.addUnsafe(vectors[j].ID, vectors[j].Vector)
        }
        h.Unlock()
        
        // Flush commit log periodically
        if h.commitlog != nil {
            h.commitlog.Flush()
        }
    }
    
    return nil
}
```

### 3. Lock Optimization

```go
// Fine-grained locking for different operations
func (h *hnsw) getVector(id uint64) []float32 {
    // Try cache first (read lock)
    if vec := h.cache.Get(id); vec != nil {
        return vec
    }
    
    // Load from storage (no lock needed)
    vec := h.vectorForID(id)
    
    // Update cache (write lock)
    h.cache.Put(id, vec)
    
    return vec
}
```

## Unique Features

### 1. Commit Log Persistence
- Write-ahead logging for durability
- Periodic condensing to manage size
- Crash recovery support

### 2. Advanced Compression
- Multiple algorithms (PQ, BQ, SQ)
- Compression-aware search with rescoring
- Configurable accuracy/speed trade-offs

### 3. Multi-Vector Support
- Late interaction search
- Muvera algorithm
- Per-vector distance calculations

### 4. Adaptive Search
- Dynamic ef parameter tuning
- Flat search fallback for filters
- Prefetching for cache optimization

### 5. Maintenance Operations
- Tombstone management
- Graph condensing
- Background cleanup cycles

## Summary

Weaviate's HNSW implementation showcases:
1. **Production Robustness**: Commit logging, crash recovery, maintenance operations
2. **Flexibility**: Multiple compression algorithms, multi-vector support
3. **Performance**: SIMD optimizations, caching, prefetching
4. **Adaptability**: Dynamic parameter tuning, strategy selection
5. **Go Integration**: Effective use of Go's concurrency primitives

The implementation balances performance with operational concerns, making it suitable for production deployments with varying workload characteristics.