# Weaviate Architecture Analysis

## Overview

Weaviate is an open-source vector database designed specifically for AI applications, featuring a modular architecture with native HNSW implementation, multi-modal data support, and a sophisticated module system. It emphasizes developer experience with GraphQL APIs, automatic schema inference, and seamless integration with ML models.

## System Architecture

### Modular Architecture

```
┌─────────────────────────────────────────────┐
│            API Layer                         │
│   (GraphQL, REST, gRPC)                      │
├─────────────────────────────────────────────┤
│         Gateway / Router                     │
│    (Request Handling, Auth)                  │
├─────────────────────────────────────────────┤
│          Core Engine                         │
│  ┌─────────────────────────────────────┐    │
│  │      Schema Manager                  │    │
│  │  (Classes, Properties, Modules)      │    │
│  ├─────────────────────────────────────┤    │
│  │     Object Manager                   │    │
│  │  (CRUD Operations, Validation)       │    │
│  ├─────────────────────────────────────┤    │
│  │      Search Manager                  │    │
│  │  (Query Planning, Execution)         │    │
│  ├─────────────────────────────────────┤    │
│  │      Module Manager                  │    │
│  │  (Vectorizers, Readers, Generators)  │    │
│  └─────────────────────────────────────┘    │
├─────────────────────────────────────────────┤
│         Storage Layer                        │
│  ┌─────────────────────────────────────┐    │
│  │        LSM Store                     │    │
│  │    (Object & Property Storage)       │    │
│  ├─────────────────────────────────────┤    │
│  │      Vector Index                    │    │
│  │    (HNSW Implementation)             │    │
│  ├─────────────────────────────────────┤    │
│  │    Inverted Index                    │    │
│  │    (BM25, Filtering)                 │    │
│  └─────────────────────────────────────┘    │
├─────────────────────────────────────────────┤
│      Persistence & Replication               │
│  (WAL, Snapshots, Raft Consensus)           │
└─────────────────────────────────────────────┘
```

## Core Components

### 1. Schema and Data Model

**Schema Definition**:
```go
type Schema struct {
    Classes []*Class
}

type Class struct {
    Class               string
    Description         string
    VectorIndexType     string              // "hnsw"
    VectorIndexConfig   map[string]interface{}
    InvertedIndexConfig *InvertedIndexConfig
    ModuleConfig        map[string]interface{}
    Properties          []*Property
    ShardingConfig      *ShardingConfig
}

type Property struct {
    Name            string
    DataType        []string  // ["text"], ["number"], ["vector"]
    Description     string
    ModuleConfig    map[string]interface{}
    IndexFilterable bool
    IndexSearchable bool
    Tokenization    string
}

// Example schema
schema := &Schema{
    Classes: []*Class{
        {
            Class: "Article",
            VectorIndexType: "hnsw",
            VectorIndexConfig: map[string]interface{}{
                "ef":               100,
                "efConstruction":   128,
                "maxConnections":   32,
                "vectorCacheMaxObjects": 2000000,
            },
            Properties: []*Property{
                {Name: "title", DataType: []string{"text"}},
                {Name: "content", DataType: []string{"text"}},
                {Name: "vector", DataType: []string{"vector"}},
            },
            ModuleConfig: map[string]interface{}{
                "text2vec-transformers": map[string]interface{}{
                    "vectorizeClassName": false,
                },
            },
        },
    },
}
```

### 2. HNSW Implementation

**Vector Index Architecture**:
```go
// Native HNSW implementation in Go
type hnsw struct {
    sync.RWMutex
    
    // Core graph structure
    nodes           []vertex
    entryPointID    uint64
    currentMaxLayer int
    
    // Configuration
    efConstruction  int
    maxConnections  int
    efSearch        int
    seed            int64
    
    // Distance function
    distancer       distancer.Provider
    
    // Memory management
    cache           cache.Cache
    compressed      bool
    
    // Persistence
    commitLog       *commitlog.Log
}

type vertex struct {
    id          uint64
    level       int
    connections [][]uint64  // connections per layer
    vector      []float32
    lock        sync.RWMutex
}

// Insert with concurrent safety
func (h *hnsw) Add(id uint64, vector []float32) error {
    h.Lock()
    defer h.Unlock()
    
    if h.isEmpty() {
        return h.insertFirst(id, vector)
    }
    
    // Assign layer
    level := h.selectLevel()
    node := &vertex{
        id:          id,
        level:       level,
        connections: make([][]uint64, level+1),
        vector:      vector,
    }
    
    // Find neighbors at all layers
    entryPoints := []uint64{h.entryPointID}
    for lc := h.currentMaxLayer; lc >= 0; lc-- {
        candidates := h.searchLayer(vector, entryPoints, 1, lc)
        
        if lc <= level {
            // Select M neighbors
            m := h.maximumConnections(lc)
            neighbors := h.selectNeighborsHeuristic(candidates, m, lc)
            
            // Add bidirectional connections
            for _, neighbor := range neighbors {
                h.connect(id, neighbor, lc)
                h.prune(neighbor, lc)
            }
        }
        
        // Update entry points for next layer
        entryPoints = h.getNeighbors(candidates[0], lc)
    }
    
    // Update entry point if necessary
    if level > h.currentMaxLayer {
        h.entryPointID = id
        h.currentMaxLayer = level
    }
    
    // Persist to commit log
    h.commitLog.AddNode(id, vector, level)
    
    return nil
}

// Approximate k-NN search
func (h *hnsw) SearchByVector(vector []float32, k int, allow allowList) ([]uint64, []float32, error) {
    h.RLock()
    defer h.RUnlock()
    
    if h.isEmpty() {
        return nil, nil, nil
    }
    
    // Multi-layer search
    ep := []uint64{h.entryPointID}
    curr_dist := h.distBetweenVectors(vector, h.vectorByID(h.entryPointID))
    
    for level := h.currentMaxLayer; level > 0; level-- {
        closest := h.searchLayer(vector, ep, 1, level)
        ep = []uint64{closest[0]}
    }
    
    // Search at layer 0 with ef
    candidates := h.searchLayer(vector, ep, max(h.efSearch, k), 0)
    
    // Filter and collect results
    results := make([]uint64, 0, k)
    distances := make([]float32, 0, k)
    
    for _, candidate := range candidates {
        if allow == nil || allow.Contains(candidate) {
            results = append(results, candidate)
            distances = append(distances, h.distBetweenVectors(vector, h.vectorByID(candidate)))
            
            if len(results) >= k {
                break
            }
        }
    }
    
    return results, distances, nil
}

// Efficient neighbor selection with pruning
func (h *hnsw) selectNeighborsHeuristic(candidates []uint64, m int, layer int) []uint64 {
    // Sort by distance
    sort.Slice(candidates, func(i, j int) bool {
        return h.distBetweenNodes(candidates[i], h.entryPointID) < 
               h.distBetweenNodes(candidates[j], h.entryPointID)
    })
    
    selected := make([]uint64, 0, m)
    
    for _, candidate := range candidates {
        if len(selected) >= m {
            break
        }
        
        // Pruning heuristic to maintain connectivity
        good := true
        for _, s := range selected {
            distToCandidate := h.distBetweenNodes(candidate, h.entryPointID)
            distToSelected := h.distBetweenNodes(s, h.entryPointID)
            distBetween := h.distBetweenNodes(candidate, s)
            
            if distBetween < distToCandidate && distBetween < distToSelected {
                good = false
                break
            }
        }
        
        if good {
            selected = append(selected, candidate)
        }
    }
    
    return selected
}
```

### 3. Module System

**Vectorizer Modules**:
```go
// Module interface
type Vectorizer interface {
    VectorizeObject(ctx context.Context, obj *models.Object, cfg moduletools.ClassConfig) error
    VectorizeQuery(ctx context.Context, query string, cfg moduletools.ClassConfig) ([]float32, error)
}

// Example: text2vec-transformers module
type TransformersVectorizer struct {
    client *vectorizerClient
}

func (v *TransformersVectorizer) VectorizeObject(ctx context.Context, obj *models.Object, cfg moduletools.ClassConfig) error {
    text := v.extractText(obj, cfg)
    
    vector, err := v.client.Vectorize(ctx, text)
    if err != nil {
        return err
    }
    
    obj.Vector = vector
    return nil
}

// Multi-modal support
type Multi2VecClip struct {
    client *clipClient
}

func (m *Multi2VecClip) VectorizeObject(ctx context.Context, obj *models.Object, cfg moduletools.ClassConfig) error {
    // Extract different modalities
    texts := m.extractTexts(obj)
    images := m.extractImages(obj)
    
    // Combined vectorization
    vector, err := m.client.VectorizeMultiModal(ctx, texts, images)
    if err != nil {
        return err
    }
    
    obj.Vector = vector
    return nil
}
```

**Generative Modules**:
```go
// Generative AI integration
type GenerativeOpenAI struct {
    client *openai.Client
}

func (g *GenerativeOpenAI) Generate(ctx context.Context, cfg moduletools.ClassConfig, prompt string) (string, error) {
    // Use retrieved objects as context
    systemPrompt := "Answer based on the following context:\n" + g.buildContext(cfg)
    
    response, err := g.client.CreateCompletion(ctx, &openai.CompletionRequest{
        Model:       cfg.Model(),
        Messages:    []openai.Message{
            {Role: "system", Content: systemPrompt},
            {Role: "user", Content: prompt},
        },
        Temperature: cfg.Temperature(),
    })
    
    return response.Choices[0].Message.Content, nil
}
```

### 4. Storage Engine

**LSM-Tree Based Storage**:
```go
type Store struct {
    // LSM tree for objects
    objectStore     *lsmkv.Store
    
    // Property indices
    propertyIndices map[string]*lsmkv.Store
    
    // Vector index
    vectorIndex     vectorindex.VectorIndex
    
    // Inverted index for text search
    invertedIndex   *inverted.Index
}

// Object storage with versioning
func (s *Store) PutObject(ctx context.Context, obj *storobj.Object) error {
    // Serialize object
    data, err := obj.MarshalBinary()
    if err != nil {
        return err
    }
    
    // Store in LSM tree
    bucket := s.objectStore.Bucket(obj.BucketName())
    if err := bucket.Put(obj.ID(), data); err != nil {
        return err
    }
    
    // Update vector index
    if obj.Vector != nil {
        if err := s.vectorIndex.Add(obj.VectorID(), obj.Vector); err != nil {
            return err
        }
    }
    
    // Update inverted indices
    for prop, value := range obj.Properties() {
        if err := s.updatePropertyIndex(prop, value, obj.ID()); err != nil {
            return err
        }
    }
    
    return nil
}

// Efficient filtering with roaring bitmaps
type FilterableIndex struct {
    store *lsmkv.Store
}

func (fi *FilterableIndex) Filter(filter *filters.Clause) (*roaring.Bitmap, error) {
    switch filter.Operator {
    case filters.OperatorEqual:
        return fi.equalityFilter(filter.Property, filter.Value)
    case filters.OperatorGreaterThan:
        return fi.rangeFilter(filter.Property, filter.Value, nil)
    case filters.OperatorAnd:
        return fi.combineFilters(filter.Children, roaring.And)
    case filters.OperatorOr:
        return fi.combineFilters(filter.Children, roaring.Or)
    }
    
    return nil, fmt.Errorf("unsupported operator: %v", filter.Operator)
}
```

### 5. Query Processing

**GraphQL Query Execution**:
```go
type Resolver struct {
    schema      schema.Manager
    search      search.Manager
    modules     *modules.Provider
}

// Complex GraphQL query resolution
func (r *Resolver) Get(ctx context.Context, args *models.GetParams) (*models.GraphQLResponse, error) {
    // Parse GraphQL query
    className := args.ClassName
    properties := args.Properties
    where := args.Where
    nearVector := args.NearVector
    
    // Build search params
    searchParams := &searchparams.SearchParams{
        Class:      className,
        Properties: properties,
        Filters:    r.extractFilters(where),
    }
    
    // Vector search if specified
    if nearVector != nil {
        searchParams.Vector = nearVector.Vector
        searchParams.Certainty = nearVector.Certainty
        searchParams.Limit = args.Limit
    }
    
    // Execute search
    results, err := r.search.Search(ctx, searchParams)
    if err != nil {
        return nil, err
    }
    
    // Apply additional modules (e.g., reranking, generation)
    if args.Additional != nil {
        results, err = r.applyAdditionalModules(ctx, results, args.Additional)
        if err != nil {
            return nil, err
        }
    }
    
    return r.buildResponse(results), nil
}

// Hybrid search combining vector and keyword
func (s *SearchManager) HybridSearch(ctx context.Context, params *HybridSearchParams) ([]*storobj.Object, error) {
    // Vector search
    vectorResults, vectorScores := s.vectorSearch(ctx, params.Vector, params.Alpha)
    
    // BM25 keyword search
    keywordResults, keywordScores := s.keywordSearch(ctx, params.Query, params.Properties)
    
    // Reciprocal Rank Fusion
    fusedResults := s.reciprocalRankFusion(
        vectorResults, vectorScores,
        keywordResults, keywordScores,
        params.Alpha
    )
    
    return fusedResults[:params.Limit], nil
}
```

### 6. Distributed Architecture

**Sharding and Replication**:
```go
type Sharding struct {
    Config      Config
    Nodes       []string
    LocalNode   string
    
    // Consistent hashing
    hashRing    *consistent.Consistent
    
    // Raft consensus
    raft        *raft.Raft
}

// Shard distribution
func (s *Sharding) PhysicalShard(className string, shardName string) string {
    virtualShards := s.Config.VirtualPerPhysical
    
    for i := 0; i < virtualShards; i++ {
        key := fmt.Sprintf("%s_%s_%d", className, shardName, i)
        node := s.hashRing.Get(key)
        
        if s.isHealthy(node) {
            return node
        }
    }
    
    // Fallback to any healthy node
    return s.anyHealthyNode()
}

// Replication with consensus
type Replicator struct {
    raft     *raft.Raft
    factor   int
}

func (r *Replicator) Replicate(ctx context.Context, obj *storobj.Object) error {
    // Create Raft log entry
    logEntry := &raftlog.Entry{
        Type:   raftlog.EntryTypePut,
        Object: obj,
    }
    
    // Apply through Raft consensus
    future := r.raft.Apply(logEntry, 5*time.Second)
    if err := future.Error(); err != nil {
        return err
    }
    
    return nil
}
```

## Performance Optimizations

### 1. Vector Cache

```go
type VectorCache struct {
    cache       *ristretto.Cache
    maxObjects  int64
    dimensions  int
}

func NewVectorCache(maxObjects int64, dimensions int) *VectorCache {
    cache, _ := ristretto.NewCache(&ristretto.Config{
        NumCounters: maxObjects * 10,
        MaxCost:     maxObjects * int64(dimensions) * 4, // float32
        BufferItems: 64,
        Metrics:     true,
    })
    
    return &VectorCache{
        cache:      cache,
        maxObjects: maxObjects,
        dimensions: dimensions,
    }
}

func (vc *VectorCache) Get(id uint64) ([]float32, bool) {
    if val, found := vc.cache.Get(id); found {
        atomic.AddUint64(&vc.hits, 1)
        return val.([]float32), true
    }
    
    atomic.AddUint64(&vc.misses, 1)
    return nil, false
}
```

### 2. SIMD Optimizations

```go
// Assembly optimized distance functions
//go:noescape
func l2DistanceAVX2(a, b []float32) float32

//go:noescape
func dotProductAVX512(a, b []float32) float32

// Runtime CPU detection
func init() {
    if cpu.X86.HasAVX512 {
        distanceFuncs["l2"] = l2DistanceAVX512
        distanceFuncs["dot"] = dotProductAVX512
    } else if cpu.X86.HasAVX2 {
        distanceFuncs["l2"] = l2DistanceAVX2
        distanceFuncs["dot"] = dotProductAVX2
    } else {
        distanceFuncs["l2"] = l2DistanceGeneric
        distanceFuncs["dot"] = dotProductGeneric
    }
}
```

### 3. Async Indexing

```go
type AsyncIndexer struct {
    queue       chan *indexJob
    workers     int
    vectorIndex vectorindex.VectorIndex
}

type indexJob struct {
    id     uint64
    vector []float32
    done   chan error
}

func (ai *AsyncIndexer) Start() {
    for i := 0; i < ai.workers; i++ {
        go ai.worker()
    }
}

func (ai *AsyncIndexer) worker() {
    for job := range ai.queue {
        err := ai.vectorIndex.Add(job.id, job.vector)
        job.done <- err
    }
}

func (ai *AsyncIndexer) AddAsync(id uint64, vector []float32) <-chan error {
    done := make(chan error, 1)
    ai.queue <- &indexJob{id: id, vector: vector, done: done}
    return done
}
```

## Monitoring and Operations

### 1. Metrics and Telemetry

```go
type Metrics struct {
    // Prometheus metrics
    objectsImported    prometheus.Counter
    queriesTotal       prometheus.Counter
    queryLatency       prometheus.Histogram
    vectorIndexSize    prometheus.Gauge
    vectorCacheHitRate prometheus.Gauge
}

func (m *Metrics) TrackQuery(queryType string, duration time.Duration) {
    m.queriesTotal.WithLabelValues(queryType).Inc()
    m.queryLatency.WithLabelValues(queryType).Observe(duration.Seconds())
}

// OpenTelemetry tracing
func (s *Server) handleRequest(ctx context.Context, req *Request) (*Response, error) {
    ctx, span := otel.Tracer("weaviate").Start(ctx, "handleRequest")
    defer span.End()
    
    span.SetAttributes(
        attribute.String("class", req.Class),
        attribute.Int("limit", req.Limit),
    )
    
    // Process request with tracing
    return s.process(ctx, req)
}
```

### 2. Backup and Recovery

```go
type BackupManager struct {
    backend BackupBackend
}

func (bm *BackupManager) CreateBackup(id string, classes []string) error {
    backup := &Backup{
        ID:        id,
        Classes:   classes,
        CreatedAt: time.Now(),
    }
    
    // Snapshot each class
    for _, class := range classes {
        // Create consistent snapshot
        snapshot, err := bm.createClassSnapshot(class)
        if err != nil {
            return err
        }
        
        // Upload to backend (S3, GCS, etc.)
        if err := bm.backend.Upload(backup.ID, class, snapshot); err != nil {
            return err
        }
    }
    
    return bm.backend.WriteMetadata(backup)
}
```

## Configuration

### 1. Docker Compose Setup

```yaml
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-transformers'
      ENABLE_MODULES: 'text2vec-transformers,generative-openai'
      TRANSFORMERS_INFERENCE_API: 'http://t2v-transformers:8080'
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate
      
  t2v-transformers:
    image: semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1
    environment:
      ENABLE_CUDA: '0'
      
volumes:
  weaviate_data:
```

### 2. Production Configuration

```yaml
# config.yaml
persistence:
  dataPath: "/var/lib/weaviate"
  
query:
  maximumResults: 10000
  
schema:
  validationEnforcement: strict
  
vectorIndex:
  hnsw:
    maxConnections: 64
    efConstruction: 256
    ef: 100
    vectorCacheMaxObjects: 2000000
    
limits:
  maxImportGoroutines: 100
  maxImportBatchSize: 200
  
monitoring:
  prometheus:
    enabled: true
    port: 2112
    
cluster:
  join: ["weaviate-0:7000", "weaviate-1:7000", "weaviate-2:7000"]
```

## Summary

Weaviate's architecture demonstrates:
1. **AI-Native Design**: Built specifically for vector search and AI applications
2. **Modular Architecture**: Extensible with vectorizers, readers, and generators
3. **Developer Experience**: GraphQL API, automatic schema management
4. **Multi-Modal Support**: Text, images, and other data types
5. **Production Ready**: Distributed architecture, monitoring, and backup capabilities

The architecture enables Weaviate to serve as a complete vector database platform with seamless integration of AI models and a focus on developer productivity.