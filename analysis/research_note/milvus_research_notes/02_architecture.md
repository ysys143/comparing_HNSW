# Milvus Architecture Analysis

## Overview

Milvus is a purpose-built vector database designed from the ground up for similarity search and AI applications. It features a distributed architecture with separation of storage and compute, supporting multiple index types including HNSW, IVF, and DiskANN for different performance and scale requirements.

## System Architecture

### Cloud-Native Architecture

```
┌─────────────────────────────────────────────┐
│           SDK/Client Layer                   │
│    (Python, Java, Go, Node.js, REST)        │
├─────────────────────────────────────────────┤
│            Proxy Layer                       │
│   (Request Routing, Authentication)          │
├─────────────────────────────────────────────┤
│         Coordinator Services                 │
│  ┌─────────────────────────────────────┐    │
│  │       Root Coordinator               │    │
│  │   (DDL, Cluster Metadata, TSO)      │    │
│  ├─────────────────────────────────────┤    │
│  │       Data Coordinator               │    │
│  │   (Data Management, Segments)        │    │
│  ├─────────────────────────────────────┤    │
│  │       Query Coordinator              │    │
│  │   (Query Distribution, Balance)      │    │
│  ├─────────────────────────────────────┤    │
│  │       Index Coordinator              │    │
│  │   (Index Building, Management)       │    │
│  └─────────────────────────────────────┘    │
├─────────────────────────────────────────────┤
│           Worker Nodes                       │
│  ┌─────────────────────────────────────┐    │
│  │         Query Nodes                  │    │
│  │    (Vector Search, Filtering)        │    │
│  ├─────────────────────────────────────┤    │
│  │         Data Nodes                   │    │
│  │    (Data Insertion, Flush)           │    │
│  ├─────────────────────────────────────┤    │
│  │         Index Nodes                  │    │
│  │    (Index Building, Optimization)    │    │
│  └─────────────────────────────────────┘    │
├─────────────────────────────────────────────┤
│         Storage Layer                        │
│  ┌─────────────────────────────────────┐    │
│  │      Object Storage (MinIO/S3)       │    │
│  │   (Segments, Indexes, Binlogs)       │    │
│  ├─────────────────────────────────────┤    │
│  │      Message Queue (Pulsar/Kafka)    │    │
│  │   (WAL, Streaming, Pub/Sub)          │    │
│  ├─────────────────────────────────────┤    │
│  │      Metadata Store (etcd)           │    │
│  │   (Schema, Collection Metadata)      │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

## Core Components

### 1. Data Model

**Collection Schema**:
```go
type CollectionSchema struct {
    Name        string
    Description string
    AutoID      bool
    Fields      []*FieldSchema
}

type FieldSchema struct {
    FieldID      int64
    Name         string
    IsPrimaryKey bool
    DataType     DataType
    TypeParams   map[string]string  // e.g., dim for vectors
    IndexParams  map[string]string  // e.g., nlist, m, efConstruction
}
```

**Segment Structure**:
```go
type Segment struct {
    SegmentID    int64
    CollectionID int64
    PartitionID  int64
    NumRows      int64
    State        SegmentState
    
    // Vector data storage
    VectorFields map[int64]*VectorFieldData
    ScalarFields map[int64]*ScalarFieldData
    
    // Index information
    IndexInfos   map[int64]*IndexInfo
}
```

### 2. HNSW Implementation

**Knowhere Integration**:
```cpp
// Milvus uses Knowhere library for vector indexes
class HnswIndex : public VecIndex {
public:
    struct Config {
        int M = 16;              // Number of bi-directional links
        int efConstruction = 200; // Size of dynamic candidate list
        int ef = 64;             // Search parameter
        MetricType metric_type;   // L2, IP, COSINE
    };

    // Build HNSW index
    Status Build(const DatasetPtr& dataset) override {
        auto hnsw = std::make_shared<hnswlib::HierarchicalNSW>(
            dataset->Get<int64_t>(meta::DIM),
            dataset->Get<int64_t>(meta::ROWS),
            M_, efConstruction_
        );
        
        // Parallel index building
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            hnsw->addPoint(data + i * dim, i);
        }
        
        index_ = hnsw;
        return Status::OK();
    }
    
    // Search in HNSW graph
    DatasetPtr Search(const DatasetPtr& query, const Config& config) override {
        auto result = std::make_shared<Dataset>();
        
        #pragma omp parallel for
        for (int i = 0; i < nq; ++i) {
            auto neighbors = index_->searchKnn(
                query_data + i * dim, 
                config.k,
                config.ef
            );
            // Store results
        }
        
        return result;
    }
};
```

### 3. Streaming Architecture

**Write Path**:
```go
// Data flow through message queue
type DataNode struct {
    // Receive insert messages from Pulsar/Kafka
    msgStream msgstream.MsgStream
    
    // Buffer management
    insertBuffer *InsertBuffer
    
    // Segment management
    segmentManager *SegmentManager
}

func (dn *DataNode) ProcessInsert(msg *msgstream.InsertMsg) error {
    // 1. Write to WAL (message queue)
    err := dn.msgStream.Produce(msg)
    
    // 2. Buffer in memory
    dn.insertBuffer.Insert(msg.Timestamps, msg.RowData)
    
    // 3. Flush to segment when buffer full
    if dn.insertBuffer.Full() {
        segment := dn.segmentManager.AllocateSegment()
        segment.Insert(dn.insertBuffer.Data())
        
        // 4. Persist to object storage
        dn.FlushSegment(segment)
    }
    
    return nil
}
```

**Query Path**:
```go
type QueryNode struct {
    // Loaded segments
    historical []*Segment  // Sealed segments from storage
    streaming  []*Segment  // Growing segments from stream
    
    // Vector search engine
    searchEngine *SearchEngine
}

func (qn *QueryNode) Search(req *SearchRequest) (*SearchResult, error) {
    // 1. Search historical segments
    historicalResults := qn.searchHistorical(req)
    
    // 2. Search streaming segments
    streamingResults := qn.searchStreaming(req)
    
    // 3. Merge results
    finalResults := qn.mergeResults(historicalResults, streamingResults)
    
    // 4. Apply filters and reduce
    return qn.reduce(finalResults, req.TopK)
}
```

### 4. Distributed Coordination

**Root Coordinator**:
```go
type RootCoord struct {
    // Metadata management
    meta *MetaTable
    
    // Timestamp Oracle (TSO)
    tsoAllocator *TimestampAllocator
    
    // DDL operations
    ddlTasks chan Task
}

// Global timestamp allocation for consistency
func (rc *RootCoord) AllocTimestamp(count uint32) (uint64, error) {
    return rc.tsoAllocator.Alloc(count)
}

// Collection creation with distributed consensus
func (rc *RootCoord) CreateCollection(req *CreateCollectionRequest) error {
    // 1. Validate schema
    if err := validateSchema(req.Schema); err != nil {
        return err
    }
    
    // 2. Allocate collection ID
    collectionID := rc.idAllocator.AllocID()
    
    // 3. Create metadata in etcd
    collection := &Collection{
        ID:     collectionID,
        Schema: req.Schema,
        State:  CollectionCreated,
    }
    
    // 4. Notify other coordinators
    rc.notifyCoordinators(collection)
    
    return rc.meta.AddCollection(collection)
}
```

**Query Coordinator**:
```go
type QueryCoord struct {
    // Cluster topology
    cluster *QueryCluster
    
    // Load balancer
    balancer *LoadBalancer
    
    // Segment distribution
    dist *SegmentDistribution
}

// Dynamic load balancing
func (qc *QueryCoord) Balance() error {
    // 1. Collect node metrics
    metrics := qc.collectNodeMetrics()
    
    // 2. Calculate optimal distribution
    plan := qc.balancer.Plan(metrics, qc.dist)
    
    // 3. Execute rebalancing
    for _, action := range plan.Actions {
        switch action.Type {
        case LoadSegment:
            qc.loadSegment(action.NodeID, action.SegmentID)
        case ReleaseSegment:
            qc.releaseSegment(action.NodeID, action.SegmentID)
        }
    }
    
    return nil
}
```

### 5. Index Building Pipeline

**Index Node Operations**:
```cpp
class IndexNode {
public:
    // Asynchronous index building
    Status BuildIndex(const BuildIndexRequest& request) {
        // 1. Load data from object storage
        auto dataset = LoadDataset(request.segment_id);
        
        // 2. Select index type
        auto index = IndexFactory::CreateIndex(request.index_type);
        
        // 3. Configure index parameters
        index->SetConfig(request.index_params);
        
        // 4. Build index (possibly GPU-accelerated)
        if (request.device_type == GPU) {
            return BuildOnGPU(index, dataset);
        }
        
        index->Build(dataset);
        
        // 5. Save index to object storage
        SaveIndex(request.index_id, index);
        
        // 6. Notify completion
        NotifyIndexCoord(request.build_id, SUCCESS);
        
        return Status::OK();
    }
    
private:
    // GPU-accelerated index building
    Status BuildOnGPU(IndexPtr index, DatasetPtr dataset) {
        #ifdef MILVUS_GPU_VERSION
        auto gpu_index = ToGPUIndex(index);
        gpu_index->BuildGPU(dataset);
        return Status::OK();
        #else
        return Status::NotSupported("GPU not available");
        #endif
    }
};
```

## Storage Architecture

### 1. Segment Management

```go
type SegmentManager struct {
    // Segment lifecycle
    growing   map[int64]*GrowingSegment
    sealed    map[int64]*SealedSegment
    flushing  map[int64]*FlushingSegment
    
    // Memory management
    memoryPool *MemoryPool
}

// Segment state transitions
func (sm *SegmentManager) SealSegment(segmentID int64) error {
    segment := sm.growing[segmentID]
    
    // 1. Mark as sealed
    segment.Seal()
    
    // 2. Build bloom filter for deletions
    segment.BuildBloomFilter()
    
    // 3. Create binlog files
    binlogs := segment.CreateBinlogs()
    
    // 4. Upload to object storage
    sm.uploadBinlogs(binlogs)
    
    // 5. Update state
    sm.sealed[segmentID] = segment.ToSealed()
    delete(sm.growing, segmentID)
    
    return nil
}
```

### 2. Object Storage Layout

```
/milvus-bucket/
├── collection_1/
│   ├── segments/
│   │   ├── segment_1/
│   │   │   ├── fields/
│   │   │   │   ├── vector_field/
│   │   │   │   │   └── binlog_1
│   │   │   │   └── scalar_field/
│   │   │   │       └── binlog_1
│   │   │   └── indexes/
│   │   │       └── vector_field/
│   │   │           └── hnsw_index
│   │   └── segment_2/
│   │       └── ...
│   └── meta/
│       └── collection_meta.json
└── logs/
    └── wal/
        └── ...
```

### 3. Delta Management

```go
// Handle updates and deletes
type DeltaLog struct {
    SegmentID int64
    Deletes   *roaring.Bitmap
    Updates   map[int64]*UpdateRecord
}

func (d *DeltaLog) ApplyDelete(primaryKeys []int64) {
    for _, pk := range primaryKeys {
        offset := d.getOffset(pk)
        d.Deletes.Add(uint32(offset))
    }
}
```

## Query Execution

### 1. Query Planning

```go
type QueryPlan struct {
    // Vector search specification
    VectorSearch *VectorSearchPlan
    
    // Scalar filtering
    Predicates Expression
    
    // Output fields
    OutputFields []string
    
    // Execution hints
    Hints map[string]string
}

// Query optimization
func (qp *QueryPlanner) Optimize(plan *QueryPlan) *QueryPlan {
    // 1. Predicate pushdown
    plan = qp.pushDownPredicates(plan)
    
    // 2. Partition pruning
    plan = qp.prunePartitions(plan)
    
    // 3. Segment filtering
    plan = qp.filterSegments(plan)
    
    // 4. Index selection
    plan = qp.selectBestIndex(plan)
    
    return plan
}
```

### 2. Hybrid Search

```go
// Combine vector and scalar search
func (qn *QueryNode) HybridSearch(req *HybridSearchRequest) (*SearchResult, error) {
    // 1. Execute scalar filtering
    filteredSegments := qn.executePredicates(req.Predicates)
    
    // 2. Perform vector search on filtered data
    vectorResults := make([]*SegmentSearchResult, 0)
    for _, segment := range filteredSegments {
        if segment.HasIndex() {
            // Use index for search
            result := segment.SearchWithIndex(req.Vectors, req.TopK)
            vectorResults = append(vectorResults, result)
        } else {
            // Brute force search
            result := segment.BruteForceSearch(req.Vectors, req.TopK)
            vectorResults = append(vectorResults, result)
        }
    }
    
    // 3. Merge and rank results
    return qn.mergeResults(vectorResults, req.TopK)
}
```

### 3. GPU Acceleration

```cpp
// GPU-accelerated search
class GPUSearchEngine {
public:
    SearchResult Search(const SearchRequest& request) {
        // 1. Transfer data to GPU
        thrust::device_vector<float> d_query(request.query_vectors);
        thrust::device_vector<float> d_base(base_vectors_);
        
        // 2. Launch CUDA kernels
        dim3 block(256);
        dim3 grid((n_base_ + block.x - 1) / block.x);
        
        ComputeDistances<<<grid, block>>>(
            d_query.data().get(),
            d_base.data().get(),
            d_distances_.data().get(),
            n_query_, n_base_, dim_
        );
        
        // 3. Find top-k on GPU
        thrust::device_vector<int> d_indices(request.topk * n_query_);
        FindTopK<<<grid, block>>>(
            d_distances_.data().get(),
            d_indices.data().get(),
            n_base_, request.topk
        );
        
        // 4. Transfer results back
        thrust::host_vector<int> h_indices = d_indices;
        return BuildResult(h_indices);
    }
};
```

## Performance Optimizations

### 1. Memory Pool

```cpp
class MemoryPool {
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<Block> blocks_;
    std::mutex mutex_;
    
public:
    void* Allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Find reusable block
        for (auto& block : blocks_) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                return block.ptr;
            }
        }
        
        // Allocate new block
        void* ptr = std::aligned_alloc(64, size);  // 64-byte alignment
        blocks_.push_back({ptr, size, true});
        return ptr;
    }
};
```

### 2. SIMD Optimization

```cpp
// AVX-512 optimized distance computation
float L2DistanceAVX512(const float* a, const float* b, int d) {
    __m512 sum = _mm512_setzero_ps();
    
    for (int i = 0; i < d; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }
    
    return _mm512_reduce_add_ps(sum);
}
```

### 3. Cache Management

```go
type SegmentCache struct {
    cache    *lru.Cache
    capacity int64
    used     int64
}

func (sc *SegmentCache) Load(segmentID int64) (*Segment, error) {
    // Check cache
    if segment, ok := sc.cache.Get(segmentID); ok {
        return segment.(*Segment), nil
    }
    
    // Load from storage
    segment, err := sc.loadFromStorage(segmentID)
    if err != nil {
        return nil, err
    }
    
    // Add to cache with eviction
    if sc.used + segment.Size() > sc.capacity {
        sc.evict()
    }
    
    sc.cache.Add(segmentID, segment)
    sc.used += segment.Size()
    
    return segment, nil
}
```

## High Availability

### 1. Replica Management

```go
type ReplicaManager struct {
    replicas map[int64][]*Replica  // collectionID -> replicas
}

func (rm *ReplicaManager) CreateReplica(collectionID int64, num int) error {
    collection := rm.getCollection(collectionID)
    
    for i := 0; i < num; i++ {
        replica := &Replica{
            ID:           rm.allocateReplicaID(),
            CollectionID: collectionID,
            Nodes:        rm.selectNodes(collection.Segments),
        }
        
        // Distribute segments across nodes
        rm.distributeSegments(replica)
        
        rm.replicas[collectionID] = append(rm.replicas[collectionID], replica)
    }
    
    return nil
}
```

### 2. Failure Detection

```go
type HealthChecker struct {
    nodes    map[int64]*NodeInfo
    interval time.Duration
}

func (hc *HealthChecker) Monitor() {
    ticker := time.NewTicker(hc.interval)
    
    for range ticker.C {
        for nodeID, node := range hc.nodes {
            if !hc.ping(node) {
                hc.handleNodeFailure(nodeID)
            }
        }
    }
}

func (hc *HealthChecker) handleNodeFailure(nodeID int64) {
    // 1. Mark node as unhealthy
    hc.nodes[nodeID].State = NodeUnhealthy
    
    // 2. Trigger segment redistribution
    segments := hc.getNodeSegments(nodeID)
    hc.redistributeSegments(segments)
    
    // 3. Update routing table
    hc.updateRoutingTable()
}
```

## Configuration

### 1. System Configuration

```yaml
# milvus.yaml
etcd:
  endpoints:
    - localhost:2379
  
pulsar:
  address: pulsar://localhost:6650
  
minio:
  address: localhost:9000
  accessKeyID: minioadmin
  secretAccessKey: minioadmin
  
# Performance tuning
queryNode:
  cache:
    capacity: 32GB
  search:
    maxNq: 16384
    maxTopK: 16384
    
dataNode:
  segment:
    insertBufSize: 16MB
    syncPeriod: 600s
    
indexNode:
  scheduler:
    buildParallel: 4
```

### 2. Collection Configuration

```python
# Python SDK example
collection_schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]
)

index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {
        "M": 16,
        "efConstruction": 200
    }
}
```

## Monitoring and Observability

### 1. Metrics Collection

```go
type MetricsCollector struct {
    // Prometheus metrics
    searchLatency    prometheus.Histogram
    searchThroughput prometheus.Counter
    indexBuildTime   prometheus.Histogram
    segmentCount     prometheus.Gauge
}

func (mc *MetricsCollector) RecordSearch(duration time.Duration, numVectors int) {
    mc.searchLatency.Observe(duration.Seconds())
    mc.searchThroughput.Add(float64(numVectors))
}
```

### 2. Distributed Tracing

```go
func (qn *QueryNode) SearchWithTracing(ctx context.Context, req *SearchRequest) (*SearchResult, error) {
    span, ctx := opentracing.StartSpanFromContext(ctx, "vector_search")
    defer span.Finish()
    
    span.SetTag("collection", req.CollectionName)
    span.SetTag("topk", req.TopK)
    span.SetTag("nq", len(req.Vectors))
    
    // Trace each phase
    loadSpan := opentracing.StartSpan("load_segments", opentracing.ChildOf(span.Context()))
    segments := qn.loadSegments(req.CollectionID)
    loadSpan.Finish()
    
    searchSpan := opentracing.StartSpan("search_segments", opentracing.ChildOf(span.Context()))
    results := qn.searchSegments(segments, req)
    searchSpan.Finish()
    
    return results, nil
}
```

## Summary

Milvus architecture demonstrates:
1. **Cloud-Native Design**: Kubernetes-ready with horizontal scalability
2. **Separation of Concerns**: Storage, compute, and coordination layers
3. **Stream Processing**: Real-time data ingestion with consistency
4. **Multiple Index Types**: HNSW, IVF, DiskANN for different use cases
5. **Production Features**: HA, monitoring, GPU support, and enterprise readiness

The architecture enables Milvus to handle billion-scale vector datasets while maintaining sub-second query latency and high availability.