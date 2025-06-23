# Qdrant Overview

## Project Information

- **Language**: Rust (edition 2024)
- **License**: Apache License 2.0
- **Repository**: https://github.com/qdrant/qdrant
- **Analysis Version**: Commit 530430fac2a3ca872504f276d2c91a5c91f43fa0 (2025-05-23)

## System Architecture Overview

Qdrant is a vector similarity search engine built in Rust, designed for production use with a focus on performance, scalability, and advanced filtering capabilities. It features a sophisticated HNSW implementation with dynamic strategy selection for filtered searches.

### Key Components

1. **Core Segment Library** (`/lib/segment/`)
   - HNSW index implementation with filterable search
   - Vector storage with multiple backends
   - Payload indexing for structured data
   - Distance metrics and SIMD optimizations

2. **Index Types**
   - **HNSW**: Primary index with graph-based search
   - **Plain**: Brute-force for small datasets
   - **Sparse**: For sparse vector support
   - **Payload Indices**: For filtering on metadata

3. **Storage Layer**
   - In-memory storage
   - Memory-mapped files (mmap)
   - RocksDB for persistence
   - Quantization support (scalar, product, binary)

4. **API Layer**
   - REST API (Actix-web)
   - gRPC API (Tonic)
   - Batch operations
   - Streaming support

## Core Features

### Vector Capabilities
- **Vector Types**: Dense (f32, f16, uint8), Sparse, Multi-vectors
- **Dimensions**: Up to 65,536 dimensions
- **Distance Metrics**: Cosine, Euclidean, Dot Product, Manhattan
- **Quantization**: Scalar, Product, Binary quantization

### Advanced Filtering
- **Dynamic Strategy Selection**: Automatically chooses between pre/post-filtering
- **Cardinality Estimation**: Statistical sampling for filter selectivity
- **Payload-Based Subgraphs**: Pre-built HNSW graphs for common filters
- **Integrated Filtering**: Filter checking during graph traversal

### HNSW Implementation Features
- **Configurable Parameters**: M, ef_construction, ef_search
- **GPU Acceleration**: Optional GPU support for index building
- **Quantized Search**: Search on quantized vectors with rescoring
- **Filter-Aware Traversal**: Early termination based on filters

### System Features
- **Distributed Support**: Raft-based consensus for clustering
- **Snapshots**: Point-in-time backups
- **Collections**: Logical data separation
- **Sharding**: Horizontal scaling
- **Telemetry**: Built-in performance monitoring

## API Design

### REST API
```http
PUT /collections/{collection_name}/points
{
  "points": [
    {
      "id": 1,
      "vector": [0.1, 0.2, 0.3],
      "payload": {"city": "Berlin", "price": 100}
    }
  ]
}

POST /collections/{collection_name}/points/search
{
  "vector": [0.1, 0.2, 0.3],
  "filter": {
    "must": [
      {"key": "city", "match": {"value": "Berlin"}},
      {"key": "price", "range": {"gte": 50, "lte": 200}}
    ]
  },
  "limit": 10
}
```

### gRPC API
- Protocol buffer definitions
- Streaming support
- Binary efficiency
- Type safety

## Technical Highlights

1. **Filterable HNSW Innovation**
   - Dynamic pre/post-filtering decision
   - Statistical cardinality estimation
   - Payload-aware index building
   - Integrated filter evaluation

2. **Performance Optimizations**
   - SIMD instructions (AVX, SSE, NEON)
   - Memory-mapped storage
   - Quantization techniques
   - GPU acceleration

3. **Production Features**
   - Graceful error handling
   - Comprehensive telemetry
   - Snapshot/restore
   - Rolling updates

4. **Rust Benefits**
   - Memory safety without GC
   - Zero-cost abstractions
   - Excellent concurrency
   - Cross-platform support

## Notable Design Decisions

1. **Rust Implementation**: Performance and safety without garbage collection
2. **Dynamic Filtering Strategy**: Adaptive approach based on query characteristics
3. **Payload Indexing**: First-class support for metadata filtering
4. **Multiple Storage Backends**: Flexibility for different use cases
5. **Quantization Options**: Trade-off between memory and accuracy

## Analysis Focus Areas

Based on the initial investigation and existing documentation:

1. **HNSW Filterable Implementation** (primary focus)
2. **Cardinality Estimation Algorithm**
3. **Payload-Based Subgraph Generation**
4. **Quantization Techniques**
5. **GPU Acceleration Architecture**
6. **Distributed Consensus Implementation**

## Next Steps

1. Detailed code structure analysis (Phase 2)
2. Deep dive into HNSW filterable implementation
3. Analysis of cardinality estimation
4. Payload indexing strategy examination
5. Performance profiling setup