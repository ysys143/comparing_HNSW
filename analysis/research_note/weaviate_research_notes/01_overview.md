# Weaviate Overview

## Project Information

- **Language**: Go (version 1.24.0)
- **License**: BSD 3-Clause License
- **Repository**: https://github.com/weaviate/weaviate
- **Analysis Version**: Commit 1b34fac7b0a46322cb94b7e34228941ee207bc04 (2025-06-20)

## System Architecture Overview

Weaviate is a cloud-native vector database built in Go, designed for semantic search with a strong focus on extensibility through modules. It features a custom HNSW implementation with advanced compression techniques and dynamic indexing strategies.

### Key Components

1. **Vector Index Layer** (`/adapters/repos/db/vector/`)
   - **HNSW**: Primary index with extensive optimizations
   - **Flat**: Brute-force search for small datasets
   - **Dynamic**: Automatic switching between flat and HNSW
   - Compression: PQ, BQ, SQ implementations

2. **Storage Layer**
   - **LSMKV**: Custom Log-Structured Merge Key-Value store
   - **Inverted Index**: For text and property filtering
   - **Roaring Bitmaps**: Efficient filter representation
   - **Segment Strategies**: Multiple storage optimization approaches

3. **API Layer**
   - REST API with OpenAPI specification
   - GraphQL for flexible queries
   - gRPC for high-performance operations
   - Batch import capabilities

4. **Module System**
   - 30+ vectorizer modules (OpenAI, Cohere, Hugging Face, etc.)
   - Generative AI integrations
   - Reranker modules
   - Custom transformers

## Core Features

### HNSW Implementation
- **Configurable Parameters**:
  - `efConstruction`: Build-time search width
  - `ef`: Query-time search width (dynamic adjustment)
  - `maxConnections`: Graph connectivity
  - `cleanupIntervalSeconds`: Tombstone cleanup
- **Filter Strategies**:
  - Sweeping: Filter during graph traversal
  - Acorn: Pre-filter before search
- **Compression Support**: PQ, BQ, SQ for memory efficiency
- **SIMD Optimizations**: AMD64 and ARM64 specific implementations

### Vector Capabilities
- **Distance Metrics**: Cosine, Dot, L2-squared, Manhattan, Hamming
- **Multi-vector Support**: Muvera algorithm for multiple vectors per object
- **Dimensions**: Up to 65,536 dimensions
- **Data Types**: float32, with compression to various formats

### Storage Features
- **LSMKV Store**: Optimized for write-heavy workloads
- **Segment Strategies**: Replace/delete optimization
- **Memory Mapping**: Efficient disk-based operations
- **Backup Support**: S3, Azure, GCS, filesystem

### System Features
- **Multi-tenancy**: Isolated tenant data
- **Clustering**: Raft-based consensus
- **Sharding**: Horizontal scaling
- **Schema Management**: Strong typing with auto-schema
- **Monitoring**: Prometheus metrics, tracing

## API Design

### REST API
```http
POST /v1/objects
{
  "class": "Article",
  "properties": {
    "title": "Example",
    "content": "..."
  },
  "vector": [0.1, 0.2, 0.3, ...]
}

GET /v1/objects?nearVector={"vector":[0.1,0.2,0.3]}&limit=10
```

### GraphQL API
```graphql
{
  Get {
    Article(
      nearVector: {vector: [0.1, 0.2, 0.3]}
      limit: 10
    ) {
      title
      content
      _additional {
        distance
      }
    }
  }
}
```

### Batch Operations
```http
POST /v1/batch/objects
{
  "objects": [
    {
      "class": "Article",
      "properties": {...},
      "vector": [...]
    }
  ]
}
```

## Technical Highlights

1. **Performance Optimizations**
   - SIMD instructions for distance calculations
   - Vector compression techniques
   - Efficient filtering with roaring bitmaps
   - Cache prefetching for vectors

2. **Extensibility**
   - Pluggable vectorizer modules
   - Custom distance metrics
   - Module lifecycle management
   - API for custom modules

3. **Production Features**
   - Comprehensive monitoring
   - Graceful shutdown
   - Resource management
   - Multi-level caching

4. **Advanced Capabilities**
   - Hybrid search (vector + keyword)
   - Generative search
   - Cross-reference support
   - Automatic schema inference

## Notable Design Decisions

1. **Go Implementation**: Balance of performance and development speed
2. **Module System**: Extensibility without core modifications
3. **Custom LSMKV**: Optimized for vector workload patterns
4. **Dynamic Indexing**: Automatic optimization based on dataset size
5. **Filter Strategies**: Multiple approaches for different use cases

## Analysis Focus Areas

Based on the initial investigation:

1. **HNSW Implementation Details**
2. **Compression Techniques (PQ, BQ, SQ)**
3. **Dynamic Index Selection**
4. **LSMKV Storage Engine**
5. **Filter Strategy Selection**
6. **SIMD Optimizations**

## Next Steps

1. Detailed code structure analysis (Phase 2)
2. Deep dive into HNSW implementation
3. Analysis of compression algorithms
4. Storage engine examination
5. Performance profiling setup