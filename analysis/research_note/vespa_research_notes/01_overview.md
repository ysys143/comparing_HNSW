# Vespa Overview

## Project Information

- **Language**: C++ (core engine) + Java (container/API layer)
- **License**: Apache License 2.0
- **Repository**: https://github.com/vespa-engine/vespa
- **Analysis Version**: Commit be9e9cbd32cc3974289b78b5d9e6394638d56a9e (2025-06-21)

## System Architecture Overview

Vespa is a large-scale search and serving engine that provides tensor computation and nearest neighbor search capabilities alongside traditional search features. It features a production-grade HNSW implementation deeply integrated with its distributed serving infrastructure.

### Key Components

1. **SearchLib** (`/searchlib/`)
   - Core C++ search library
   - HNSW index implementation
   - Tensor storage and operations
   - Distance metric implementations

2. **Tensor/Vector Components** (`/searchlib/src/vespa/searchlib/tensor/`)
   - `hnsw_index`: Main HNSW implementation
   - `hnsw_graph`: Graph structure management
   - Dense and sparse tensor stores
   - Multiple distance functions

3. **Container Layer** (Java)
   - HTTP/REST API handling
   - Query processing
   - Result rendering
   - Configuration management

4. **Storage & Distribution**
   - Document store with sharding
   - Attribute-based storage
   - Generation-based concurrency control
   - Memory-mapped files

## Core Features

### HNSW Implementation
- **Configurable Parameters**:
  - `max_links_at_level_0`: Base level connectivity
  - `max_links_on_inserts`: New node connections
  - `neighbors_to_explore_at_construction`: Build-time exploration
  - `heuristic_select_neighbors`: Smart neighbor selection
- **Two-phase Optimization**: For larger graphs
- **Lock-free Operations**: Using generation tracking
- **Node ID Mapping**: Flexible document management

### Distance Metrics
- Angular distance (cosine similarity)
- Euclidean distance
- Geo-degrees distance
- Hamming distance
- Prenormalized angular distance

### Tensor Operations
- **Dense Tensors**: Optimized for vectors
- **Sparse Tensors**: For high-dimensional sparse data
- **Mixed Tensors**: Combination of dense/sparse
- **Tensor Expressions**: Query-time computations

### System Features
- **Distributed Search**: Multi-node deployments
- **Real-time Updates**: Document streaming
- **Ranking Framework**: Multi-phase ranking
- **Query Language**: YQL (Vespa Query Language)
- **Schema Management**: Strong typing for fields

## API Design

### REST API
```http
POST /search/
{
  "yql": "select * from sources * where {targetHits: 10}nearestNeighbor(embedding, query_embedding)",
  "query_embedding": [0.1, 0.2, 0.3, ...],
  "ranking.profile": "similarity"
}
```

### Document API
```http
PUT /document/v1/namespace/doctype/docid/1
{
  "fields": {
    "title": "Example",
    "embedding": [0.1, 0.2, 0.3, ...]
  }
}
```

### Configuration
```xml
<field name="embedding" type="tensor<float>(x[384])" indexing="attribute | index">
  <attribute>
    <distance-metric>angular</distance-metric>
  </attribute>
  <index>
    <hnsw>
      <max-links-per-node>16</max-links-per-node>
      <neighbors-to-explore-at-insert>200</neighbors-to-explore-at-insert>
    </hnsw>
  </index>
</field>
```

## Technical Highlights

1. **Performance Optimizations**
   - Lock-free data structures
   - Generation-based concurrency
   - Memory-mapped storage
   - SIMD optimizations

2. **Production Features**
   - Graceful updates
   - Online reindexing
   - Monitoring and metrics
   - Resource management

3. **Integration**
   - Tensor computations in ranking
   - Hybrid search (keywords + vectors)
   - Multi-modal search
   - Streaming search

4. **Scalability**
   - Horizontal scaling
   - Data distribution
   - Query routing
   - Load balancing

## Notable Design Decisions

1. **Integrated Platform**: Vectors as first-class citizens in a search platform
2. **Generation-Based Concurrency**: Lock-free reads during updates
3. **C++/Java Split**: Performance-critical code in C++, orchestration in Java
4. **Schema-Driven**: Strong typing and configuration
5. **Multi-Phase Ranking**: Combine multiple signals including vectors

## Analysis Focus Areas

Based on the initial investigation:

1. **HNSW Implementation Details**
2. **Generation-Based Concurrency Model**
3. **Tensor Storage Architecture**
4. **Distance Function Optimizations**
5. **Integration with Search Pipeline**
6. **Distributed Vector Search**

## Next Steps

1. Detailed code structure analysis (Phase 2)
2. Deep dive into HNSW implementation
3. Analysis of generation-based concurrency
4. Tensor storage examination
5. Performance profiling setup