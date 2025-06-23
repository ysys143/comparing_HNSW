# Elasticsearch Overview

## Project Information

- **Language**: Java (primary implementation)
- **License**: Triple license - GNU AGPLv3, SSPL v1, Elastic License 2.0
- **Repository**: https://github.com/elastic/elasticsearch
- **Analysis Version**: Commit cb451dac0b46ab0ad05b792dd7365679eb2c946d (2025-06-20)

## System Architecture Overview

Elasticsearch is a mature, distributed search and analytics engine that has evolved to include sophisticated vector search capabilities. Originally built on Apache Lucene for text search, it now provides first-class support for dense vector operations through deep integration with Lucene's vector search features.

### Key Components

1. **Core Server** (`/server/`)
   - Vector field mapping (`dense_vector` type)
   - Vector search queries and scoring
   - Custom vector codecs
   - Integration with Lucene's HNSW implementation

2. **Vector Index Types**
   - **HNSW** (default): Standard and quantized variants
     - `hnsw`: Standard HNSW
     - `int8_hnsw`: 8-bit quantization (default)
     - `int4_hnsw`: 4-bit quantization
     - `bbq_hnsw`: Binary quantization
   - **Flat**: Exact search variants
     - `flat`: Standard exact search
     - `int8_flat`: 8-bit quantized
     - `int4_flat`: 4-bit quantized
     - `bbq_flat`: Binary quantized
   - **IVF**: Inverted file (experimental)

3. **Storage Layer**
   - Lucene-based vector storage
   - Custom codecs for different formats
   - Support for up to 4096 dimensions (float/byte)
   - Magnitude caching for cosine similarity

## Core Features

### Vector Search Capabilities
- **Similarity Metrics**: 
  - Cosine similarity
  - Dot product
  - L2 norm (Euclidean distance)
  - Max inner product
- **Search Types**:
  - Approximate kNN (HNSW-based)
  - Exact kNN (brute-force)
  - Hybrid search (combine with text/filters)
- **Performance Optimizations**:
  - Multiple quantization levels
  - Filter heuristics (FANOUT, ACORN)
  - Rescoring for quantized vectors

### Integration Features
- Seamless integration with existing search API
- Query-time vector construction
- Script-based vector generation
- Combined text and vector search

### Scalability
- Distributed vector search across shards
- Segment-level optimization
- Configurable indexing parameters
- Memory-efficient quantization options

## API Design

### Search API
- Vector search through standard `/_search` endpoint
- `knn` clause for k-nearest neighbor queries
- Support for pre-filtering and post-filtering
- Hybrid scoring with other query types

### Index Configuration
```json
{
  "mappings": {
    "properties": {
      "vector_field": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "cosine",
        "index_options": {
          "type": "int8_hnsw",
          "m": 16,
          "ef_construction": 100
        }
      }
    }
  }
}
```

## Technical Highlights

1. **Lucene Integration**
   - Leverages Lucene's vector search capabilities
   - Custom extensions for Elasticsearch-specific features
   - Codec-level optimizations

2. **Quantization Support**
   - Multiple quantization levels (8-bit, 4-bit, binary)
   - Automatic rescoring for accuracy
   - Configurable trade-offs

3. **Production Readiness**
   - Mature codebase with extensive testing
   - Backwards compatibility guarantees
   - Feature flags for experimental features
   - Comprehensive monitoring and metrics

## Notable Design Decisions

1. **First-Class Vector Type**: `dense_vector` as a native field type
2. **Quantization by Default**: int8_hnsw as default for balance of speed/accuracy
3. **Unified Search API**: Vector search integrated into main search endpoint
4. **Lucene Foundation**: Building on proven search infrastructure
5. **Flexible Index Options**: Multiple index types for different use cases

## Analysis Focus Areas

Based on the initial investigation, the following areas warrant deeper analysis:

1. **HNSW Implementation** (via Lucene integration)
2. **Quantization Techniques** (int8, int4, binary)
3. **Filter Integration** (FANOUT vs ACORN heuristics)
4. **Distributed Search Coordination**
5. **Memory Management and Caching**
6. **Performance Characteristics**

## Next Steps

1. Detailed code structure analysis (Phase 2)
2. Deep dive into vector codec implementations
3. Analysis of quantization algorithms
4. Performance profiling setup
5. Distributed search flow examination