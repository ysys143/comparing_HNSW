# Elasticsearch HNSW Implementation Analysis

## Overview

Elasticsearch implements HNSW through Apache Lucene's KnnVectorQuery, providing distributed vector search capabilities with automatic sharding and replication. The implementation focuses on scalability and integration with Elasticsearch's existing search infrastructure.

## Core Implementation

### 1. **Lucene HNSW Structure**

```java
// lucene/core/src/java/org/apache/lucene/index/KnnGraphValues.java
public abstract class KnnGraphValues {
    protected final int size;
    
    public abstract void seek(int level, int node) throws IOException;
    public abstract int nextNeighbor() throws IOException;
    public abstract int numLevels() throws IOException;
    public abstract int entryNode() throws IOException;
}

// lucene/core/src/java/org/apache/lucene/util/hnsw/HnswGraph.java
public final class HnswGraph extends KnnGraphValues {
    private final int numLevels;
    private final int entryNode;
    private final NodesByLevel nodesByLevel[];
    private final long[] graphOffsetsByLevel;
    
    // Graph structure with level-wise node storage
    static class NodesByLevel {
        private final int level;
        private final int[] nodes;
        private final IntIntHashMap nodeIndexToOrdinal;
        
        NodesByLevel(int level, int[] nodes) {
            this.level = level;
            this.nodes = nodes;
            this.nodeIndexToOrdinal = new IntIntHashMap(nodes.length);
            for (int i = 0; i < nodes.length; i++) {
                nodeIndexToOrdinal.put(nodes[i], i);
            }
        }
    }
}
```

### 2. **Index Building**

```java
// lucene/core/src/java/org/apache/lucene/util/hnsw/HnswGraphBuilder.java
public final class HnswGraphBuilder {
    private final int maxConn;
    private final int beamWidth;
    private final VectorSimilarityFunction similarityFunction;
    private final RandomVectorScorer scorer;
    private final HnswGraphSearcher searcher;
    
    // Scratch data structures for building
    private final SparseFixedBitSet visitedNodes;
    private final NeighborQueue beamCandidates;
    
    public OnHeapHnswGraph build(int numVectors) throws IOException {
        OnHeapHnswGraph graph = new OnHeapHnswGraph(maxConn, numVectors);
        
        for (int node = 0; node < numVectors; node++) {
            addNode(graph, node);
            
            if ((node + 1) % 10000 == 0) {
                // Progress reporting
                logger.info("Added {} nodes to HNSW graph", node + 1);
            }
        }
        
        return graph;
    }
    
    private void addNode(OnHeapHnswGraph graph, int node) throws IOException {
        int level = assignLevel(node, random);
        graph.addNode(level, node);
        
        // Find nearest neighbors at each level
        for (int lc = 0; lc <= level; lc++) {
            int entryPoint = graph.entryNode();
            
            NeighborQueue candidates = searcher.searchLevel(
                scorer.scorer(node),
                beamWidth,
                lc,
                entryPoint,
                graph,
                visitedNodes
            );
            
            int maxConnOnLevel = lc == 0 ? maxConn * 2 : maxConn;
            selectAndLinkNeighbors(graph, node, candidates, maxConnOnLevel, lc);
        }
    }
    
    // Neighbor selection with pruning
    private void selectAndLinkNeighbors(
            OnHeapHnswGraph graph,
            int node,
            NeighborQueue candidates,
            int maxConn,
            int level) {
        
        // Select diverse neighbors using heuristic algorithm
        NeighborArray neighbors = selectDiverse(candidates, maxConn);
        
        // Add bidirectional links
        for (int i = 0; i < neighbors.size(); i++) {
            int neighbor = neighbors.node[i];
            graph.addEdge(level, node, neighbor);
            graph.addEdge(level, neighbor, node);
            
            // Prune neighbor's connections if needed
            if (graph.numNeighbors(level, neighbor) > maxConn) {
                pruneConnections(graph, level, neighbor, maxConn);
            }
        }
    }
}
```

### 3. **Search Implementation**

```java
// lucene/core/src/java/org/apache/lucene/util/hnsw/HnswGraphSearcher.java
public class HnswGraphSearcher {
    
    public static NeighborQueue search(
            RandomVectorScorer scorer,
            int topK,
            HnswGraph graph,
            BitSet acceptOrds,
            int visitedLimit) throws IOException {
        
        int entryNode = graph.entryNode();
        if (entryNode == -1) {
            return new NeighborQueue(1, false);
        }
        
        // Start from top layer
        int numLevels = graph.numLevels();
        BitSet visited = new FixedBitSet(graph.size());
        
        // Search top layers - find closest entry point
        for (int level = numLevels - 1; level >= 1; level--) {
            float[] candidates = searchLayer(
                scorer, 
                graph, 
                level, 
                new int[]{entryNode},
                visited
            );
            entryNode = findClosest(candidates);
        }
        
        // Search bottom layer with full candidate set
        NeighborQueue results = new NeighborQueue(topK, false);
        searchLevel(
            scorer,
            topK,
            0,
            entryNode,
            graph,
            visited,
            acceptOrds,
            visitedLimit,
            results
        );
        
        return results;
    }
    
    private static void searchLevel(
            RandomVectorScorer scorer,
            int topK,
            int level,
            int entryPoint,
            HnswGraph graph,
            BitSet visited,
            BitSet acceptOrds,
            int visitedLimit,
            NeighborQueue results) throws IOException {
        
        NeighborQueue candidates = new NeighborQueue(topK, false);
        float score = scorer.score(entryPoint);
        candidates.add(entryPoint, score);
        visited.set(entryPoint);
        results.add(entryPoint, score);
        
        int visitedCount = 1;
        
        while (candidates.size() > 0 && visitedCount < visitedLimit) {
            float lowerBound = results.topScore();
            int node = candidates.pop();
            
            if (candidates.topScore() < lowerBound) {
                break;
            }
            
            graph.seek(level, node);
            int neighbor;
            while ((neighbor = graph.nextNeighbor()) != -1) {
                if (visited.get(neighbor)) {
                    continue;
                }
                
                visited.set(neighbor);
                visitedCount++;
                
                float neighborScore = scorer.score(neighbor);
                if (neighborScore > lowerBound || results.size() < topK) {
                    candidates.add(neighbor, neighborScore);
                    if (acceptOrds == null || acceptOrds.get(neighbor)) {
                        results.add(neighbor, neighborScore);
                    }
                }
            }
        }
    }
}
```

### 4. **Elasticsearch Integration**

```java
// x-pack/plugin/vectors/src/main/java/org/elasticsearch/xpack/vectors/query/KnnVectorQueryBuilder.java
public class KnnVectorQueryBuilder extends AbstractQueryBuilder<KnnVectorQueryBuilder> {
    private final String fieldName;
    private final float[] queryVector;
    private final int numCandidates;
    private final Float similarity;
    
    @Override
    protected Query doToQuery(SearchExecutionContext context) throws IOException {
        MappedFieldType fieldType = context.getFieldType(fieldName);
        if (!(fieldType instanceof DenseVectorFieldType)) {
            throw new IllegalArgumentException("Field '" + fieldName + "' is not a dense_vector field");
        }
        
        DenseVectorFieldType vectorFieldType = (DenseVectorFieldType) fieldType;
        
        // Create Lucene KNN query
        return new KnnFloatVectorQuery(
            fieldName,
            queryVector,
            numCandidates,
            buildFilter(context)
        );
    }
}

// Mapping configuration
public class DenseVectorFieldMapper extends FieldMapper {
    public static class Builder extends FieldMapper.Builder {
        private final Parameter<Integer> dims;
        private final Parameter<VectorSimilarity> similarity;
        private final Parameter<IndexOptions> indexOptions;
        
        public Builder(String name) {
            super(name);
            this.dims = Parameter.intParam("dims", false, m -> {}, 1, 2048);
            this.similarity = Parameter.enumParam(
                "similarity", 
                VectorSimilarity.class,
                m -> {},
                VectorSimilarity.COSINE
            );
            this.indexOptions = new Parameter<>(
                "index_options",
                IndexOptions::parse,
                m -> {},
                () -> null
            );
        }
        
        @Override
        public DenseVectorFieldMapper build(MapperBuilderContext context) {
            return new DenseVectorFieldMapper(
                name,
                new DenseVectorFieldType(
                    name,
                    dims.getValue(),
                    similarity.getValue(),
                    indexOptions.getValue()
                ),
                multiFieldsBuilder.build(this, context),
                copyTo
            );
        }
    }
}
```

### 5. **Distributed Search**

```java
// server/src/main/java/org/elasticsearch/search/vectors/KnnSearchBuilder.java
public class KnnSearchBuilder implements Writeable, ToXContentObject {
    private final String field;
    private final float[] queryVector;
    private final int k;
    private final int numCandidates;
    private final Float similarity;
    
    public SearchRequest buildSearchRequest(String index) {
        SearchSourceBuilder searchSource = new SearchSourceBuilder();
        
        // Add KNN section
        searchSource.knnSearch(Arrays.asList(
            new KnnSearchBuilder(field, queryVector, k, numCandidates)
        ));
        
        // Add any filters
        if (filterQuery != null) {
            searchSource.query(filterQuery);
        }
        
        return new SearchRequest(index).source(searchSource);
    }
    
    // Merge results from multiple shards
    public static List<KnnSearchHit> mergeKnnResults(
            List<List<KnnSearchHit>> shardResults,
            int k) {
        
        PriorityQueue<KnnSearchHit> mergedResults = new PriorityQueue<>(
            k,
            Comparator.comparing(KnnSearchHit::score).reversed()
        );
        
        for (List<KnnSearchHit> shardHits : shardResults) {
            for (KnnSearchHit hit : shardHits) {
                mergedResults.offer(hit);
                if (mergedResults.size() > k) {
                    mergedResults.poll();
                }
            }
        }
        
        List<KnnSearchHit> results = new ArrayList<>(mergedResults);
        Collections.reverse(results);
        return results;
    }
}
```

### 6. **Performance Optimizations**

```java
// Vector encoding and quantization
public class VectorEncoder {
    // Byte quantization for reduced memory
    public static byte[] encodeFloatVectorAsByte(float[] vector) {
        byte[] encoded = new byte[vector.length];
        
        // Find min/max for normalization
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        for (float v : vector) {
            min = Math.min(min, v);
            max = Math.max(max, v);
        }
        
        float scale = (max - min) / 255f;
        
        // Quantize to bytes
        for (int i = 0; i < vector.length; i++) {
            encoded[i] = (byte) ((vector[i] - min) / scale);
        }
        
        return encoded;
    }
}

// Panama Vector API optimizations
public class SimdVectorOperations {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    public static float dotProduct(float[] a, float[] b) {
        if (VECTOR_ACCESS_OOB_ENABLED) {
            return simdDotProduct(a, b);
        }
        return scalarDotProduct(a, b);
    }
    
    private static float simdDotProduct(float[] a, float[] b) {
        FloatVector sum = FloatVector.zero(SPECIES);
        int i = 0;
        int bound = SPECIES.loopBound(a.length);
        
        for (; i < bound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            sum = va.fma(vb, sum);
        }
        
        float result = sum.reduceLanes(VectorOperators.ADD);
        
        // Tail handling
        for (; i < a.length; i++) {
            result += a[i] * b[i];
        }
        
        return result;
    }
}
```

### 7. **Index Management**

```java
// Index settings and lifecycle
public class VectorIndexSettings {
    // HNSW parameters
    public static final Setting<Integer> HNSW_M = Setting.intSetting(
        "index.knn.algo_param.m",
        16,
        2,
        Property.IndexScope
    );
    
    public static final Setting<Integer> HNSW_EF_CONSTRUCTION = Setting.intSetting(
        "index.knn.algo_param.ef_construction",
        200,
        2,
        Property.IndexScope
    );
    
    public static final Setting<Integer> HNSW_EF_SEARCH = Setting.intSetting(
        "index.knn.algo_param.ef_search",
        100,
        1,
        Property.IndexScope,
        Property.Dynamic
    );
}

// Force merge for optimal graph structure
public class VectorIndexOptimizer {
    public void optimizeIndex(String indexName) {
        ForceMergeRequest request = new ForceMergeRequest(indexName);
        request.maxNumSegments(1);  // Single segment for best performance
        request.flush(true);
        
        client.admin().indices().forceMerge(request, new ActionListener<>() {
            @Override
            public void onResponse(ForceMergeResponse response) {
                logger.info("Vector index {} optimized successfully", indexName);
            }
        });
    }
}
```

## Configuration Options

### Index Mapping
```json
{
  "mappings": {
    "properties": {
      "embedding": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "cosine",
        "index_options": {
          "type": "hnsw",
          "m": 16,
          "ef_construction": 200
        }
      }
    }
  }
}
```

### Search Configuration
```json
{
  "knn": {
    "field": "embedding",
    "query_vector": [0.1, 0.2, ...],
    "k": 10,
    "num_candidates": 100,
    "filter": {
      "term": { "category": "electronics" }
    }
  },
  "_source": ["title", "description"]
}
```

## Performance Characteristics

### Advantages
- Distributed search across shards
- Automatic failover and replication
- Integration with Elasticsearch features
- Segment-based storage efficiency

### Trade-offs
- JVM overhead
- Segment merging impacts
- Limited to Lucene's HNSW implementation
- No GPU acceleration

## Best Practices

### 1. **Index Design**
```java
// Optimal shard configuration
int numShards = calculateOptimalShards(numVectors, vectorDimensions);
int numReplicas = clusterSize > 1 ? 1 : 0;

// Settings for large-scale deployments
Settings indexSettings = Settings.builder()
    .put("index.number_of_shards", numShards)
    .put("index.number_of_replicas", numReplicas)
    .put("index.refresh_interval", "30s")  // Reduce refresh overhead
    .put("index.knn.algo_param.m", 32)     // Higher M for better recall
    .put("index.knn.algo_param.ef_construction", 400)
    .build();
```

### 2. **Query Optimization**
```java
// Pre-filter optimization
QueryBuilder preFilter = QueryBuilders.boolQuery()
    .must(QueryBuilders.termQuery("status", "active"))
    .filter(QueryBuilders.rangeQuery("price").gte(100).lte(1000));

KnnSearchBuilder knnSearch = new KnnSearchBuilder(
    "embedding",
    queryVector,
    50,  // k
    200  // num_candidates = 4 * k for good recall
).filter(preFilter);
```

### 3. **Memory Management**
```yaml
# JVM settings for vector search
-Xms31g
-Xmx31g
-XX:+UseG1GC
-XX:MaxGCPauseMillis=200
-XX:+ParallelRefProcEnabled
-XX:MaxDirectMemorySize=16g
```

## Code References

### Core Files
- `lucene/core/src/java/org/apache/lucene/util/hnsw/` - HNSW implementation
- `lucene/core/src/java/org/apache/lucene/index/KnnVectorQuery.java` - Query interface
- `x-pack/plugin/vectors/` - Elasticsearch vector plugin
- `server/src/main/java/org/elasticsearch/search/vectors/` - Distributed search

## Comparison Notes
- Leverages mature Lucene implementation
- Strong distributed capabilities
- Limited customization compared to specialized databases
- Trade-off: Ecosystem integration vs. vector-specific optimizations