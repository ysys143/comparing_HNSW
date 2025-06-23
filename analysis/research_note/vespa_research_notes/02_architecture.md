# Vespa Architecture Analysis

## Overview

Vespa is a real-time big data serving engine that provides integrated search, recommendation, and ranking capabilities. It implements HNSW for approximate nearest neighbor search while maintaining its core strengths in structured search, real-time indexing, and complex ranking. Vespa's architecture emphasizes low-latency serving at scale with automatic data distribution and fault tolerance.

## System Architecture

### Distributed Architecture

```
┌─────────────────────────────────────────────┐
│            Application Layer                 │
│      (HTTP API, Query Language)              │
├─────────────────────────────────────────────┤
│           Container Cluster                  │
│  ┌─────────────────────────────────────┐    │
│  │       Query Processing               │    │
│  │   (Parsing, Planning, Routing)       │    │
│  ├─────────────────────────────────────┤    │
│  │        Searchers Chain               │    │
│  │   (Query Rewriting, Federation)      │    │
│  ├─────────────────────────────────────┤    │
│  │      Document Processing             │    │
│  │   (Feed Handling, Transformation)    │    │
│  ├─────────────────────────────────────┤    │
│  │        Ranking Framework             │    │
│  │   (ML Models, Feature Extraction)    │    │
│  └─────────────────────────────────────┘    │
├─────────────────────────────────────────────┤
│           Content Cluster                    │
│  ┌─────────────────────────────────────┐    │
│  │        Distributors                  │    │
│  │   (Document Routing, Consistency)    │    │
│  ├─────────────────────────────────────┤    │
│  │      Content Nodes                   │    │
│  │  ┌─────────────────────────────┐    │    │
│  │  │    Proton Search Core        │    │    │
│  │  │  (Memory Index, Disk Index)  │    │    │
│  │  ├─────────────────────────────┤    │    │
│  │  │    HNSW Implementation       │    │    │
│  │  │  (Native C++ Integration)    │    │    │
│  │  ├─────────────────────────────┤    │    │
│  │  │    Attribute Store           │    │    │
│  │  │  (In-memory Field Storage)   │    │    │
│  │  ├─────────────────────────────┤    │    │
│  │  │    Document Store            │    │    │
│  │  │  (Persistent Storage)        │    │    │
│  │  └─────────────────────────────┘    │    │
│  └─────────────────────────────────────┘    │
├─────────────────────────────────────────────┤
│           Configuration Cluster              │
│      (ZooKeeper, Config Servers)             │
└─────────────────────────────────────────────┘
```

## Core Components

### 1. Document Model and Tensor Support

**Schema Definition**:
```xml
<schema name="products" version="1.0">
  <document>
    <field name="id" type="string" indexing="summary | attribute" />
    <field name="title" type="string" indexing="index | summary" />
    <field name="embedding" type="tensor&lt;float&gt;(x[384])" 
           indexing="attribute | index">
      <attribute>
        <distance-metric>euclidean</distance-metric>
      </attribute>
      <index>
        <hnsw>
          <max-links-per-node>16</max-links-per-node>
          <neighbors-to-explore-at-insert>200</neighbors-to-explore-at-insert>
        </hnsw>
      </index>
    </field>
  </document>
  
  <rank-profile name="similarity" inherits="default">
    <inputs>
      <query(query_embedding) tensor&lt;float&gt;(x[384]) />
    </inputs>
    <first-phase>
      <expression>closeness(field, embedding)</expression>
    </first-phase>
  </rank-profile>
</schema>
```

**Tensor Operations**:
```java
// Vespa's tensor algebra system
public class TensorType {
    private final List<Dimension> dimensions;
    
    public static class Dimension {
        private final String name;
        private final Optional<Long> size;  // Indexed dimension
        private final Optional<String> label; // Mapped dimension
    }
}

// Tensor computation
Tensor queryEmbedding = Tensor.from("tensor<float>(x[384]):[0.1, 0.2, ...]");
Tensor distance = queryEmbedding.map((x, y) -> x - y)
                                .map(diff -> diff * diff)
                                .reduce(Reduce.Aggregator.sum, "x");
```

### 2. HNSW Implementation in Proton

**Native C++ Implementation**:
```cpp
namespace search::tensor {

class HnswIndex : public NearestNeighborIndex {
private:
    struct Node {
        uint32_t docid;
        uint32_t level;
        std::vector<std::vector<uint32_t>> neighbors; // Per level
    };
    
    struct Config {
        uint32_t max_links_per_node = 16;
        uint32_t neighbors_to_explore_at_insert = 200;
        uint32_t neighbors_to_explore_at_search = 100;
        double level_multiplier = 1.0 / log(2.0);
        DistanceFunction distance_func;
    };
    
    std::vector<Node> _graph;
    std::atomic<uint32_t> _entry_point;
    Config _config;
    
public:
    // Real-time insertion
    void add_document(uint32_t docid, ConstArrayRef<float> vector) override {
        auto level = select_level();
        Node node{docid, level};
        
        // Find neighbors at all levels
        auto candidates = search_layer(vector, _config.neighbors_to_explore_at_insert, level);
        
        // Connect bidirectionally
        for (uint32_t lev = 0; lev <= level; ++lev) {
            auto neighbors = select_neighbors(candidates[lev], M(lev));
            node.neighbors[lev] = neighbors;
            
            // Update reverse links
            for (auto neighbor : neighbors) {
                add_reverse_link(neighbor, docid, lev);
                prune_connections(neighbor, lev);
            }
        }
        
        _graph[docid] = std::move(node);
        update_entry_point(docid, level);
    }
    
    // Approximate search
    std::vector<NearestNeighborResult> find_top_k(
        ConstArrayRef<float> query,
        uint32_t k,
        const BitVector* filter) override {
        
        auto entry_points = get_entry_points();
        auto visited = std::make_unique<BitVector>(_graph.size());
        
        // Multi-layer search
        for (int level = _graph[_entry_point].level; level >= 0; --level) {
            entry_points = search_layer_with_filter(
                query, entry_points, level, visited.get(), filter);
        }
        
        // Extract top K
        return extract_top_k(entry_points, k);
    }
    
private:
    // Efficient neighbor selection with pruning
    std::vector<uint32_t> select_neighbors(
        const std::vector<Candidate>& candidates,
        uint32_t m) {
        
        std::vector<uint32_t> selected;
        selected.reserve(m);
        
        // Heuristic to maintain connectivity
        for (const auto& candidate : candidates) {
            if (selected.size() >= m) break;
            
            bool should_add = true;
            for (auto existing : selected) {
                auto dist = distance(candidate.docid, existing);
                if (dist < candidate.distance) {
                    should_add = false;
                    break;
                }
            }
            
            if (should_add) {
                selected.push_back(candidate.docid);
            }
        }
        
        return selected;
    }
};

} // namespace search::tensor
```

### 3. Container Cluster Processing

**Query Processing Pipeline**:
```java
public class VespaSearcher extends Searcher {
    private final TensorTransformer tensorTransformer;
    private final ExecutorService executor;
    
    @Override
    public Result search(Query query, Execution execution) {
        // 1. Parse query
        var queryTensor = extractQueryTensor(query);
        
        // 2. Query rewriting
        if (query.properties().getBoolean("rewrite.enable")) {
            query = rewriteQuery(query, queryTensor);
        }
        
        // 3. Scatter to content nodes
        var futures = scatterQuery(query, execution);
        
        // 4. Gather and merge results
        var results = gatherResults(futures);
        
        // 5. Apply ranking
        return rankResults(results, query);
    }
    
    private List<Future<Result>> scatterQuery(Query query, Execution execution) {
        var nodes = selectContentNodes(query);
        
        return nodes.stream()
            .map(node -> executor.submit(() -> 
                searchNode(node, query, execution)))
            .collect(Collectors.toList());
    }
}
```

**Ranking Framework**:
```java
public class RankProfile {
    private final String name;
    private final RankExpression firstPhase;
    private final Optional<RankExpression> secondPhase;
    private final Map<String, Tensor> constants;
    
    // Two-phase ranking
    public double computeScore(MatchFeatures features) {
        // First phase - run on all matches
        double firstPhaseScore = firstPhase.evaluate(features);
        
        // Second phase - run on top K from first phase
        if (secondPhase.isPresent() && features.isTopK()) {
            return secondPhase.get().evaluate(features);
        }
        
        return firstPhaseScore;
    }
}

// ONNX model integration
public class OnnxModel extends RankingExpression {
    private final OnnxEvaluator evaluator;
    
    public Tensor evaluate(Context context) {
        var inputs = prepareInputs(context);
        return evaluator.evaluate(inputs);
    }
}
```

### 4. Content Node Architecture

**Proton Search Core**:
```cpp
class Proton {
    // Document database per document type
    class DocumentDB {
        std::unique_ptr<MemoryIndex> _memoryIndex;
        std::unique_ptr<DiskIndex> _diskIndex;
        std::unique_ptr<AttributeManager> _attributes;
        std::unique_ptr<DocumentStore> _docStore;
        std::unique_ptr<TransactionLog> _transLog;
        
    public:
        // Real-time indexing
        void putDocument(const Document& doc) {
            // 1. Write to transaction log
            _transLog->append(doc);
            
            // 2. Update memory index
            _memoryIndex->insertDocument(doc);
            
            // 3. Update attributes (including HNSW)
            _attributes->update(doc);
            
            // 4. Schedule background flush
            scheduleFlush();
        }
        
        // Hybrid search
        SearchResult search(const SearchRequest& request) {
            // Search memory index
            auto memResults = _memoryIndex->search(request);
            
            // Search disk index
            auto diskResults = _diskIndex->search(request);
            
            // Search attributes (HNSW for vectors)
            auto attrResults = _attributes->search(request);
            
            // Merge results
            return mergeResults(memResults, diskResults, attrResults);
        }
    };
};
```

**Memory Management**:
```cpp
class AttributeVector {
protected:
    // Efficient memory layout
    class ValueStore {
        std::vector<char> _data;
        std::vector<uint32_t> _offsets;
        
    public:
        void set(uint32_t docid, ConstArrayRef<T> value) {
            auto offset = _offsets[docid];
            auto size = value.size() * sizeof(T);
            
            // Grow if needed
            if (offset + size > _data.size()) {
                grow(offset + size);
            }
            
            memcpy(&_data[offset], value.data(), size);
        }
    };
    
    // Generation management for safe concurrent access
    GenerationHandler _genHandler;
    
public:
    void commit() {
        _genHandler.incGeneration();
        _genHandler.updateFirstUsedGeneration();
    }
};

// Tensor attribute with HNSW index
class TensorAttribute : public AttributeVector {
    std::unique_ptr<HnswIndex> _hnswIndex;
    TensorStore _tensorStore;
    
public:
    void setTensor(uint32_t docid, const Tensor& tensor) {
        _tensorStore.store(docid, tensor);
        
        if (_hnswIndex) {
            _hnswIndex->add_document(docid, tensor.cells());
        }
    }
};
```

### 5. Distribution and Redundancy

**Content Distribution**:
```java
public class IdealStateManager {
    private final int redundancy;
    private final int searchableCopies;
    
    // Bucket distribution algorithm
    public List<StorageNode> getIdealNodes(BucketId bucket) {
        List<StorageNode> idealNodes = new ArrayList<>();
        
        // Consistent hashing with virtual nodes
        int nodeCount = cluster.getNodeCount();
        for (int copy = 0; copy < redundancy; copy++) {
            int nodeIndex = hash(bucket, copy) % nodeCount;
            idealNodes.add(cluster.getNode(nodeIndex));
        }
        
        return idealNodes;
    }
    
    // Maintain availability during failures
    public void handleNodeFailure(StorageNode failedNode) {
        var bucketsOnNode = getBucketsOnNode(failedNode);
        
        for (BucketId bucket : bucketsOnNode) {
            var currentNodes = getCurrentNodes(bucket);
            var idealNodes = getIdealNodes(bucket);
            
            // Find replacement nodes
            var replacements = idealNodes.stream()
                .filter(node -> !currentNodes.contains(node))
                .limit(redundancy - currentNodes.size() + 1)
                .collect(Collectors.toList());
            
            // Initiate bucket migration
            for (StorageNode target : replacements) {
                migrateBucket(bucket, currentNodes.get(0), target);
            }
        }
    }
}
```

## Query Processing

### 1. YQL Query Language

```java
// Vespa Query Language for complex queries
public class YQLParser {
    // Parse YQL with vector search
    public Query parse(String yql) {
        // Example: select * from products where title contains "laptop" 
        //          and ({targetHits: 100}nearestNeighbor(embedding, query_embedding))
        
        var ast = parseYQL(yql);
        var query = new Query();
        
        // Handle nearest neighbor operator
        if (ast.hasNearestNeighbor()) {
            var nnNode = ast.getNearestNeighbor();
            query.getRanking().getFeatures().put(
                "query(query_embedding)", 
                nnNode.getQueryVector()
            );
            query.getRanking().setProfile("similarity");
            query.setHits(nnNode.getTargetHits());
        }
        
        // Handle filters
        if (ast.hasWhere()) {
            query.setFilter(buildFilter(ast.getWhere()));
        }
        
        return query;
    }
}
```

### 2. Streaming Search

```cpp
// Low-latency streaming search for logged data
class StreamingSearcher {
    // Search while data streams through
    class StreamingVisitor : public IDocumentVisitor {
        const SearchRequest& _request;
        SearchResult& _result;
        
    public:
        void visit(const Document& doc) override {
            // Evaluate against streaming document
            if (matches(_request.query, doc)) {
                double score = computeScore(_request.ranking, doc);
                _result.addHit(doc.getId(), score);
            }
        }
    };
    
    SearchResult search(const SearchRequest& request) {
        SearchResult result;
        StreamingVisitor visitor(request, result);
        
        // Visit documents in parallel
        _docStore.accept(visitor, request.selection);
        
        return result;
    }
};
```

### 3. Grouping and Aggregation

```java
public class GroupingExecutor {
    // Complex aggregations with vector data
    public GroupingResult execute(GroupingRequest request) {
        // Example: Group by category and find nearest in each group
        var grouping = new Grouping()
            .setRoot(new GroupingNode()
                .addChild(new EachOperation()
                    .setGroupBy(new AttributeValue("category"))
                    .addChild(new EachOperation()
                        .setMax(10)
                        .addOutput(new MinAggregator(
                            new DistanceExpression("embedding", queryVector))))));
        
        return executeGrouping(grouping, request);
    }
}
```

## Performance Optimizations

### 1. SIMD and Vectorization

```cpp
// CPU-optimized distance calculations
namespace vespalib::hwaccelrated {

class GenericAccelrator : public IAccelrated {
public:
    float dotProduct(const float* a, const float* b, size_t sz) override {
        float sum = 0;
        size_t i = 0;
        
        // AVX2 implementation
        #ifdef __AVX2__
        const size_t width = 8;
        for (; i + width <= sz; i += width) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 prod = _mm256_mul_ps(va, vb);
            sum += horizontal_add(prod);
        }
        #endif
        
        // Scalar fallback
        for (; i < sz; ++i) {
            sum += a[i] * b[i];
        }
        
        return sum;
    }
};

} // namespace vespalib::hwaccelrated
```

### 2. Memory-Mapped Files

```cpp
class MMapFileStore {
    struct MMapFile {
        void* addr;
        size_t size;
        int fd;
    };
    
    std::unordered_map<FileId, MMapFile> _mmapFiles;
    
public:
    ConstArrayRef<char> read(FileId fileId, size_t offset, size_t len) {
        auto& file = _mmapFiles[fileId];
        
        // Direct memory access without copying
        return ConstArrayRef<char>(
            static_cast<char*>(file.addr) + offset, len);
    }
    
    void write(FileId fileId, size_t offset, ConstArrayRef<char> data) {
        auto& file = _mmapFiles[fileId];
        
        // Write with msync for durability
        memcpy(static_cast<char*>(file.addr) + offset, data.data(), data.size());
        msync(file.addr, file.size, MS_ASYNC);
    }
};
```

### 3. Adaptive Query Planning

```java
public class CostBasedOptimizer {
    // Choose between HNSW and brute force based on selectivity
    public QueryPlan optimize(Query query) {
        var stats = getStatistics(query);
        
        // Estimate result set size after filtering
        double selectivity = estimateSelectivity(query.getFilter());
        long estimatedDocs = (long)(stats.getDocumentCount() * selectivity);
        
        // Cost model for HNSW vs brute force
        double hnswCost = computeHnswCost(query, stats);
        double bruteForceCost = estimatedDocs * stats.getVectorDimension() * 0.001;
        
        if (hnswCost < bruteForceCost) {
            return new HnswQueryPlan(query);
        } else {
            return new BruteForceQueryPlan(query);
        }
    }
}
```

## Serving Infrastructure

### 1. Load Balancing

```java
public class SearchDispatcher {
    private final LoadBalancer loadBalancer;
    private final CircuitBreaker circuitBreaker;
    
    public Result dispatch(Query query) {
        // Select content nodes based on load
        var nodes = loadBalancer.selectNodes(query, getHealthyNodes());
        
        // Circuit breaker pattern
        return circuitBreaker.executeWithFallback(
            () -> dispatchToNodes(query, nodes),
            () -> fallbackSearch(query)
        );
    }
    
    class AdaptiveLoadBalancer implements LoadBalancer {
        public List<Node> selectNodes(Query query, List<Node> candidates) {
            // Sort by combined score of latency and queue depth
            return candidates.stream()
                .sorted((a, b) -> Double.compare(
                    scoreNode(a, query),
                    scoreNode(b, query)))
                .limit(query.getOffset() + query.getHits())
                .collect(Collectors.toList());
        }
        
        private double scoreNode(Node node, Query query) {
            return node.getAverageLatency() * 0.7 + 
                   node.getQueueDepth() * 0.3;
        }
    }
}
```

### 2. Feature Store Integration

```java
// Real-time feature computation for ranking
public class FeatureStore {
    private final Map<String, FeatureComputer> computers;
    
    public RankFeatures computeFeatures(Document doc, Query query) {
        var features = new RankFeatures();
        
        // Parallel feature computation
        computers.entrySet().parallelStream()
            .forEach(entry -> {
                var name = entry.getKey();
                var computer = entry.getValue();
                features.put(name, computer.compute(doc, query));
            });
        
        return features;
    }
}

// Example: Freshness-weighted similarity
public class FreshnessWeightedSimilarity implements FeatureComputer {
    public double compute(Document doc, Query query) {
        var similarity = cosineSimilarity(
            doc.getTensor("embedding"),
            query.getTensor("query_embedding")
        );
        
        var age = System.currentTimeMillis() - doc.getTimestamp();
        var freshness = Math.exp(-age / (24 * 3600 * 1000)); // Daily decay
        
        return similarity * freshness;
    }
}
```

## Configuration and Deployment

### 1. Application Package

```xml
<!-- services.xml -->
<services version="1.0">
  <container id="query" version="1.0">
    <search/>
    <nodes>
      <node hostalias="node1"/>
      <node hostalias="node2"/>
    </nodes>
  </container>
  
  <content id="products" version="1.0">
    <redundancy>2</redundancy>
    <documents>
      <document type="product" mode="index"/>
    </documents>
    <nodes>
      <node hostalias="node3" distribution-key="0"/>
      <node hostalias="node4" distribution-key="1"/>
      <node hostalias="node5" distribution-key="2"/>
    </nodes>
    <tuning>
      <searchnode>
        <summary>
          <io>
            <read>directio</read>
          </io>
        </summary>
      </searchnode>
    </tuning>
  </content>
</services>
```

### 2. Performance Tuning

```xml
<!-- Proton tuning for vector search -->
<tuning>
  <searchnode>
    <flushstrategy>
      <native>
        <total>
          <maxmemorygain>8g</maxmemorygain>
          <diskbloatfactor>0.2</diskbloatfactor>
        </total>
      </native>
    </flushstrategy>
    <index>
      <io>
        <write>directio</write>
        <read>mmap</read>
      </io>
    </index>
    <attribute>
      <io>
        <write>directio</write>
      </io>
    </attribute>
  </searchnode>
</tuning>
```

## Monitoring and Operations

### 1. Metrics Framework

```java
public class VespaMetrics {
    // Comprehensive metrics collection
    private final MetricReceiver metricReceiver;
    
    public void recordSearch(SearchStatistics stats) {
        var context = metricReceiver.createContext(Map.of(
            "documentType", stats.getDocumentType(),
            "rankProfile", stats.getRankProfile()
        ));
        
        metricReceiver.declareGauge("query.latency", stats.getLatency(), context);
        metricReceiver.declareCounter("query.count", 1, context);
        metricReceiver.declareGauge("hits.count", stats.getHitCount(), context);
        metricReceiver.declareGauge("totalhits.count", stats.getTotalHits(), context);
        
        // Vector search specific metrics
        if (stats.hasVectorSearch()) {
            metricReceiver.declareGauge("vector.candidates", 
                stats.getVectorCandidates(), context);
            metricReceiver.declareGauge("vector.distancecomputations", 
                stats.getDistanceComputations(), context);
        }
    }
}
```

### 2. Self-Healing

```java
public class SelfHealingManager {
    // Automatic recovery from failures
    public void monitorAndHeal() {
        // Detect and fix index corruption
        if (detectIndexCorruption()) {
            rebuildIndex();
        }
        
        // Rebalance data distribution
        if (isDataImbalanced()) {
            rebalanceData();
        }
        
        // Restart unhealthy services
        for (Service service : getUnhealthyServices()) {
            restartService(service);
        }
    }
}
```

## Summary

Vespa's architecture demonstrates:
1. **Integrated Platform**: Combines structured search, vector search, and ranking
2. **Real-time Serving**: Low-latency query processing with real-time updates
3. **Scalability**: Horizontal scaling with automatic data distribution
4. **Flexibility**: Tensor computations and complex ranking expressions
5. **Production Ready**: Self-healing, monitoring, and operational excellence

The architecture enables Vespa to serve as a complete search and recommendation platform with native vector search capabilities integrated into its core engine.