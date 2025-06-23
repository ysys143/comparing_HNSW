# Vespa HNSW Implementation Analysis

## Overview

Vespa's HNSW implementation is designed for large-scale production deployments with a focus on concurrent access, generation-based memory management, and deep integration with Vespa's document processing pipeline. It features template-based design for flexibility and MIPS distance optimization.

## Graph Construction

### Core Graph Structure

```cpp
template <HnswIndexType type>
class HnswIndex : public NearestNeighborIndex {
    using GraphType = HnswGraph<type>;
    GraphType _graph;
    LevelGenerator _level_generator;
    Config _cfg;
    
    // RCU-based concurrent access
    std::atomic<EntryNode> _entry_node;
    mutable std::atomic<const GlobalFilter *> _global_filter;
};

template <HnswIndexType type>
struct HnswGraph {
    using NodeType = typename HnswIndexTraits<type>::NodeType;
    
    NodeStore nodes;                    // RcuVector for concurrent access
    LevelArrayStore levels_store;       // Generation-based storage
    LinkArrayStore links_store;         // Compact link storage
    
    std::atomic<uint32_t> _entry_nodeid;
    std::atomic<int32_t> _entry_level;
};
```

### Two-Phase Node Insertion

Vespa uses a prepare-commit pattern for safe concurrent insertion:

```cpp
template <HnswIndexType type>
class HnswIndex {
    struct PreparedAddNode {
        uint32_t nodeid;
        std::vector<std::vector<uint32_t>> connections_per_level;
        vespalib::GenerationHandler::Guard guard;
    };
    
    PreparedAddNode prepare_add_document(uint32_t docid, 
                                       VectorBundle vectors) {
        auto guard = _graph.node_refs_size.getGuard();
        uint32_t nodeid = _graph.make_node(docid, vectors);
        int level = _level_generator.max_level();
        
        PreparedAddNode prepared(nodeid, level, guard);
        
        // Find neighbors at each level
        for (int cur_level = level; cur_level >= 0; --cur_level) {
            auto neighbors = find_nearest_in_layer(vectors, cur_level);
            prepared.connections_per_level[cur_level] = neighbors;
        }
        
        return prepared;
    }
    
    void complete_add_document(PreparedAddNode prepared) {
        // Add links atomically
        for (int level = 0; level <= prepared.level; ++level) {
            for (uint32_t neighbor : prepared.connections_per_level[level]) {
                _graph.add_link(prepared.nodeid, neighbor, level);
                _graph.add_link(neighbor, prepared.nodeid, level);
            }
        }
        
        // Update entry point if needed
        update_entry_node(prepared.nodeid, prepared.level);
    }
};
```

### Neighbor Selection with Heuristic

```cpp
template <typename DistanceFunction>
NearestNeighborIndex::SelectResult
select_neighbors(const HnswCandidateVector& neighbors, 
                uint32_t max_links,
                const BitVector* filter) {
    if (!_cfg.heuristic_select_neighbors()) {
        return select_neighbors_simple(neighbors, max_links);
    }
    
    NearestNeighborIndex::SelectResult result;
    result.used.reserve(max_links);
    
    // Heuristic selection to improve graph connectivity
    for (const auto& candidate : neighbors) {
        if (result.used.size() >= max_links) break;
        
        bool good = true;
        for (uint32_t selected : result.used) {
            double dist_to_selected = distance(candidate.nodeid, selected);
            if (dist_to_selected < candidate.distance) {
                good = false;
                break;
            }
        }
        
        if (good) {
            result.used.push_back(candidate.nodeid);
        } else {
            result.unused.push_back(candidate.nodeid);
        }
    }
    
    return result;
}
```

### MIPS Distance Optimization

```cpp
class MipsDistanceFunctionFactoryBase {
    double _max_squared_norm;
    
    template<typename FloatType>
    double calc_distance(const vespalib::eval::TypedCells& lhs,
                        const vespalib::eval::TypedCells& rhs) {
        auto lhs_vector = lhs.typify<FloatType>();
        auto rhs_vector = rhs.typify<FloatType>();
        
        // Transform MIPS to angular distance
        double dot_product = dot_product_bounded(lhs_vector, rhs_vector);
        double rhs_norm_sq = norm_squared(rhs_vector);
        
        // Adjust for maximum norm
        double transform = std::sqrt(_max_squared_norm - rhs_norm_sq);
        return -dot_product / transform;
    }
};
```

## Search Algorithm

### Search Implementation with RCU

```cpp
template <HnswIndexType type>
struct SearchBestNeighbors {
    const HnswGraph<type>& graph;
    const TypedCells& target;
    const GlobalFilter* filter;
    uint32_t explore_k;
    vespalib::doom::Steady doom;
    
    HnswCandidateVector search() {
        // Use appropriate visited tracker based on graph size
        if (graph.size() < 65536) {
            VisitedTracker<uint16_t> visited(graph.size());
            return search_impl(visited);
        } else {
            VisitedTracker<uint32_t> visited(graph.size());
            return search_impl(visited);
        }
    }
    
    template <typename VisitedTracker>
    HnswCandidateVector search_impl(VisitedTracker& visited) {
        uint32_t entry_nodeid = graph.get_entry_nodeid();
        int entry_level = graph.get_entry_level();
        
        // Search from top to bottom layer
        HnswCandidate entry_point(entry_nodeid, calc_distance(entry_nodeid));
        
        for (int level = entry_level; level > 0; --level) {
            entry_point = find_nearest_in_layer(entry_point, level, visited);
        }
        
        // Final search at layer 0
        return search_layer_with_filter(entry_point, 0, visited);
    }
};
```

### Global Filter Integration

```cpp
template <typename VisitedTracker>
void SearchBestNeighbors::search_layer_with_filter(
    HnswCandidateVector& candidates,
    uint32_t k,
    const LinkArrayRef& neighbors,
    VisitedTracker& visited) {
    
    for (uint32_t neighbor_nodeid : neighbors) {
        if (visited.try_visit(neighbor_nodeid)) {
            
            // Apply global filter
            if (filter && !filter->check(get_docid(neighbor_nodeid))) {
                continue;
            }
            
            double dist = calc_distance(neighbor_nodeid);
            
            if (dist < candidates.top().distance || 
                candidates.size() < k) {
                candidates.emplace(neighbor_nodeid, dist);
                
                if (candidates.size() > k) {
                    candidates.pop();
                }
            }
        }
    }
}
```

### Generation-Based Memory Management

```cpp
class HnswGraph {
    vespalib::GenerationHandler _gen_handler;
    
    void remove_node(uint32_t nodeid) {
        auto node_ref = _node_refs[nodeid].load(std::memory_order_acquire);
        _node_refs[nodeid].store(AtomicNodeRef(), std::memory_order_release);
        
        // Schedule for later cleanup
        _gen_handler.incGeneration();
        _gen_handler.scheduleDestroy(node_ref);
    }
    
    void commit() {
        _gen_handler.incGeneration();
        _gen_handler.updateFirstUsedGeneration();
    }
    
    void reclaim_memory() {
        _gen_handler.reclaim_memory();
        _nodes.reclaim_memory(_gen_handler.get_oldest_used_generation());
        _links.reclaim_memory(_gen_handler.get_oldest_used_generation());
    }
};
```

## Memory Optimization

### Compact Link Storage

```cpp
class LinkArrayStore {
    using RefType = vespalib::datastore::AtomicEntryRef;
    using LinksBuffer = vespalib::Array<uint32_t>;
    
    vespalib::datastore::ArrayStore<uint32_t> _store;
    
    LinkArrayRef get(RefType ref) const {
        if (!ref.valid()) {
            return LinkArrayRef();
        }
        auto array = _store.get(ref);
        return LinkArrayRef(array.data(), array.size());
    }
    
    RefType add(const std::vector<uint32_t>& links) {
        return _store.add(links);
    }
};
```

### Level Array Optimization

```cpp
class LevelArrayStore {
    using Store = vespalib::datastore::ArrayStore<int16_t>;
    Store _store;
    
    // Compact storage for node levels
    AtomicEntryRef add(vespalib::ConstArrayRef<int16_t> levels) {
        return _store.add(levels);
    }
};
```

### Template-Based Node Types

```cpp
template <>
struct HnswIndexTraits<HnswIndexType::SINGLE> {
    using NodeType = HnswSimpleNode;
};

template <>
struct HnswIndexTraits<HnswIndexType::MULTI> {
    using NodeType = HnswTestNode;  // Supports multiple vectors
};

// Compile-time optimization based on node type
template <typename NodeType>
double calc_distance(const NodeType& node, const TypedCells& target) {
    if constexpr (std::is_same_v<NodeType, HnswSimpleNode>) {
        // Optimized single-vector distance
        return distance_function->calc(node.vector(), target);
    } else {
        // Multi-vector distance calculation
        return calc_multi_distance(node.vectors(), target);
    }
}
```

## Concurrent Access

### RCU (Read-Copy-Update) Pattern

```cpp
class HnswIndex {
    void set_global_filter(const GlobalFilter& filter) {
        const GlobalFilter* old_filter = 
            _global_filter.exchange(&filter, std::memory_order_release);
        
        if (old_filter != nullptr) {
            // Schedule old filter for deletion after grace period
            _gen_handler.scheduleDestroy(old_filter);
        }
    }
    
    const GlobalFilter* get_global_filter() const {
        return _global_filter.load(std::memory_order_acquire);
    }
};
```

### Lock-Free Entry Point Updates

```cpp
void update_entry_node(uint32_t nodeid, int32_t level) {
    EntryNode new_entry(nodeid, level);
    EntryNode old_entry = _entry_node.load(std::memory_order_relaxed);
    
    while (level > old_entry.level && 
           !_entry_node.compare_exchange_weak(old_entry, new_entry)) {
        // Retry with updated old_entry
    }
}
```

## Configuration

```cpp
struct HnswIndexConfig {
    uint32_t max_links_at_level_0() const { return m * 2; }
    uint32_t max_links_on_inserts() const { return m; }
    uint32_t neighbors_to_explore_at_construction() const { 
        return std::max(ef_construction, m + 1); 
    }
    
    uint32_t m = 16;
    uint32_t ef_construction = 200;
    uint32_t max_squared_norm = 0;
    bool heuristic_select_neighbors = true;
};
```

## Unique Features

### 1. Generation-Based Memory Management
- Safe memory reclamation without locks
- Deferred destruction after grace period
- Efficient batch cleanup

### 2. Template-Based Design
- Compile-time optimization
- Support for different node types
- Zero-cost abstraction

### 3. MIPS Distance Transform
- Optimized for maximum inner product search
- Norm tracking and adjustment
- Mathematical transformation to angular distance

### 4. Two-Phase Commit
- Prepare phase for calculation
- Commit phase for atomic updates
- Safe concurrent modifications

### 5. Adaptive Visited Tracking
- 16-bit tracker for small graphs
- 32-bit tracker for large graphs
- Memory-efficient bit vectors

## Performance Optimizations

### 1. Memory Layout
```cpp
// Cache-friendly node storage
struct HnswSimpleNode {
    AtomicVectorRef _vector_ref;
    AtomicLevelRef _level_ref;
    AtomicLinkRef _link_ref;
    
    // Atomic operations for lock-free access
    VectorRef get_vector_ref() const {
        return _vector_ref.load(std::memory_order_acquire);
    }
};
```

### 2. SIMD Distance Calculations
```cpp
template<>
double calc_distance<float>(const float* a, const float* b, size_t sz) {
    return vespalib::hwaccelrated::IAccelrated::getAccelerator()
        .squaredEuclideanDistance(a, b, sz);
}
```

### 3. Batch Operations
```cpp
void add_document_batch(const std::vector<Document>& docs) {
    // Prepare all documents
    std::vector<PreparedAddNode> prepared;
    for (const auto& doc : docs) {
        prepared.push_back(prepare_add_document(doc));
    }
    
    // Commit in batch
    for (auto& p : prepared) {
        complete_add_document(std::move(p));
    }
    
    commit();  // Single generation increment
}
```

## Summary

Vespa's HNSW implementation excels in:
1. **Concurrent Access**: Lock-free operations with RCU
2. **Memory Efficiency**: Generation-based management and compact storage
3. **Type Safety**: Template-based design with compile-time optimization
4. **Production Readiness**: Integrated monitoring, doom-based cancellation
5. **Flexibility**: Support for multiple vector types and distance functions

The implementation is optimized for Vespa's large-scale document processing needs while maintaining excellent single-node performance.