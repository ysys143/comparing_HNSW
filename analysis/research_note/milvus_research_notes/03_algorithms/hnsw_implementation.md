# Milvus HNSW Implementation Analysis

## Overview

Milvus implements HNSW through its Knowhere library, which provides a unified interface for multiple vector index types. The HNSW implementation is based on hnswlib but enhanced with distributed capabilities, GPU support, and integration with Milvus's segmented storage architecture.

## Core Implementation

### 1. **Knowhere HNSW Wrapper**

```cpp
// knowhere/index/vector_index/IndexHNSW.h
namespace knowhere {

class IndexHNSW : public VecIndex {
public:
    IndexHNSW() {
        index_type_ = IndexEnum::INDEX_HNSW;
        stats_ = std::make_shared<LibCallStats>();
    }

    BinarySet
    Serialize(const Config& config = Config()) override {
        if (!index_) {
            KNOWHERE_THROW_MSG("index not initialize");
        }

        try {
            hnswlib::HierarchicalNSW<float>* hnsw_index = 
                static_cast<hnswlib::HierarchicalNSW<float>*>(index_.get());
            
            size_t index_size = hnsw_index->cal_size();
            std::shared_ptr<uint8_t[]> index_binary(new uint8_t[index_size]);
            hnsw_index->SaveIndexToMemory(index_binary.get());

            BinarySet res_set;
            res_set.Append("HNSW", index_binary, index_size);
            return res_set;
        } catch (std::exception& e) {
            KNOWHERE_THROW_MSG(e.what());
        }
    }

    void
    Load(const BinarySet& binary_set) override {
        try {
            auto binary = binary_set.GetByName("HNSW");
            
            hnswlib::SpaceInterface<float>* space = nullptr;
            if (metric_type_ == metric::L2) {
                space = new hnswlib::L2Space(dim_);
            } else if (metric_type_ == metric::IP) {
                space = new hnswlib::InnerProductSpace(dim_);
            }

            index_ = std::make_shared<hnswlib::HierarchicalNSW<float>>(
                space, 
                binary->data.get(),
                binary->size
            );
        } catch (std::exception& e) {
            KNOWHERE_THROW_MSG(e.what());
        }
    }
};

} // namespace knowhere
```

### 2. **Build Implementation**

```cpp
// knowhere/index/vector_index/IndexHNSW.cpp
void
IndexHNSW::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    GET_TENSOR_DATA_DIM(dataset_ptr)

    metric_type_ = config[Metric::TYPE];
    if (metric_type_ == metric::L2) {
        index_ = std::make_shared<hnswlib::HierarchicalNSW<float>>(
            new hnswlib::L2Space(dim),
            rows,
            config[IndexParams::M].get<int64_t>(),
            config[IndexParams::efConstruction].get<int64_t>(),
            config[Seed].get<int64_t>()
        );
    } else if (metric_type_ == metric::IP) {
        index_ = std::make_shared<hnswlib::HierarchicalNSW<float>>(
            new hnswlib::InnerProductSpace(dim),
            rows,
            config[IndexParams::M].get<int64_t>(),
            config[IndexParams::efConstruction].get<int64_t>(),
            config[Seed].get<int64_t>()
        );
    } else {
        KNOWHERE_THROW_MSG("Unsupported metric type");
    }

    // Set number of threads for parallel construction
    if (config.contains(meta::ROWS)) {
        index_->setNumThreads(config[meta::ROWS].get<int64_t>());
    }
}

void
IndexHNSW::AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    GET_TENSOR_DATA(dataset_ptr)
    
    auto hnsw_index = static_cast<hnswlib::HierarchicalNSW<float>*>(index_.get());
    
    // Parallel insertion with OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < rows; ++i) {
        hnsw_index->addPoint(
            (const void*)(p_data + i * dim),
            i,
            config[Replace].get<bool>()
        );
    }
    
    // Update max elements if needed
    if (hnsw_index->cur_element_count > hnsw_index->max_elements_) {
        hnsw_index->resizeIndex(hnsw_index->cur_element_count);
    }
}
```

### 3. **Search Implementation**

```cpp
DatasetPtr
IndexHNSW::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
    GET_TENSOR_DATA_DIM(dataset_ptr)
    
    auto k = config[meta::TOPK].get<int64_t>();
    auto ef = config[IndexParams::ef].get<int64_t>();
    
    auto hnsw_index = static_cast<hnswlib::HierarchicalNSW<float>*>(index_.get());
    hnsw_index->setEf(ef);
    
    // Result storage
    auto p_id = new int64_t[rows * k];
    auto p_dist = new float[rows * k];
    
    // Parallel search
    #pragma omp parallel for
    for (int64_t i = 0; i < rows; ++i) {
        // Custom filter function for bitset
        auto filter = [&bitset](idx_t idx) -> bool {
            return !bitset || !bitset.test(idx);
        };
        
        // Search with filter
        auto result = hnsw_index->searchKnnCloserFirst(
            (const void*)(p_data + i * dim),
            k,
            filter
        );
        
        // Copy results
        size_t result_size = result.size();
        for (size_t j = 0; j < k; ++j) {
            if (j < result_size) {
                p_id[i * k + j] = result[j].second;
                p_dist[i * k + j] = result[j].first;
            } else {
                p_id[i * k + j] = -1;
                p_dist[i * k + j] = std::numeric_limits<float>::max();
            }
        }
    }
    
    return GenResultDataset(rows, k, p_id, p_dist);
}
```

### 4. **Milvus Integration Layer**

```cpp
// internal/core/src/index/HnswIndexNode.cpp
class HnswIndexNode : public IndexNode {
private:
    std::unique_ptr<knowhere::IndexHNSW> index_;
    
public:
    Status
    Build(const Config& config) override {
        try {
            auto dataset = PrepareDataset();
            
            // Configure HNSW parameters
            knowhere::Config knowhere_config;
            knowhere_config[knowhere::meta::DIM] = dim_;
            knowhere_config[knowhere::meta::ROWS] = row_count_;
            knowhere_config[knowhere::IndexParams::M] = config.get("M", 16);
            knowhere_config[knowhere::IndexParams::efConstruction] = 
                config.get("efConstruction", 200);
            knowhere_config[knowhere::Metric::TYPE] = 
                GetKnowhereMetricType(metric_type_);
            
            // Build index
            index_ = std::make_unique<knowhere::IndexHNSW>();
            index_->Train(dataset, knowhere_config);
            index_->AddWithoutIds(dataset, knowhere_config);
            
            return Status::OK();
        } catch (std::exception& e) {
            return Status(SERVER_UNEXPECTED_ERROR, e.what());
        }
    }
    
    Status
    Search(const SearchInfo& search_info,
           const float* query_data,
           int64_t query_rows,
           SearchResult& search_result) override {
        try {
            // Prepare query dataset
            auto query_dataset = knowhere::GenDataset(
                query_rows, dim_, query_data
            );
            
            // Configure search parameters
            knowhere::Config search_config;
            search_config[knowhere::meta::TOPK] = search_info.topk_;
            search_config[knowhere::IndexParams::ef] = 
                search_info.search_params_.get("ef", 64);
            
            // Execute search
            auto result = index_->Query(
                query_dataset,
                search_config,
                search_info.bitset_
            );
            
            // Process results
            ProcessSearchResult(result, search_result);
            
            return Status::OK();
        } catch (std::exception& e) {
            return Status(SERVER_UNEXPECTED_ERROR, e.what());
        }
    }
};
```

### 5. **Distributed HNSW Implementation**

```cpp
// internal/distributed/indexnode/hnsw_builder.cpp
class DistributedHnswBuilder {
private:
    struct Segment {
        int64_t segment_id;
        std::unique_ptr<HnswIndexNode> index;
        std::shared_ptr<storage::ChunkManager> chunk_manager;
    };
    
    std::vector<Segment> segments_;
    
public:
    Status
    BuildSegmentIndex(int64_t segment_id, const FieldData& field_data) {
        auto segment_index = std::make_unique<HnswIndexNode>();
        
        // Load vector data from segment
        auto vectors = LoadVectorsFromSegment(segment_id, field_data);
        
        // Build HNSW for this segment
        Config build_config;
        build_config["M"] = GetDynamicM(vectors.size());
        build_config["efConstruction"] = GetDynamicEf(vectors.size());
        
        segment_index->Build(build_config);
        
        // Store in distributed storage
        auto serialized = segment_index->Serialize();
        StoreIndexToObjectStorage(segment_id, serialized);
        
        segments_.push_back({segment_id, std::move(segment_index), nullptr});
        
        return Status::OK();
    }
    
    // Dynamic parameter tuning based on data size
    int GetDynamicM(size_t num_vectors) {
        if (num_vectors < 10000) return 16;
        if (num_vectors < 100000) return 32;
        if (num_vectors < 1000000) return 48;
        return 64;
    }
    
    int GetDynamicEf(size_t num_vectors) {
        return std::min(static_cast<int>(num_vectors), 
                       std::max(200, GetDynamicM(num_vectors) * 2));
    }
};
```

### 6. **GPU-Accelerated HNSW**

```cpp
// knowhere/index/gpu/IndexGPUHNSW.cpp
namespace knowhere {

class IndexGPUHNSW : public GPUIndex {
private:
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> cpu_index_;
    std::unique_ptr<faiss::IndexHNSWFlat> gpu_index_;
    
public:
    void
    Train(const DatasetPtr& dataset, const Config& config) override {
        // First build on CPU
        cpu_index_ = BuildCPUIndex(dataset, config);
        
        // Convert to GPU format
        ConvertToGPU(config);
    }
    
    void
    ConvertToGPU(const Config& config) {
        int gpu_id = config[meta::DEVICEID].get<int64_t>();
        
        // Create GPU resources
        auto gpu_resource = FaissGpuResourceMgr::GetInstance().
            GetRes(gpu_id);
            
        // Configure GPU index
        faiss::gpu::GpuIndexHNSWFlatConfig gpu_config;
        gpu_config.device = gpu_id;
        gpu_config.hnsw_M = cpu_index_->M_;
        
        // Build GPU index from CPU index
        gpu_index_ = std::make_unique<faiss::gpu::GpuIndexHNSWFlat>(
            gpu_resource->faiss_res.get(),
            cpu_index_->d_,
            cpu_index_->M_,
            metric_type_ == metric::L2 ? 
                faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT,
            gpu_config
        );
        
        // Copy graph structure to GPU
        CopyGraphToGPU();
    }
    
    DatasetPtr
    QueryGPU(const DatasetPtr& dataset, const Config& config) {
        GET_TENSOR_DATA_DIM(dataset)
        
        auto k = config[meta::TOPK].get<int64_t>();
        
        // Allocate GPU memory for results
        float* d_distances;
        int64_t* d_labels;
        cudaMalloc(&d_distances, rows * k * sizeof(float));
        cudaMalloc(&d_labels, rows * k * sizeof(int64_t));
        
        // Execute GPU search
        gpu_index_->search(rows, p_data, k, d_distances, d_labels);
        
        // Copy results back to CPU
        auto h_distances = new float[rows * k];
        auto h_labels = new int64_t[rows * k];
        cudaMemcpy(h_distances, d_distances, 
                   rows * k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_labels, d_labels, 
                   rows * k * sizeof(int64_t), cudaMemcpyDeviceToHost);
        
        cudaFree(d_distances);
        cudaFree(d_labels);
        
        return GenResultDataset(rows, k, h_labels, h_distances);
    }
};

} // namespace knowhere
```

### 7. **Memory Management and Optimization**

```cpp
// knowhere/index/vector_index/helpers/DynamicResultSet.cpp
template<typename DistType>
class DynamicResultSet {
private:
    struct ResultItem {
        DistType distance;
        idx_t id;
        
        bool operator<(const ResultItem& other) const {
            return distance < other.distance;
        }
    };
    
    std::priority_queue<ResultItem> heap_;
    size_t capacity_;
    
public:
    void
    AddResult(DistType dist, idx_t id) {
        if (heap_.size() < capacity_) {
            heap_.push({dist, id});
        } else if (dist < heap_.top().distance) {
            heap_.pop();
            heap_.push({dist, id});
        }
    }
    
    std::vector<std::pair<DistType, idx_t>>
    GetSortedResults() {
        std::vector<std::pair<DistType, idx_t>> results;
        results.reserve(heap_.size());
        
        while (!heap_.empty()) {
            auto item = heap_.top();
            heap_.pop();
            results.push_back({item.distance, item.id});
        }
        
        std::reverse(results.begin(), results.end());
        return results;
    }
};

// Memory pool for graph construction
class GraphMemoryPool {
private:
    struct Block {
        std::unique_ptr<uint8_t[]> data;
        size_t size;
        size_t used;
    };
    
    std::vector<Block> blocks_;
    size_t block_size_;
    
public:
    void* Allocate(size_t size) {
        // Find block with enough space
        for (auto& block : blocks_) {
            if (block.size - block.used >= size) {
                void* ptr = block.data.get() + block.used;
                block.used += size;
                return ptr;
            }
        }
        
        // Allocate new block
        size_t new_block_size = std::max(block_size_, size);
        blocks_.push_back({
            std::make_unique<uint8_t[]>(new_block_size),
            new_block_size,
            size
        });
        
        return blocks_.back().data.get();
    }
    
    void Reset() {
        for (auto& block : blocks_) {
            block.used = 0;
        }
    }
};
```

## Configuration and Tuning

### Index Parameters
```yaml
index_type: HNSW
index_params:
  M: 16               # Number of connections
  efConstruction: 200 # Construction search width
  
search_params:
  ef: 64              # Search width (must be >= k)
```

### Segment-Level Configuration
```cpp
// Automatic parameter adjustment based on segment size
struct HnswAutoTuner {
    static Config GetOptimalParams(size_t num_vectors, size_t dim) {
        Config config;
        
        // M parameter tuning
        if (num_vectors < 1000) {
            config["M"] = 8;
        } else if (num_vectors < 10000) {
            config["M"] = 16;
        } else if (num_vectors < 100000) {
            config["M"] = 32;
        } else {
            config["M"] = 48;
        }
        
        // efConstruction tuning
        config["efConstruction"] = std::min(
            static_cast<int>(num_vectors),
            config["M"].get<int>() * 64
        );
        
        // Memory prediction
        size_t memory_per_node = sizeof(idx_t) * config["M"].get<int>() * 2;
        size_t total_memory = num_vectors * (memory_per_node + dim * sizeof(float));
        
        LOG_INFO("HNSW index will use approximately {} MB", 
                 total_memory / (1024 * 1024));
        
        return config;
    }
};
```

## Performance Characteristics

### Advantages
- GPU acceleration support
- Distributed index building
- Dynamic parameter tuning
- Integration with Milvus's segment architecture

### Trade-offs
- Memory overhead from Knowhere wrapper
- Serialization costs for distributed storage
- Limited customization of hnswlib core

## Best Practices

### 1. **Segment Management**
```cpp
// Optimal segment size for HNSW
const size_t OPTIMAL_SEGMENT_SIZE = 1000000;  // 1M vectors

// Merge small segments for better performance
void MergeSmallSegments(std::vector<Segment>& segments) {
    std::vector<Segment> merged;
    std::vector<size_t> to_merge;
    
    for (size_t i = 0; i < segments.size(); ++i) {
        if (segments[i].row_count < OPTIMAL_SEGMENT_SIZE / 10) {
            to_merge.push_back(i);
        }
    }
    
    if (to_merge.size() > 1) {
        MergeSegments(segments, to_merge, merged);
    }
}
```

### 2. **Search Optimization**
```python
# Client-side search optimization
def optimized_search(collection, query_vectors, k=10):
    # Dynamically adjust ef based on k
    ef = max(64, k * 2)
    
    # Use consistency level based on requirements
    consistency = "Eventually" if k > 100 else "Strong"
    
    results = collection.search(
        data=query_vectors,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"ef": ef}},
        limit=k,
        consistency_level=consistency
    )
    
    return results
```

### 3. **Index Building Strategy**
```python
# Batch index building for large datasets
def build_hnsw_index(collection, vectors, batch_size=100000):
    total = len(vectors)
    
    for i in range(0, total, batch_size):
        batch = vectors[i:i + batch_size]
        
        # Insert batch
        collection.insert(batch)
        
        # Flush periodically to trigger index building
        if (i + batch_size) % 1000000 == 0:
            collection.flush()
            
    # Final flush and index
    collection.flush()
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": 16, "efConstruction": 200}
        }
    )
```

## Code References

### Core Implementation
- `knowhere/index/vector_index/IndexHNSW.h` - Main HNSW wrapper
- `internal/core/src/index/HnswIndexNode.cpp` - Milvus integration
- `knowhere/index/gpu/IndexGPUHNSW.cpp` - GPU acceleration
- `internal/distributed/indexnode/` - Distributed building

## Comparison Notes
- Built on hnswlib with significant enhancements
- Strong distributed and GPU capabilities
- Integrated with Milvus's segmented architecture
- Trade-off: Flexibility for system integration