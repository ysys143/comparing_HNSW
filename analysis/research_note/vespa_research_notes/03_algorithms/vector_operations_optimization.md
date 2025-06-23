# Vespa Vector Operations Optimization Analysis

## Overview
Vespa implements highly optimized vector operations through C++ with extensive SIMD usage, custom memory allocators, and sophisticated caching mechanisms. The implementation is designed for both real-time serving and large-scale batch operations.

## Core Optimizations

### 1. **SIMD Implementation**
```cpp
// searchlib/src/vespa/searchlib/tensor/distance_functions.h
namespace vespalib::hwaccelrated {

template<typename T>
class EuclideanDistanceOptimized {
public:
    static float calc(const T* a, const T* b, size_t sz) {
        return IAccelrated::getAccelerator().squaredEuclideanDistance(a, b, sz);
    }
};

// CPU-specific implementations
class IAccelrated {
public:
    virtual float squaredEuclideanDistance(const float* a, const float* b, size_t sz) const = 0;
    virtual float dotProduct(const float* a, const float* b, size_t sz) const = 0;
    
    static const IAccelrated& getAccelerator() {
        // Runtime CPU detection
        if (hasAVX512()) return avx512Accelerator;
        if (hasAVX2()) return avx2Accelerator;
        if (hasAVX()) return avxAccelerator;
        return genericAccelerator;
    }
};

// AVX-512 implementation
class Avx512Accel : public IAccelrated {
    float squaredEuclideanDistance(const float* a, const float* b, size_t sz) const override {
        size_t i = 0;
        __m512 sum = _mm512_setzero_ps();
        
        // Main loop - process 16 elements at a time
        for (; i + 16 <= sz; i += 16) {
            __m512 av = _mm512_loadu_ps(a + i);
            __m512 bv = _mm512_loadu_ps(b + i);
            __m512 diff = _mm512_sub_ps(av, bv);
            sum = _mm512_fmadd_ps(diff, diff, sum);
        }
        
        // Horizontal sum
        float result = _mm512_reduce_add_ps(sum);
        
        // Handle remainder
        for (; i < sz; ++i) {
            float diff = a[i] - b[i];
            result += diff * diff;
        }
        
        return result;
    }
};
```

### 2. **Memory Management**

#### Custom Allocators
```cpp
// vespalib/src/vespa/vespalib/util/alloc.h
class AlignedAlloc {
    static constexpr size_t HUGEPAGE_SIZE = 2 * 1024 * 1024;
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
public:
    static void* allocate(size_t sz, size_t alignment = CACHE_LINE_SIZE) {
        if (sz >= HUGEPAGE_SIZE) {
            // Use huge pages for large allocations
            return allocateHugePage(sz);
        }
        return std::aligned_alloc(alignment, sz);
    }
    
private:
    static void* allocateHugePage(size_t sz) {
        void* ptr = mmap(nullptr, sz,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                        -1, 0);
        if (ptr == MAP_FAILED) {
            // Fallback to regular allocation
            return std::aligned_alloc(CACHE_LINE_SIZE, sz);
        }
        return ptr;
    }
};
```

#### Memory Pool for Tensors
```cpp
// searchlib/src/vespa/searchlib/tensor/tensor_buffer_store.h
class TensorBufferStore {
    struct BufferPool {
        std::vector<std::unique_ptr<char[]>> buffers;
        std::stack<char*> available;
        std::mutex mutex;
        size_t buffer_size;
        
        char* allocate() {
            std::lock_guard<std::mutex> guard(mutex);
            if (available.empty()) {
                auto buffer = std::make_unique<char[]>(buffer_size);
                char* ptr = buffer.get();
                buffers.push_back(std::move(buffer));
                return ptr;
            }
            char* ptr = available.top();
            available.pop();
            return ptr;
        }
        
        void deallocate(char* ptr) {
            std::lock_guard<std::mutex> guard(mutex);
            available.push(ptr);
        }
    };
    
    std::array<BufferPool, 16> pools;  // Different sizes
    
public:
    void* allocate(size_t size) {
        size_t pool_idx = getPoolIndex(size);
        return pools[pool_idx].allocate();
    }
};
```

### 3. **Tensor Operations**

#### Optimized Dense Tensor Operations
```cpp
// eval/src/vespa/eval/tensor/dense_tensor_operations.h
namespace vespalib::tensor {

class DenseTensorOperations {
public:
    // Blocked matrix multiplication for cache efficiency
    static void matmul_blocked(
        const float* a, const float* b, float* c,
        size_t m, size_t n, size_t k,
        size_t block_size = 64) {
        
        #pragma omp parallel for
        for (size_t i0 = 0; i0 < m; i0 += block_size) {
            for (size_t k0 = 0; k0 < k; k0 += block_size) {
                for (size_t j0 = 0; j0 < n; j0 += block_size) {
                    // Process block
                    size_t i_max = std::min(i0 + block_size, m);
                    size_t k_max = std::min(k0 + block_size, k);
                    size_t j_max = std::min(j0 + block_size, n);
                    
                    for (size_t i = i0; i < i_max; ++i) {
                        for (size_t kk = k0; kk < k_max; ++kk) {
                            float a_ik = a[i * k + kk];
                            
                            // Vectorized inner loop
                            #pragma omp simd
                            for (size_t j = j0; j < j_max; ++j) {
                                c[i * n + j] += a_ik * b[kk * n + j];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Fused operations to reduce memory bandwidth
    static void add_multiply_add(
        const float* a, const float* b, const float* c,
        float alpha, float beta, float* result, size_t size) {
        
        size_t i = 0;
        
        // AVX-512 implementation
        #ifdef __AVX512F__
        __m512 alpha_vec = _mm512_set1_ps(alpha);
        __m512 beta_vec = _mm512_set1_ps(beta);
        
        for (; i + 16 <= size; i += 16) {
            __m512 av = _mm512_loadu_ps(a + i);
            __m512 bv = _mm512_loadu_ps(b + i);
            __m512 cv = _mm512_loadu_ps(c + i);
            
            // result = alpha * (a + b) + beta * c
            __m512 sum = _mm512_add_ps(av, bv);
            __m512 scaled_sum = _mm512_mul_ps(alpha_vec, sum);
            __m512 scaled_c = _mm512_mul_ps(beta_vec, cv);
            __m512 res = _mm512_add_ps(scaled_sum, scaled_c);
            
            _mm512_storeu_ps(result + i, res);
        }
        #endif
        
        // Scalar remainder
        for (; i < size; ++i) {
            result[i] = alpha * (a[i] + b[i]) + beta * c[i];
        }
    }
};

}  // namespace vespalib::tensor
```

### 4. **HNSW-Specific Optimizations**

#### Distance Calculation Cache
```cpp
// searchlib/src/vespa/searchlib/tensor/hnsw_index.h
class HnswIndex {
    // Cache recently computed distances
    struct DistanceCache {
        struct Entry {
            uint32_t query_id;
            uint32_t doc_id;
            float distance;
        };
        
        static constexpr size_t CACHE_SIZE = 1 << 20;  // 1M entries
        std::vector<Entry> cache;
        std::atomic<uint32_t> current_query_id{0};
        
        float get_distance(uint32_t query_id, uint32_t doc_id, 
                          std::function<float()> compute) {
            size_t hash = hash_combine(query_id, doc_id) & (CACHE_SIZE - 1);
            Entry& entry = cache[hash];
            
            if (entry.query_id == query_id && entry.doc_id == doc_id) {
                return entry.distance;  // Cache hit
            }
            
            // Cache miss - compute and store
            float distance = compute();
            entry = {query_id, doc_id, distance};
            return distance;
        }
    };
    
    mutable DistanceCache distance_cache;
};
```

#### Batch Node Processing
```cpp
// Process multiple nodes in parallel
void search_layer_batch(
    const TypedCells& query,
    const HnswCandidate* candidates,
    size_t num_candidates,
    uint32_t level,
    uint32_t ef,
    BestNeighbors& best) {
    
    // Prefetch node data
    for (size_t i = 0; i < std::min(size_t(8), num_candidates); ++i) {
        const auto& node = get_node(candidates[i].docid);
        __builtin_prefetch(&node, 0, 3);
        __builtin_prefetch(get_vector_ptr(candidates[i].docid), 0, 3);
    }
    
    // Process candidates in batches for better cache utilization
    constexpr size_t BATCH_SIZE = 4;
    for (size_t i = 0; i < num_candidates; i += BATCH_SIZE) {
        size_t batch_end = std::min(i + BATCH_SIZE, num_candidates);
        
        // Compute distances for batch
        float distances[BATCH_SIZE];
        for (size_t j = i; j < batch_end; ++j) {
            distances[j - i] = calc_distance(query, candidates[j].docid);
        }
        
        // Update best neighbors
        for (size_t j = i; j < batch_end; ++j) {
            best.try_insert(candidates[j].docid, distances[j - i]);
        }
    }
}
```

### 5. **Parallel Query Execution**

#### Thread-Local Storage
```cpp
class HnswSearcher {
    // Thread-local scratch space to avoid allocations
    struct ThreadLocalData {
        std::vector<HnswCandidate> candidates;
        vespalib::hash_set<uint32_t> visited;
        std::vector<float> distances;
        
        void reset() {
            candidates.clear();
            visited.clear();
            // Keep capacity to avoid reallocation
        }
    };
    
    static thread_local ThreadLocalData tld;
    
public:
    SearchResult search(const TypedCells& query, uint32_t k, uint32_t ef) {
        tld.reset();
        
        // Use pre-allocated structures
        return search_internal(query, k, ef, tld);
    }
};
```

### 6. **Quantization Support**

#### Binary Quantization with Hamming Distance
```cpp
class BinaryQuantizedVectors {
    // Pack float vectors into bits
    static void quantize_avx2(const float* input, uint64_t* output, size_t dim) {
        size_t num_words = (dim + 63) / 64;
        
        for (size_t w = 0; w < num_words; ++w) {
            uint64_t word = 0;
            size_t base = w * 64;
            
            // Process 8 floats at a time with AVX2
            for (size_t i = 0; i < 64 && base + i < dim; i += 8) {
                __m256 values = _mm256_loadu_ps(input + base + i);
                __m256 zero = _mm256_setzero_ps();
                __m256 cmp = _mm256_cmp_ps(values, zero, _CMP_GT_OQ);
                
                // Extract comparison results to bits
                int mask = _mm256_movemask_ps(cmp);
                word |= uint64_t(mask) << i;
            }
            
            output[w] = word;
        }
    }
    
    // Optimized Hamming distance
    static uint32_t hamming_distance(const uint64_t* a, const uint64_t* b, size_t num_words) {
        uint32_t distance = 0;
        
        #pragma omp simd reduction(+:distance)
        for (size_t i = 0; i < num_words; ++i) {
            distance += __builtin_popcountll(a[i] ^ b[i]);
        }
        
        return distance;
    }
};
```

### 7. **Attribute Vector Optimization**

#### Streaming Updates
```cpp
// searchlib/src/vespa/searchlib/attribute/dense_tensor_attribute.h
class DenseTensorAttribute {
    // Lock-free read path with versioning
    struct TensorEntry {
        std::atomic<uint64_t> version;
        std::unique_ptr<vespalib::eval::Value> tensor;
        
        vespalib::eval::Value* read() const {
            uint64_t v;
            vespalib::eval::Value* t;
            do {
                v = version.load(std::memory_order_acquire);
                if (v & 1) continue;  // Write in progress
                t = tensor.get();
            } while (v != version.load(std::memory_order_acquire));
            return t;
        }
        
        void write(std::unique_ptr<vespalib::eval::Value> new_tensor) {
            uint64_t v = version.load();
            version.store(v + 1, std::memory_order_release);
            tensor = std::move(new_tensor);
            version.store(v + 2, std::memory_order_release);
        }
    };
};
```

## Performance Characteristics

### Advantages
- Comprehensive SIMD coverage (SSE, AVX, AVX-512)
- Sophisticated memory management with huge pages
- Excellent cache utilization strategies
- Production-tested at massive scale

### Limitations
- Complex codebase with steep learning curve
- Higher memory overhead for caching structures
- C++ complexity for extensions

### Configuration
```xml
<!-- services.xml -->
<content>
    <search>
        <query-profiles>
            <query-profile id="default">
                <field name="ranking.matching.numThreadsPerSearch">4</field>
                <field name="ranking.matching.numSearchPartitions">4</field>
            </query-profile>
        </query-profiles>
    </search>
    
    <tuning>
        <searchnode>
            <requestthreads>
                <threads>32</threads>
            </requestthreads>
            <search>
                <memory>
                    <activedocs>
                        <ratio>0.2</ratio>
                    </activedocs>
                </memory>
            </search>
        </searchnode>
    </tuning>
</content>
```

## Code References

### Core Implementation
- `searchlib/src/vespa/searchlib/tensor/` - Tensor and distance functions
- `vespalib/src/vespa/vespalib/hwaccelrated/` - SIMD implementations
- `eval/src/vespa/eval/` - Tensor evaluation framework
- `searchlib/src/vespa/searchlib/attribute/` - Attribute storage

## Comparison Notes
- Most comprehensive optimization approach
- Designed for both batch and real-time workloads
- Excellent for large-scale production deployments
- Trade-off: Complexity vs. performance and flexibility