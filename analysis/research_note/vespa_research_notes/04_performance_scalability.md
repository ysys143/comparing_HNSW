# Vespa Performance & Scalability Analysis

## Overview

Vespa is designed for real-time serving at scale with a unique architecture that separates content nodes (storage/indexing) from container nodes (query processing). The system employs aggressive optimizations including thread-per-core design, lock-free data structures, and custom memory allocators for predictable low-latency performance.

## Memory Management

### 1. **Custom Memory Allocators**

```cpp
// searchlib/src/vespa/searchlib/common/allocator.h
namespace vespalib::alloc {

class Allocator {
public:
    enum Type {
        HEAP,           // Standard malloc/free
        MMAP,           // Memory-mapped allocation
        HUGE_PAGE,      // Huge page allocation
        ALIGNED_HEAP    // Aligned allocation for SIMD
    };
    
    static Alloc alloc(size_t sz, Type type = HEAP) {
        switch (type) {
            case MMAP:
                return allocMMap(sz);
            case HUGE_PAGE:
                return allocHugePage(sz);
            case ALIGNED_HEAP:
                return allocAlignedHeap(sz);
            default:
                return allocHeap(sz);
        }
    }
    
private:
    static Alloc allocMMap(size_t sz) {
        void *ptr = mmap(nullptr, sz, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            throw std::bad_alloc();
        }
        return Alloc(ptr, sz, [](void *p, size_t s) { munmap(p, s); });
    }
    
    static Alloc allocHugePage(size_t sz) {
        // Align to huge page size (2MB on x86_64)
        static constexpr size_t HUGE_PAGE_SIZE = 2 * 1024 * 1024;
        sz = (sz + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);
        
        void *ptr = mmap(nullptr, sz, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (ptr == MAP_FAILED) {
            // Fallback to regular mmap
            return allocMMap(sz);
        }
        return Alloc(ptr, sz, [](void *p, size_t s) { munmap(p, s); });
    }
};

// Memory pool for vector operations
template<typename T>
class VectorMemoryPool {
private:
    struct Block {
        std::unique_ptr<T[]> data;
        std::atomic<size_t> used{0};
        size_t capacity;
        
        Block(size_t cap) : data(std::make_unique<T[]>(cap)), capacity(cap) {}
    };
    
    std::vector<std::unique_ptr<Block>> blocks;
    std::atomic<size_t> current_block{0};
    const size_t block_size;
    mutable std::mutex expand_mutex;
    
public:
    VectorMemoryPool(size_t block_sz = 1024 * 1024) : block_size(block_sz) {
        blocks.push_back(std::make_unique<Block>(block_size));
    }
    
    T* allocate(size_t n) {
        while (true) {
            size_t block_idx = current_block.load(std::memory_order_acquire);
            if (block_idx >= blocks.size()) {
                expandPool();
                continue;
            }
            
            Block& block = *blocks[block_idx];
            size_t old_used = block.used.fetch_add(n, std::memory_order_relaxed);
            
            if (old_used + n <= block.capacity) {
                return &block.data[old_used];
            }
            
            // This block is full, try next
            current_block.compare_exchange_weak(block_idx, block_idx + 1);
        }
    }
    
private:
    void expandPool() {
        std::lock_guard<std::mutex> lock(expand_mutex);
        if (current_block.load() >= blocks.size()) {
            blocks.push_back(std::make_unique<Block>(block_size));
        }
    }
};
}
```

### 2. **Proton Memory Management**

```cpp
// searchcore/src/vespa/searchcore/proton/server/memory_flush_config_updater.cpp
class MemoryFlushConfigUpdater : public IFlushStrategyFactory {
private:
    const ResourceUsageState &_resourceState;
    std::atomic<uint64_t> _totalMemory;
    std::atomic<uint64_t> _memoryLimit;
    
public:
    void updateFlushStrategy() {
        uint64_t totalMem = _totalMemory.load();
        uint64_t usedMem = _resourceState.memoryUsage();
        double utilization = static_cast<double>(usedMem) / totalMem;
        
        if (utilization > 0.9) {
            // Aggressive flush when memory pressure is high
            triggerEmergencyFlush();
        } else if (utilization > 0.7) {
            // Normal flush strategy
            triggerNormalFlush();
        }
    }
    
    void triggerEmergencyFlush() {
        // Flush largest memory consumers first
        auto targets = identifyFlushTargets();
        std::sort(targets.begin(), targets.end(),
            [](const auto& a, const auto& b) {
                return a->getMemoryUsage() > b->getMemoryUsage();
            });
        
        for (auto& target : targets) {
            if (target->getMemoryUsage() > EMERGENCY_THRESHOLD) {
                target->flush();
            }
        }
    }
};

// Attribute vector memory management
class AttributeMemoryManager {
private:
    std::unique_ptr<vespalib::alloc::Alloc> _buffer;
    std::atomic<size_t> _usedBytes{0};
    const size_t _growFactor = 2;
    
public:
    void ensureCapacity(size_t needed) {
        size_t used = _usedBytes.load(std::memory_order_relaxed);
        size_t capacity = _buffer ? _buffer->size() : 0;
        
        if (used + needed > capacity) {
            size_t newCapacity = std::max(capacity * _growFactor, used + needed);
            growBuffer(newCapacity);
        }
    }
    
    void growBuffer(size_t newCapacity) {
        // Allocate new buffer with huge pages for large allocations
        auto allocType = newCapacity > 100 * 1024 * 1024 
            ? vespalib::alloc::Allocator::HUGE_PAGE 
            : vespalib::alloc::Allocator::MMAP;
            
        auto newBuffer = vespalib::alloc::Allocator::alloc(newCapacity, allocType);
        
        // Copy existing data
        if (_buffer) {
            memcpy(newBuffer.get(), _buffer->get(), _usedBytes.load());
        }
        
        _buffer = std::make_unique<vespalib::alloc::Alloc>(std::move(newBuffer));
    }
};
```

### 3. **Resource Limits and Budgets**

```cpp
// searchcore/src/vespa/searchcore/proton/server/resource_usage_tracker.cpp
class ResourceUsageTracker {
private:
    struct ResourceLimits {
        double memoryLimit = 0.9;      // 90% of available memory
        double diskLimit = 0.85;       // 85% of available disk
        double cpuLimit = 0.95;        // 95% CPU utilization
        size_t openFilesLimit = 60000; // File descriptor limit
    };
    
    ResourceLimits _limits;
    std::atomic<bool> _aboveLimit{false};
    
public:
    ResourceUsageState sampleUsage() {
        ResourceUsageState state;
        
        // Memory usage
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        state.memoryUsed = usage.ru_maxrss * 1024; // Convert to bytes
        
        // Disk usage
        struct statvfs stat;
        statvfs(".", &stat);
        state.diskUsed = (stat.f_blocks - stat.f_bavail) * stat.f_frsize;
        state.diskTotal = stat.f_blocks * stat.f_frsize;
        
        // Open files
        state.openFiles = countOpenFiles();
        
        // Check limits
        checkResourceLimits(state);
        
        return state;
    }
    
    void checkResourceLimits(const ResourceUsageState& state) {
        bool aboveLimit = false;
        
        if (state.memoryUsage() > _limits.memoryLimit) {
            LOG(warning, "Memory usage above limit: %.2f%%", 
                state.memoryUsage() * 100);
            aboveLimit = true;
        }
        
        if (state.diskUsage() > _limits.diskLimit) {
            LOG(warning, "Disk usage above limit: %.2f%%", 
                state.diskUsage() * 100);
            aboveLimit = true;
        }
        
        _aboveLimit.store(aboveLimit, std::memory_order_release);
        
        if (aboveLimit) {
            notifyResourcePressure();
        }
    }
};
```

## Concurrency Model

### 1. **Thread-Per-Core Architecture**

```cpp
// vespalib/src/vespa/vespalib/util/cpu_usage.cpp
class CpuUsage {
private:
    struct ThreadSample {
        std::thread::id thread_id;
        uint64_t user_time;
        uint64_t system_time;
        int cpu_id;
    };
    
    std::vector<ThreadSample> _samples;
    std::mutex _lock;
    
public:
    void bindThreadToCpu(int cpu_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        
        int result = pthread_setaffinity_np(pthread_self(), 
                                          sizeof(cpu_set_t), &cpuset);
        if (result != 0) {
            LOG(warning, "Failed to bind thread to CPU %d", cpu_id);
        }
        
        // Set thread name for debugging
        char name[16];
        snprintf(name, sizeof(name), "vespa-cpu-%d", cpu_id);
        pthread_setname_np(pthread_self(), name);
    }
    
    void optimizeNumaAffinity() {
        int numa_node = numa_node_of_cpu(sched_getcpu());
        
        // Bind memory allocations to local NUMA node
        numa_set_preferred(numa_node);
        numa_set_localalloc();
        
        // Pre-fault memory on local node
        size_t prealloc_size = 1024 * 1024 * 1024; // 1GB
        void* mem = numa_alloc_onnode(prealloc_size, numa_node);
        memset(mem, 0, prealloc_size);
        numa_free(mem, prealloc_size);
    }
};

// Thread pool with CPU affinity
class AffinityThreadPool {
private:
    struct Worker {
        std::thread thread;
        std::atomic<bool> should_stop{false};
        BlockingQueue<std::function<void()>> tasks;
        int cpu_id;
        
        void run() {
            // Bind to specific CPU
            CpuUsage::bindThreadToCpu(cpu_id);
            CpuUsage::optimizeNumaAffinity();
            
            while (!should_stop.load(std::memory_order_relaxed)) {
                std::function<void()> task;
                if (tasks.pop(task, std::chrono::milliseconds(100))) {
                    task();
                }
            }
        }
    };
    
    std::vector<std::unique_ptr<Worker>> _workers;
    
public:
    AffinityThreadPool(size_t num_threads) {
        _workers.reserve(num_threads);
        
        for (size_t i = 0; i < num_threads; ++i) {
            auto worker = std::make_unique<Worker>();
            worker->cpu_id = i % std::thread::hardware_concurrency();
            worker->thread = std::thread(&Worker::run, worker.get());
            _workers.push_back(std::move(worker));
        }
    }
    
    void submit(std::function<void()> task, size_t thread_hint = 0) {
        size_t worker_idx = thread_hint % _workers.size();
        _workers[worker_idx]->tasks.push(std::move(task));
    }
};
```

### 2. **Lock-Free Data Structures**

```cpp
// vespalib/src/vespa/vespalib/util/lock_free_queue.hpp
template<typename T>
class LockFreeQueue {
private:
    struct Node {
        std::atomic<T*> data{nullptr};
        std::atomic<Node*> next{nullptr};
    };
    
    alignas(64) std::atomic<Node*> _head;
    alignas(64) std::atomic<Node*> _tail;
    
public:
    LockFreeQueue() {
        Node* dummy = new Node;
        _head.store(dummy);
        _tail.store(dummy);
    }
    
    void push(T item) {
        Node* newNode = new Node;
        T* data = new T(std::move(item));
        newNode->data.store(data, std::memory_order_relaxed);
        
        Node* prevTail = _tail.exchange(newNode, std::memory_order_acq_rel);
        prevTail->next.store(newNode, std::memory_order_release);
    }
    
    bool pop(T& item) {
        Node* head = _head.load(std::memory_order_acquire);
        Node* next = head->next.load(std::memory_order_acquire);
        
        if (next == nullptr) {
            return false;
        }
        
        T* data = next->data.exchange(nullptr, std::memory_order_acquire);
        if (data == nullptr) {
            return false; // Another thread got it
        }
        
        // Try to advance head
        _head.compare_exchange_weak(head, next, std::memory_order_release);
        
        item = std::move(*data);
        delete data;
        delete head;
        
        return true;
    }
};

// Lock-free vector index for concurrent updates
class ConcurrentVectorIndex {
private:
    struct Segment {
        static constexpr size_t SIZE = 1024;
        std::atomic<float*> vectors[SIZE];
        std::atomic<uint32_t> versions[SIZE];
        std::atomic<size_t> count{0};
    };
    
    std::vector<std::unique_ptr<Segment>> _segments;
    std::atomic<size_t> _size{0};
    
public:
    void insert(size_t id, const float* vector, size_t dim) {
        size_t segment_idx = id / Segment::SIZE;
        size_t local_idx = id % Segment::SIZE;
        
        // Ensure segment exists
        ensureSegment(segment_idx);
        
        // Allocate and copy vector
        float* new_vector = new float[dim];
        std::memcpy(new_vector, vector, dim * sizeof(float));
        
        // Update with version increment
        Segment& segment = *_segments[segment_idx];
        float* old_vector = segment.vectors[local_idx].exchange(new_vector);
        segment.versions[local_idx].fetch_add(1, std::memory_order_release);
        
        if (old_vector == nullptr) {
            segment.count.fetch_add(1, std::memory_order_relaxed);
            _size.fetch_add(1, std::memory_order_relaxed);
        } else {
            delete[] old_vector;
        }
    }
    
    bool get(size_t id, float* out_vector, size_t dim) {
        size_t segment_idx = id / Segment::SIZE;
        size_t local_idx = id % Segment::SIZE;
        
        if (segment_idx >= _segments.size()) {
            return false;
        }
        
        Segment& segment = *_segments[segment_idx];
        
        // Read with version check for consistency
        uint32_t version_before = segment.versions[local_idx].load(std::memory_order_acquire);
        float* vector = segment.vectors[local_idx].load(std::memory_order_acquire);
        
        if (vector == nullptr) {
            return false;
        }
        
        std::memcpy(out_vector, vector, dim * sizeof(float));
        
        uint32_t version_after = segment.versions[local_idx].load(std::memory_order_acquire);
        
        // Retry if version changed during read
        if (version_before != version_after) {
            return get(id, out_vector, dim);
        }
        
        return true;
    }
    
private:
    void ensureSegment(size_t idx) {
        if (idx >= _segments.size()) {
            static std::mutex resize_mutex;
            std::lock_guard<std::mutex> lock(resize_mutex);
            
            while (idx >= _segments.size()) {
                _segments.push_back(std::make_unique<Segment>());
            }
        }
    }
};
```

### 3. **Message Passing Architecture**

```cpp
// messagebus/src/vespa/messagebus/messenger.cpp
class Messenger : public IMessenger {
private:
    struct Task {
        virtual ~Task() = default;
        virtual void run() = 0;
    };
    
    template<typename Handler>
    struct ConcreteTask : Task {
        Handler handler;
        ConcreteTask(Handler h) : handler(std::move(h)) {}
        void run() override { handler(); }
    };
    
    LockFreeQueue<std::unique_ptr<Task>> _queue;
    std::vector<std::thread> _threads;
    std::atomic<bool> _running{true};
    
public:
    Messenger(size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            _threads.emplace_back([this] { run(); });
        }
    }
    
    template<typename Handler>
    void enqueue(Handler&& handler) {
        _queue.push(std::make_unique<ConcreteTask<Handler>>(
            std::forward<Handler>(handler)));
    }
    
private:
    void run() {
        while (_running.load(std::memory_order_relaxed)) {
            std::unique_ptr<Task> task;
            if (_queue.pop(task)) {
                task->run();
            } else {
                std::this_thread::yield();
            }
        }
    }
};

// Async message handling for vector operations
class VectorMessageBus {
private:
    Messenger _messenger;
    std::unordered_map<std::string, MessageHandler> _handlers;
    
public:
    VectorMessageBus() : _messenger(std::thread::hardware_concurrency()) {}
    
    void sendVectorUpdate(const std::string& index, size_t doc_id, 
                         const std::vector<float>& vector) {
        _messenger.enqueue([this, index, doc_id, vector] {
            auto it = _handlers.find("vector_update");
            if (it != _handlers.end()) {
                VectorUpdateMessage msg{index, doc_id, vector};
                it->second(msg);
            }
        });
    }
    
    void sendVectorQuery(const std::string& index, 
                        const std::vector<float>& query,
                        size_t k,
                        std::function<void(QueryResult)> callback) {
        _messenger.enqueue([this, index, query, k, callback] {
            auto it = _handlers.find("vector_query");
            if (it != _handlers.end()) {
                VectorQueryMessage msg{index, query, k};
                QueryResult result = it->second(msg);
                callback(result);
            }
        });
    }
};
```

## I/O Optimization

### 1. **Async I/O with io_uring**

```cpp
// fastos/src/vespa/fastos/linux_file_async.cpp
class LinuxFileAsync : public FastOS_FileInterface {
private:
    struct io_uring _ring;
    static constexpr size_t QUEUE_DEPTH = 256;
    
public:
    LinuxFileAsync() {
        struct io_uring_params params = {};
        params.flags = IORING_SETUP_SQPOLL | IORING_SETUP_SQ_AFF;
        params.sq_thread_cpu = sched_getcpu();
        params.sq_thread_idle = 10000; // 10ms
        
        int ret = io_uring_queue_init_params(QUEUE_DEPTH, &_ring, &params);
        if (ret < 0) {
            throw std::runtime_error("Failed to initialize io_uring");
        }
    }
    
    void readAsync(void* buffer, size_t length, off_t offset,
                   std::function<void(ssize_t)> callback) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&_ring);
        if (!sqe) {
            throw std::runtime_error("io_uring queue full");
        }
        
        auto* req = new ReadRequest{buffer, callback};
        
        io_uring_prep_read(sqe, _fd, buffer, length, offset);
        io_uring_sqe_set_data(sqe, req);
        io_uring_sqe_set_flags(sqe, IOSQE_ASYNC);
        
        io_uring_submit(&_ring);
    }
    
    void writeAsync(const void* buffer, size_t length, off_t offset,
                    std::function<void(ssize_t)> callback) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&_ring);
        if (!sqe) {
            throw std::runtime_error("io_uring queue full");
        }
        
        auto* req = new WriteRequest{buffer, callback};
        
        io_uring_prep_write(sqe, _fd, buffer, length, offset);
        io_uring_sqe_set_data(sqe, req);
        io_uring_sqe_set_flags(sqe, IOSQE_ASYNC);
        
        // Link multiple writes for ordering
        if (_has_pending_writes) {
            io_uring_sqe_set_flags(sqe, IOSQE_IO_LINK);
        }
        
        io_uring_submit(&_ring);
        _has_pending_writes = true;
    }
    
    void processCompletions() {
        struct io_uring_cqe* cqe;
        unsigned head;
        
        io_uring_for_each_cqe(&_ring, head, cqe) {
            auto* req = static_cast<RequestBase*>(io_uring_cqe_get_data(cqe));
            req->callback(cqe->res);
            delete req;
        }
        
        io_uring_cq_advance(&_ring, head);
    }
    
    // Vectored I/O for efficiency
    void readvAsync(const struct iovec* iov, int iovcnt, off_t offset,
                    std::function<void(ssize_t)> callback) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&_ring);
        
        auto* req = new VectoredReadRequest{iov, iovcnt, callback};
        
        io_uring_prep_readv(sqe, _fd, iov, iovcnt, offset);
        io_uring_sqe_set_data(sqe, req);
        
        io_uring_submit(&_ring);
    }
};
```

### 2. **Memory-Mapped Files and Direct I/O**

```cpp
// searchlib/src/vespa/searchlib/common/mappedfilehandle.cpp
class MappedFileHandle {
private:
    void* _mapAddr{nullptr};
    size_t _mapSize{0};
    int _fd{-1};
    bool _readonly;
    
public:
    void* map(size_t offset, size_t length, bool populate = false) {
        // Align offset to page boundary
        size_t page_size = sysconf(_SC_PAGESIZE);
        size_t aligned_offset = offset & ~(page_size - 1);
        size_t adjustment = offset - aligned_offset;
        size_t aligned_length = length + adjustment;
        
        int prot = _readonly ? PROT_READ : (PROT_READ | PROT_WRITE);
        int flags = MAP_SHARED;
        
        if (populate) {
            flags |= MAP_POPULATE | MAP_LOCKED;
        }
        
        void* addr = mmap(nullptr, aligned_length, prot, flags, 
                         _fd, aligned_offset);
        
        if (addr == MAP_FAILED) {
            throw std::runtime_error("mmap failed");
        }
        
        // Advise kernel about access pattern
        if (madvise(addr, aligned_length, MADV_SEQUENTIAL) != 0) {
            LOG(debug, "madvise failed: %s", strerror(errno));
        }
        
        // Touch pages to pre-fault if requested
        if (populate) {
            volatile char* p = static_cast<char*>(addr);
            for (size_t i = 0; i < aligned_length; i += page_size) {
                p[i];
            }
        }
        
        return static_cast<char*>(addr) + adjustment;
    }
    
    void prefetch(void* addr, size_t length) {
        if (madvise(addr, length, MADV_WILLNEED) != 0) {
            LOG(debug, "madvise WILLNEED failed: %s", strerror(errno));
        }
        
        // Also use posix_fadvise for file-level prefetching
        off_t offset = static_cast<char*>(addr) - static_cast<char*>(_mapAddr);
        posix_fadvise(_fd, offset, length, POSIX_FADV_WILLNEED);
    }
};

// Direct I/O for large sequential writes
class DirectIOWriter {
private:
    int _fd;
    size_t _alignment;
    std::unique_ptr<char[]> _alignedBuffer;
    size_t _bufferSize;
    size_t _bufferUsed{0};
    
public:
    DirectIOWriter(const std::string& filename, size_t bufferSize = 1024 * 1024) 
        : _bufferSize(bufferSize) {
        _alignment = getpagesize();
        
        // Open with O_DIRECT for bypass of page cache
        _fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_DIRECT | O_SYNC, 0644);
        if (_fd < 0) {
            throw std::runtime_error("Failed to open file for direct I/O");
        }
        
        // Allocate aligned buffer
        void* ptr;
        if (posix_memalign(&ptr, _alignment, _bufferSize) != 0) {
            throw std::bad_alloc();
        }
        _alignedBuffer.reset(static_cast<char*>(ptr));
    }
    
    void write(const void* data, size_t size) {
        const char* src = static_cast<const char*>(data);
        
        while (size > 0) {
            size_t toWrite = std::min(size, _bufferSize - _bufferUsed);
            
            memcpy(_alignedBuffer.get() + _bufferUsed, src, toWrite);
            _bufferUsed += toWrite;
            src += toWrite;
            size -= toWrite;
            
            if (_bufferUsed == _bufferSize) {
                flush();
            }
        }
    }
    
    void flush() {
        if (_bufferUsed > 0) {
            // Pad to alignment boundary
            size_t alignedSize = (_bufferUsed + _alignment - 1) & ~(_alignment - 1);
            
            // Zero padding
            memset(_alignedBuffer.get() + _bufferUsed, 0, alignedSize - _bufferUsed);
            
            ssize_t written = ::write(_fd, _alignedBuffer.get(), alignedSize);
            if (written != alignedSize) {
                throw std::runtime_error("Direct I/O write failed");
            }
            
            _bufferUsed = 0;
        }
    }
};
```

### 3. **Document Store Optimization**

```cpp
// searchlib/src/vespa/searchlib/docstore/logdatastore.cpp
class LogDataStore : public IDataStore {
private:
    struct WriteCache {
        std::vector<std::pair<uint32_t, vespalib::DataBuffer>> _entries;
        size_t _totalSize{0};
        static constexpr size_t FLUSH_SIZE = 8 * 1024 * 1024; // 8MB
        
        void add(uint32_t lid, vespalib::DataBuffer&& data) {
            _totalSize += data.getDataLen();
            _entries.emplace_back(lid, std::move(data));
            
            if (_totalSize >= FLUSH_SIZE) {
                return true; // Signal flush needed
            }
            return false;
        }
    };
    
    WriteCache _writeCache;
    std::unique_ptr<CompressionStrategy> _compression;
    
public:
    void write(uint32_t lid, const void* data, size_t len) {
        vespalib::DataBuffer buffer;
        
        // Compress if beneficial
        if (_compression && len > 1024) {
            vespalib::DataBuffer compressed;
            if (_compression->compress(data, len, compressed)) {
                if (compressed.getDataLen() < len * 0.8) {
                    buffer = std::move(compressed);
                } else {
                    buffer.writeBytes(data, len);
                }
            }
        } else {
            buffer.writeBytes(data, len);
        }
        
        if (_writeCache.add(lid, std::move(buffer))) {
            flushWriteCache();
        }
    }
    
    void flushWriteCache() {
        if (_writeCache._entries.empty()) return;
        
        // Sort by file offset for sequential writes
        std::sort(_writeCache._entries.begin(), _writeCache._entries.end(),
            [this](const auto& a, const auto& b) {
                return getFileOffset(a.first) < getFileOffset(b.first);
            });
        
        // Batch write with io_uring
        std::vector<struct iovec> iovecs;
        iovecs.reserve(_writeCache._entries.size() * 2); // Header + data
        
        for (const auto& [lid, buffer] : _writeCache._entries) {
            // Write header
            LidHeader header{lid, buffer.getDataLen()};
            iovecs.push_back({&header, sizeof(header)});
            
            // Write data
            iovecs.push_back({buffer.getData(), buffer.getDataLen()});
        }
        
        // Submit vectored write
        _file->writev(iovecs.data(), iovecs.size());
        
        _writeCache._entries.clear();
        _writeCache._totalSize = 0;
    }
};

// Compression strategy for document store
class ZstdCompressionStrategy : public CompressionStrategy {
private:
    ZSTD_CCtx* _cctx;
    ZSTD_DCtx* _dctx;
    int _level;
    
public:
    ZstdCompressionStrategy(int level = 3) : _level(level) {
        _cctx = ZSTD_createCCtx();
        _dctx = ZSTD_createDCtx();
        
        // Configure for speed
        ZSTD_CCtx_setParameter(_cctx, ZSTD_c_compressionLevel, _level);
        ZSTD_CCtx_setParameter(_cctx, ZSTD_c_strategy, ZSTD_fast);
    }
    
    bool compress(const void* src, size_t srcSize, 
                 vespalib::DataBuffer& dst) override {
        size_t dstCapacity = ZSTD_compressBound(srcSize);
        dst.ensureFree(dstCapacity);
        
        size_t compressedSize = ZSTD_compressCCtx(_cctx,
            dst.getFree(), dstCapacity,
            src, srcSize,
            _level);
        
        if (ZSTD_isError(compressedSize)) {
            return false;
        }
        
        dst.moveFreeToData(compressedSize);
        return true;
    }
};
```

## Performance Monitoring and Optimization

### 1. **Metrics Collection Framework**

```cpp
// metrics/src/vespa/metrics/metricmanager.cpp
class VectorMetricsManager {
private:
    struct VectorMetrics {
        std::atomic<uint64_t> searchCount{0};
        std::atomic<uint64_t> searchLatencySum{0};
        std::atomic<uint64_t> insertCount{0};
        std::atomic<uint64_t> insertLatencySum{0};
        std::atomic<uint64_t> memoryUsage{0};
        std::atomic<uint64_t> diskUsage{0};
        
        void recordSearch(uint64_t latencyUs) {
            searchCount.fetch_add(1, std::memory_order_relaxed);
            searchLatencySum.fetch_add(latencyUs, std::memory_order_relaxed);
        }
        
        double getAverageSearchLatency() const {
            uint64_t count = searchCount.load(std::memory_order_relaxed);
            if (count == 0) return 0.0;
            return searchLatencySum.load(std::memory_order_relaxed) / double(count);
        }
    };
    
    std::unordered_map<std::string, VectorMetrics> _metrics;
    std::mutex _metricsMutex;
    
public:
    class Timer {
        VectorMetrics* _metrics;
        std::chrono::steady_clock::time_point _start;
        bool _isSearch;
        
    public:
        Timer(VectorMetrics* metrics, bool isSearch) 
            : _metrics(metrics), _isSearch(isSearch),
              _start(std::chrono::steady_clock::now()) {}
        
        ~Timer() {
            auto duration = std::chrono::steady_clock::now() - _start;
            uint64_t microseconds = std::chrono::duration_cast<
                std::chrono::microseconds>(duration).count();
            
            if (_isSearch) {
                _metrics->recordSearch(microseconds);
            } else {
                _metrics->recordInsert(microseconds);
            }
        }
    };
    
    Timer timeSearch(const std::string& index) {
        return Timer(&getMetrics(index), true);
    }
    
    Timer timeInsert(const std::string& index) {
        return Timer(&getMetrics(index), false);
    }
    
    void updateMemoryUsage(const std::string& index, size_t bytes) {
        getMetrics(index).memoryUsage.store(bytes, std::memory_order_relaxed);
    }
    
    vespalib::string getMetricsSnapshot() {
        vespalib::asciistream json;
        json << "{\n";
        
        std::lock_guard<std::mutex> lock(_metricsMutex);
        bool first = true;
        
        for (const auto& [index, metrics] : _metrics) {
            if (!first) json << ",\n";
            first = false;
            
            json << "  \"" << index << "\": {\n"
                 << "    \"searches\": " << metrics.searchCount << ",\n"
                 << "    \"avg_search_latency_us\": " 
                 << metrics.getAverageSearchLatency() << ",\n"
                 << "    \"inserts\": " << metrics.insertCount << ",\n"
                 << "    \"memory_bytes\": " << metrics.memoryUsage << ",\n"
                 << "    \"disk_bytes\": " << metrics.diskUsage << "\n"
                 << "  }";
        }
        
        json << "\n}";
        return json.str();
    }
private:
    VectorMetrics& getMetrics(const std::string& index) {
        std::lock_guard<std::mutex> lock(_metricsMutex);
        return _metrics[index];
    }
};
```

### 2. **Performance Profiling**

```cpp
// vespalib/src/vespa/vespalib/util/cpu_profiler.cpp
class CpuProfiler {
private:
    struct ProfileData {
        std::string function;
        uint64_t samples{0};
        uint64_t self_time{0};
        uint64_t total_time{0};
        std::vector<ProfileData*> children;
    };
    
    thread_local ProfileData* _currentNode{nullptr};
    std::unordered_map<std::string, std::unique_ptr<ProfileData>> _roots;
    std::mutex _mutex;
    
public:
    class ScopedTimer {
        ProfileData* _parent;
        ProfileData* _node;
        std::chrono::steady_clock::time_point _start;
        
    public:
        ScopedTimer(const std::string& function) 
            : _parent(CpuProfiler::_currentNode),
              _start(std::chrono::steady_clock::now()) {
            
            _node = getOrCreateNode(function);
            CpuProfiler::_currentNode = _node;
            _node->samples++;
        }
        
        ~ScopedTimer() {
            auto duration = std::chrono::steady_clock::now() - _start;
            uint64_t microseconds = std::chrono::duration_cast<
                std::chrono::microseconds>(duration).count();
            
            _node->self_time += microseconds;
            _node->total_time += microseconds;
            
            if (_parent) {
                _parent->self_time -= microseconds;
            }
            
            CpuProfiler::_currentNode = _parent;
        }
    };
    
    void enableSampling(int frequency_hz = 100) {
        struct sigaction sa;
        sa.sa_handler = &CpuProfiler::signalHandler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        
        sigaction(SIGPROF, &sa, nullptr);
        
        struct itimerval timer;
        timer.it_value.tv_sec = 0;
        timer.it_value.tv_usec = 1000000 / frequency_hz;
        timer.it_interval = timer.it_value;
        
        setitimer(ITIMER_PROF, &timer, nullptr);
    }
    
    static void signalHandler(int sig) {
        if (_currentNode) {
            _currentNode->samples++;
        }
    }
    
    vespalib::string generateFlameGraph() {
        std::lock_guard<std::mutex> lock(_mutex);
        vespalib::asciistream out;
        
        for (const auto& [name, root] : _roots) {
            generateFlameGraphNode(out, root.get(), "");
        }
        
        return out.str();
    }
    
private:
    void generateFlameGraphNode(vespalib::asciistream& out, 
                               ProfileData* node, 
                               const std::string& stack) {
        std::string fullStack = stack.empty() ? node->function 
                                              : stack + ";" + node->function;
        out << fullStack << " " << node->samples << "\n";
        
        for (ProfileData* child : node->children) {
            generateFlameGraphNode(out, child, fullStack);
        }
    }
};

#define PROFILE_FUNCTION() CpuProfiler::ScopedTimer _timer(__FUNCTION__)
#define PROFILE_SCOPE(name) CpuProfiler::ScopedTimer _timer(name)
```

### 3. **Adaptive Performance Tuning**

```cpp
// searchcore/src/vespa/searchcore/proton/server/adaptive_feed_handler.cpp
class AdaptiveFeedHandler {
private:
    struct PerformanceWindow {
        static constexpr size_t WINDOW_SIZE = 1000;
        std::array<uint64_t, WINDOW_SIZE> latencies;
        size_t index{0};
        size_t count{0};
        
        void addSample(uint64_t latency) {
            latencies[index] = latency;
            index = (index + 1) % WINDOW_SIZE;
            count = std::min(count + 1, WINDOW_SIZE);
        }
        
        uint64_t getPercentile(double p) const {
            if (count == 0) return 0;
            
            std::vector<uint64_t> sorted(latencies.begin(), 
                                        latencies.begin() + count);
            std::sort(sorted.begin(), sorted.end());
            
            size_t idx = static_cast<size_t>(p * count);
            return sorted[std::min(idx, count - 1)];
        }
    };
    
    PerformanceWindow _feedLatencies;
    std::atomic<size_t> _batchSize{100};
    std::atomic<size_t> _flushInterval{1000}; // ms
    
public:
    void adaptBatchSize() {
        uint64_t p50 = _feedLatencies.getPercentile(0.5);
        uint64_t p99 = _feedLatencies.getPercentile(0.99);
        
        // Target: p99 < 100ms, p50 < 10ms
        if (p99 > 100000) { // 100ms in microseconds
            // Reduce batch size to improve latency
            size_t newSize = _batchSize.load() * 0.8;
            _batchSize.store(std::max(size_t(10), newSize));
        } else if (p99 < 50000 && p50 < 5000) {
            // Can increase batch size for better throughput
            size_t newSize = _batchSize.load() * 1.2;
            _batchSize.store(std::min(size_t(1000), newSize));
        }
        
        LOG(debug, "Adapted batch size to %zu (p50=%zu us, p99=%zu us)",
            _batchSize.load(), p50, p99);
    }
    
    void adaptFlushInterval() {
        // Analyze memory pressure and adapt flush interval
        ResourceUsageState state = sampleResourceUsage();
        
        if (state.memoryUsage() > 0.8) {
            // High memory pressure - flush more frequently
            _flushInterval.store(_flushInterval.load() * 0.5);
        } else if (state.memoryUsage() < 0.5) {
            // Low memory pressure - can batch more
            _flushInterval.store(_flushInterval.load() * 1.5);
        }
        
        // Clamp to reasonable range
        _flushInterval.store(std::clamp(_flushInterval.load(), 
                                       size_t(100), size_t(10000)));
    }
};
```

## Scalability Architecture

### 1. **Content Cluster Distribution**

```cpp
// searchcore/src/vespa/searchcore/proton/server/distributor.cpp
class ContentDistributor {
private:
    struct Node {
        std::string hostname;
        int port;
        std::atomic<uint64_t> load{0};
        std::atomic<bool> active{true};
    };
    
    std::vector<Node> _nodes;
    std::hash<std::string> _hasher;
    
public:
    Node& selectNode(const std::string& documentId) {
        // Consistent hashing with virtual nodes
        static constexpr int VIRTUAL_NODES = 100;
        
        size_t hash = _hasher(documentId);
        
        // Find active nodes
        std::vector<size_t> activeNodes;
        for (size_t i = 0; i < _nodes.size(); ++i) {
            if (_nodes[i].active.load()) {
                activeNodes.push_back(i);
            }
        }
        
        if (activeNodes.empty()) {
            throw std::runtime_error("No active nodes available");
        }
        
        // Map to virtual node space
        size_t virtualNode = hash % (activeNodes.size() * VIRTUAL_NODES);
        size_t nodeIndex = activeNodes[virtualNode / VIRTUAL_NODES];
        
        return _nodes[nodeIndex];
    }
    
    void redistribute(const std::vector<std::string>& removedNodes) {
        // Mark nodes as inactive
        for (auto& node : _nodes) {
            if (std::find(removedNodes.begin(), removedNodes.end(), 
                         node.hostname) != removedNodes.end()) {
                node.active.store(false);
            }
        }
        
        // Trigger redistribution of affected documents
        triggerRedistribution();
    }
    
    void rebalance() {
        // Calculate ideal load per node
        uint64_t totalLoad = 0;
        size_t activeCount = 0;
        
        for (const auto& node : _nodes) {
            if (node.active.load()) {
                totalLoad += node.load.load();
                activeCount++;
            }
        }
        
        if (activeCount == 0) return;
        
        uint64_t idealLoad = totalLoad / activeCount;
        
        // Find overloaded and underloaded nodes
        std::vector<size_t> overloaded, underloaded;
        
        for (size_t i = 0; i < _nodes.size(); ++i) {
            if (!_nodes[i].active.load()) continue;
            
            uint64_t load = _nodes[i].load.load();
            if (load > idealLoad * 1.2) {
                overloaded.push_back(i);
            } else if (load < idealLoad * 0.8) {
                underloaded.push_back(i);
            }
        }
        
        // Move documents from overloaded to underloaded nodes
        for (size_t from : overloaded) {
            for (size_t to : underloaded) {
                moveDocuments(_nodes[from], _nodes[to], 
                            _nodes[from].load - idealLoad);
            }
        }
    }
};
```

### 2. **Container Cluster Query Routing**

```cpp
// container-search/src/main/java/com/yahoo/search/dispatch/Dispatcher.cpp
class QueryDispatcher {
private:
    struct SearchNode {
        std::string id;
        std::shared_ptr<RpcClient> client;
        std::atomic<int> inFlightQueries{0};
        std::atomic<uint64_t> totalLatency{0};
        std::atomic<uint64_t> queryCount{0};
        
        double getAverageLatency() const {
            uint64_t count = queryCount.load();
            if (count == 0) return 0.0;
            return totalLatency.load() / double(count);
        }
    };
    
    std::vector<SearchNode> _searchNodes;
    
public:
    std::future<SearchResult> dispatch(const VectorQuery& query) {
        // Select nodes based on load and latency
        auto selectedNodes = selectNodes(query.getNumNodesNeeded());
        
        // Fan out query to selected nodes
        std::vector<std::future<PartialResult>> futures;
        
        for (auto& node : selectedNodes) {
            node->inFlightQueries.fetch_add(1);
            
            futures.push_back(std::async(std::launch::async, 
                [this, &node, &query] {
                    auto start = std::chrono::steady_clock::now();
                    
                    try {
                        auto result = node->client->search(query);
                        
                        auto duration = std::chrono::steady_clock::now() - start;
                        uint64_t latency = std::chrono::duration_cast<
                            std::chrono::microseconds>(duration).count();
                        
                        node->totalLatency.fetch_add(latency);
                        node->queryCount.fetch_add(1);
                        node->inFlightQueries.fetch_sub(1);
                        
                        return result;
                    } catch (...) {
                        node->inFlightQueries.fetch_sub(1);
                        throw;
                    }
                }
            ));
        }
        
        // Merge results
        return std::async(std::launch::async, [futures = std::move(futures)]() {
            SearchResult merged;
            
            for (auto& future : futures) {
                try {
                    auto partial = future.get();
                    merged.merge(partial);
                } catch (const std::exception& e) {
                    LOG(warning, "Failed to get result from node: %s", e.what());
                }
            }
            
            return merged;
        });
    }
    
private:
    std::vector<SearchNode*> selectNodes(size_t count) {
        // Sort nodes by score (lower is better)
        std::vector<std::pair<double, SearchNode*>> scored;
        
        for (auto& node : _searchNodes) {
            double latency = node.getAverageLatency();
            int inFlight = node.inFlightQueries.load();
            
            // Score combines latency and current load
            double score = latency + inFlight * 10000; // 10ms per query
            scored.emplace_back(score, &node);
        }
        
        std::sort(scored.begin(), scored.end());
        
        // Select best nodes
        std::vector<SearchNode*> selected;
        for (size_t i = 0; i < std::min(count, scored.size()); ++i) {
            selected.push_back(scored[i].second);
        }
        
        return selected;
    }
};
```

### 3. **Elastic Scaling**

```cpp
// config-model/src/main/java/com/yahoo/vespa/model/application/validation/AutoScaler.cpp
class AutoScaler {
private:
    struct ScalingMetrics {
        double cpuUtilization;
        double memoryUtilization;
        double queryLatency;
        double feedLatency;
        int currentNodes;
    };
    
    struct ScalingPolicy {
        double targetCpuUtilization = 0.7;
        double targetMemoryUtilization = 0.8;
        double targetQueryLatency = 50.0; // ms
        int minNodes = 2;
        int maxNodes = 100;
        int scaleUpThreshold = 3; // consecutive periods
        int scaleDownThreshold = 10;
    };
    
    ScalingPolicy _policy;
    std::deque<ScalingMetrics> _history;
    
public:
    int recommendNodeCount(const ScalingMetrics& current) {
        _history.push_back(current);
        if (_history.size() > 20) {
            _history.pop_front();
        }
        
        // Check if we need to scale up
        if (shouldScaleUp()) {
            return std::min(_policy.maxNodes, current.currentNodes + 1);
        }
        
        // Check if we can scale down
        if (shouldScaleDown()) {
            return std::max(_policy.minNodes, current.currentNodes - 1);
        }
        
        return current.currentNodes;
    }
    
private:
    bool shouldScaleUp() {
        if (_history.size() < _policy.scaleUpThreshold) {
            return false;
        }
        
        // Check recent metrics
        for (size_t i = _history.size() - _policy.scaleUpThreshold; 
             i < _history.size(); ++i) {
            const auto& metrics = _history[i];
            
            if (metrics.cpuUtilization < _policy.targetCpuUtilization &&
                metrics.memoryUtilization < _policy.targetMemoryUtilization &&
                metrics.queryLatency < _policy.targetQueryLatency) {
                return false;
            }
        }
        
        return true;
    }
    
    bool shouldScaleDown() {
        if (_history.size() < _policy.scaleDownThreshold) {
            return false;
        }
        
        // All recent metrics should be well below target
        for (size_t i = _history.size() - _policy.scaleDownThreshold; 
             i < _history.size(); ++i) {
            const auto& metrics = _history[i];
            
            if (metrics.cpuUtilization > _policy.targetCpuUtilization * 0.5 ||
                metrics.memoryUtilization > _policy.targetMemoryUtilization * 0.5 ||
                metrics.queryLatency > _policy.targetQueryLatency * 0.5) {
                return false;
            }
        }
        
        return true;
    }
};
```

## Configuration and Tuning

### 1. **System Configuration**

```yaml
# services.xml - Vespa configuration
<services>
  <container id="search" version="1.0">
    <search>
      <chain id="default" inherits="vespa">
        <searcher id="vector-searcher" class="ai.vespa.search.VectorSearcher">
          <config name="vector.config">
            <numThreadsPerSearch>4</numThreadsPerSearch>
            <timeout>100</timeout>
          </config>
        </searcher>
      </chain>
    </search>
    
    <nodes>
      <node hostalias="node1" />
      <node hostalias="node2" />
    </nodes>
    
    <config name="container.handler.threadpool">
      <maxThreads>200</maxThreads>
      <minThreads>50</minThreads>
      <queueSize>1000</queueSize>
    </config>
  </container>
  
  <content id="vectors" version="1.0">
    <redundancy>2</redundancy>
    <documents>
      <document type="vector" mode="index">
        <config name="vespa.config.search.core.proton">
          <indexing>
            <threads>8</threads>
            <tasklimit>2.0</tasklimit>
          </indexing>
          <flush>
            <memory>
              <maxmemory>4294967296</maxmemory> <!-- 4GB -->
              <each>
                <maxmemory>1073741824</maxmemory> <!-- 1GB -->
              </each>
            </memory>
          </flush>
        </config>
      </document>
    </documents>
    
    <nodes>
      <node hostalias="node1" distribution-key="0" />
      <node hostalias="node2" distribution-key="1" />
    </nodes>
    
    <tuning>
      <persistence-threads>
        <thread count="8" />
      </persistence-threads>
    </tuning>
  </content>
</services>
```

### 2. **NUMA and CPU Optimization**

```cpp
// vespalib/src/vespa/vespalib/hwaccelrated/iaccelrated.cpp
class HardwareAccelerated {
public:
    static void optimizeForPlatform() {
        // CPU feature detection
        if (__builtin_cpu_supports("avx2")) {
            LOG(info, "AVX2 support detected");
            enableAvx2Optimizations();
        }
        
        if (__builtin_cpu_supports("avx512f")) {
            LOG(info, "AVX-512 support detected");
            enableAvx512Optimizations();
        }
        
        // NUMA optimization
        int numNodes = numa_num_configured_nodes();
        if (numNodes > 1) {
            LOG(info, "NUMA system detected with %d nodes", numNodes);
            configureNuma();
        }
        
        // Huge pages
        if (isHugePagesAvailable()) {
            LOG(info, "Huge pages available");
            enableHugePages();
        }
    }
    
private:
    static void configureNuma() {
        // Bind threads to NUMA nodes
        numa_set_localalloc();
        
        // Configure memory policy
        struct bitmask* mask = numa_allocate_nodemask();
        numa_bitmask_setall(mask);
        numa_set_membind(mask);
        numa_free_nodemask(mask);
    }
    
    static void enableHugePages() {
        // Set madvise for transparent huge pages
        prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0);
        
        // Configure allocator to use huge pages
        mallopt(M_MMAP_THRESHOLD, 2 * 1024 * 1024); // 2MB
    }
};
```

## Best Practices Summary

### 1. **Memory Management**
- Use custom allocators for predictable performance
- Leverage huge pages for large allocations
- Implement memory pools for frequently allocated objects
- Monitor and respond to memory pressure

### 2. **Concurrency**
- Thread-per-core design for CPU efficiency
- Lock-free data structures where possible
- Message passing for component isolation
- NUMA-aware thread and memory placement

### 3. **I/O Optimization**
- io_uring for async I/O on Linux
- Memory-mapped files for read-heavy workloads
- Direct I/O for large sequential writes
- Compression for document storage

### 4. **Scalability**
- Separate content and container clusters
- Consistent hashing for distribution
- Adaptive load balancing
- Elastic scaling based on metrics

## Code References

- `searchlib/src/vespa/searchlib/` - Core search functionality
- `searchcore/src/vespa/searchcore/proton/` - Document processing engine
- `vespalib/src/vespa/vespalib/` - Core utilities and abstractions
- `storage/src/vespa/storage/` - Distributed storage implementation

## Comparison Notes

- **Advantages**: Extremely low latency, hardware-optimized, mature distributed system
- **Trade-offs**: Complex configuration, resource intensive, steep learning curve
- **Scalability**: Excellent horizontal and vertical scaling, proven at large scale