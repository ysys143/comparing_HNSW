# pgvector HNSW Implementation Analysis

## Overview

pgvector implements HNSW as a PostgreSQL index access method, deeply integrated with PostgreSQL's storage, concurrency, and recovery mechanisms. The implementation features a two-phase build process and parallel construction capabilities.

## Graph Construction

### Two-Phase Build Process

pgvector uses a unique two-phase approach to handle large indices efficiently:

```c
// Phase 1: In-memory construction
typedef struct HnswBuildState {
    Relation heap;
    Relation index;
    HnswGraph *graph;
    int64 memoryUsed;
    bool heap;  // true for phase 1
} HnswBuildState;

// Phase 2: Write to disk
static void
WriteTuplesInOrder(HnswBuildState *buildstate) {
    // Flush in-memory graph to disk pages
    // Create page structure
    // Update metadata
}
```

**Benefits**:
- Uses `maintenance_work_mem` efficiently
- Handles datasets larger than available memory
- Optimizes disk write patterns

### Node Insertion Algorithm

```c
static void
HnswInsertTupleOnDisk(HnswBuildState *buildstate, 
                      ItemPointer heaptid, 
                      Vector *vector, 
                      int level) {
    // Find nearest neighbors at each layer
    for (int lc = level; lc >= 0; lc--) {
        List *candidates = HnswSearchLayer(buildstate, vector, 
                                         entryPoint, lc, m);
        
        // Add bidirectional links
        foreach(lc, candidates) {
            HnswAddConnection(buildstate, tid, neighbor, lc);
            HnswAddConnection(buildstate, neighbor, tid, lc);
        }
        
        // Prune connections if needed
        if (list_length(neighbors) > m) {
            neighbors = HnswPruneConnections(neighbors, m);
        }
    }
}
```

### Layer Assignment

```c
static int
HnswGetLayerLevel(HnswBuildState *buildstate) {
    // Standard exponential decay
    float ml = -log(RandomDouble()) * buildstate->ml;
    return Min((int) ml, buildstate->maxLevel);
}
```

### Parallel Build Support

```c
typedef struct HnswSharedState {
    int nworkers;
    ConditionVariable cv;
    LWLock *lock;
    double indtuples;
    BlockNumber pagesWritten;
} HnswSharedState;

// Workers coordinate through shared memory
static void
HnswParallelBuildWorker(HnswBuildState *buildstate) {
    while ((tuple = GetNextWorkItem()) != NULL) {
        HnswInsertTuple(buildstate, tuple);
        UpdateSharedProgress(buildstate->shared);
    }
}
```

## Search Algorithm

### Entry Point Management

```c
// Stored in metapage
typedef struct HnswMetaPageData {
    uint64 entryBlkno;    // Entry point block number
    uint64 entryOffno;    // Entry point offset
    int16 entryLevel;     // Entry point level
} HnswMetaPageData;
```

### Search Implementation

```c
static List *
HnswSearchLayer(HnswScanState *scanstate, 
                Vector *query, 
                List *ep, 
                int layer, 
                int ef) {
    List *visited = NIL;
    List *candidates = list_copy(ep);
    List *w = list_copy(ep);
    
    while (list_length(candidates) > 0) {
        HnswCandidate *lc = GetClosestCandidate(candidates);
        
        if (lc->distance > GetFurthestDistance(w))
            break;
            
        // Check neighbors
        List *neighbors = HnswGetNeighbors(lc->tid, layer);
        foreach(neighbor, neighbors) {
            if (!list_member(visited, neighbor)) {
                float distance = GetDistance(query, neighbor);
                
                if (distance < GetFurthestDistance(w) || 
                    list_length(w) < ef) {
                    candidates = lappend(candidates, neighbor);
                    w = AddToHeap(w, neighbor, distance, ef);
                }
                
                visited = lappend(visited, neighbor);
            }
        }
    }
    
    return w;
}
```

### Scan State Management

```c
typedef struct HnswScanState {
    Relation index;
    Vector *query;
    int ef;
    bool first;
    List *candidates;
    int candidateIndex;
} HnswScanState;

// Incremental result retrieval
static bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir) {
    HnswScanState *scanstate = (HnswScanState *) scan->opaque;
    
    if (scanstate->first) {
        // Perform search on first call
        scanstate->candidates = HnswSearch(scanstate);
        scanstate->first = false;
    }
    
    // Return next result
    if (scanstate->candidateIndex < list_length(scanstate->candidates)) {
        scan->xs_heaptid = GetNextCandidate(scanstate);
        return true;
    }
    
    return false;
}
```

## Memory Management

### Page Structure

```c
typedef struct HnswElementData {
    ItemPointerData heaptids[HNSW_HEAPTIDS];
    uint8 level;
    uint8 deleted;
    uint16 unused;
    HnswNeighborArray neighbors[FLEXIBLE_ARRAY_MEMBER];
} HnswElementData;

typedef struct HnswNeighborArray {
    int count;
    ItemPointerData tids[FLEXIBLE_ARRAY_MEMBER];
} HnswNeighborArray;
```

### Buffer Management Integration

```c
// All page access goes through PostgreSQL buffer manager
static HnswElement
HnswGetElement(Relation index, BlockNumber blkno, OffsetNumber offno) {
    Buffer buffer = ReadBuffer(index, blkno);
    LockBuffer(buffer, BUFFER_LOCK_SHARE);
    Page page = BufferGetPage(buffer);
    HnswElement element = PageGetItem(page, 
                                    PageGetItemId(page, offno));
    UnlockReleaseBuffer(buffer);
    return element;
}
```

## Concurrency Control

### Lock Types

```c
// Page-level locking
LockBuffer(buffer, BUFFER_LOCK_SHARE);      // Read access
LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);  // Write access

// Index-level coordination
LWLockAcquire(&buildstate->lock, LW_EXCLUSIVE);
// ... critical section ...
LWLockRelease(&buildstate->lock);
```

### MVCC Compliance

- No in-place updates to maintain consistency
- New versions created for modifications
- Deleted elements marked with tombstones
- VACUUM cleans up old versions

## WAL Integration

```c
typedef struct HnswInsertWALData {
    BlockNumber blkno;
    OffsetNumber offno;
    uint16 m;
    uint16 ef;
    ItemPointerData heaptid;
    uint8 level;
    uint16 neighborCount;
    /* neighbors follow */
} HnswInsertWALData;

static void
HnswXLogInsert(Buffer buffer, HnswElement element) {
    XLogBeginInsert();
    XLogRegisterData((char *) &walData, sizeof(HnswInsertWALData));
    XLogRegisterBuffer(0, buffer, REGBUF_STANDARD);
    recptr = XLogInsert(RM_GENERIC_ID, 0);
    PageSetLSN(page, recptr);
}
```

## Performance Optimizations

### 1. Two-Phase Build
- Optimizes memory usage
- Reduces random I/O
- Enables parallel construction

### 2. Batch Processing
```c
#define HNSW_INSERT_BATCH_SIZE 1000
// Process multiple tuples before flushing
```

### 3. Early Termination
```c
// Skip processing if distance exceeds threshold
if (distance > furthest_distance && have_enough_candidates)
    break;
```

### 4. Efficient Neighbor Storage
- Compact representation on disk
- Cache-friendly access patterns
- Minimized pointer chasing

## Configuration Parameters

### Build Parameters
```sql
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops)
WITH (
    m = 16,                  -- Max connections per layer
    ef_construction = 64     -- Search width during construction
);
```

### Runtime Parameters
```sql
SET hnsw.ef_search = 40;     -- Search width for queries
```

### GUC Integration
```c
DefineCustomIntVariable("hnsw.ef_search",
                       "Sets the size of the dynamic candidate list",
                       NULL,
                       &hnsw_ef_search,
                       40, 1, 1000,
                       PGC_USERSET,
                       0,
                       NULL, NULL, NULL);
```

## Unique Features

### 1. PostgreSQL Integration
- Leverages buffer cache
- WAL for durability
- VACUUM for maintenance
- Cost-based optimization

### 2. Two-Phase Construction
- Handles large datasets efficiently
- Optimizes I/O patterns
- Supports parallel building

### 3. Standard SQL Interface
```sql
-- Simple to use
SELECT * FROM items 
ORDER BY embedding <-> '[1,2,3]' 
LIMIT 10;
```

### 4. Transaction Support
- ACID compliance
- Concurrent reads during writes
- Crash recovery

## Limitations and Trade-offs

### 1. Page-Based Storage
- Fixed page size can lead to fragmentation
- May require more I/O than custom storage

### 2. Generic Buffer Management
- Not optimized specifically for graph traversal
- Cache eviction may impact performance

### 3. SQL Overhead
- Query parsing and planning overhead
- Less efficient than direct API calls

## Summary

pgvector's HNSW implementation successfully adapts the algorithm to PostgreSQL's architecture while maintaining good performance. The two-phase build process and deep integration with PostgreSQL's infrastructure provide reliability and ease of use, making it an excellent choice for applications already using PostgreSQL.