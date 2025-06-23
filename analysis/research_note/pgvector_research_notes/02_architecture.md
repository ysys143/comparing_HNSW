# pgvector Architecture Analysis

## Overview

pgvector is a PostgreSQL extension that adds vector similarity search capabilities through deep integration with PostgreSQL's internal systems. It implements HNSW and IVFFlat algorithms while leveraging PostgreSQL's existing infrastructure for storage, concurrency, and durability.

## System Architecture

### PostgreSQL Extension Architecture

```
┌─────────────────────────────────────────────┐
│         PostgreSQL Client                   │
│         (SQL Interface)                     │
├─────────────────────────────────────────────┤
│      PostgreSQL Query Processor             │
│    (Parser, Planner, Executor)              │
├─────────────────────────────────────────────┤
│         pgvector Extension                  │
│  ┌─────────────────────────────────────┐   │
│  │     Vector Type System               │   │
│  │  (vector, halfvec, sparsevec, bit)   │   │
│  ├─────────────────────────────────────┤   │
│  │      Index Access Methods            │   │
│  │      (HNSW, IVFFlat)                 │   │
│  ├─────────────────────────────────────┤   │
│  │     Distance Functions               │   │
│  │  (L2, Cosine, IP, L1, Hamming)      │   │
│  └─────────────────────────────────────┘   │
├─────────────────────────────────────────────┤
│      PostgreSQL Storage Manager             │
│    (Buffer Cache, WAL, Checkpoints)         │
└─────────────────────────────────────────────┘
```

## Core Components

### 1. Vector Type System

**Type Definitions**:
```c
typedef struct Vector {
    int32 vl_len_;      /* varlena header */
    int16 dim;          /* number of dimensions */
    int16 unused;
    float x[FLEXIBLE_ARRAY_MEMBER];
} Vector;

typedef struct HalfVector {
    int32 vl_len_;
    int16 dim;
    int16 unused;
    half x[FLEXIBLE_ARRAY_MEMBER];  /* float16 elements */
} HalfVector;

typedef struct SparseVector {
    int32 vl_len_;
    int32 dim;          /* total dimensions */
    int32 nnz;          /* non-zero elements */
    int32 indices[FLEXIBLE_ARRAY_MEMBER];
    /* followed by values */
} SparseVector;
```

**Type Operations**:
- Input/output functions
- Binary send/receive
- Type casting
- Arithmetic operations

### 2. HNSW Implementation

**Page-Based Graph Storage**:
```c
typedef struct HnswMetaPageData {
    uint16 magicNumber;
    uint16 version;
    uint16 dimensions;
    uint16 m;
    uint16 efConstruction;
    uint64 entryPoint;
} HnswMetaPageData;

typedef struct HnswElementData {
    ItemPointerData heaptids[HNSW_HEAPTIDS];
    uint8 level;
    uint8 deleted;
    uint16 unused;
    HnswNeighborArray neighbors[FLEXIBLE_ARRAY_MEMBER];
} HnswElementData;
```

**Two-Phase Build Process**:
```c
// Phase 1: In-memory construction
static void
InitBuildState(HnswBuildState *buildstate, Relation heap, Relation index) {
    buildstate->heap = heap;
    buildstate->index = index;
    buildstate->graph = HnswInitGraph(buildstate->m, buildstate->efConstruction);
    // Build in memory using maintenance_work_mem
}

// Phase 2: Write to disk
static void
FlushPages(HnswBuildState *buildstate) {
    // Write graph pages to disk
    // Update metadata
    // Create entry point
}
```

### 3. IVFFlat Implementation

**Clustering Structure**:
```c
typedef struct IvfflatList {
    BlockNumber startPage;
    BlockNumber insertPage;
    uint16 ncenters;
    float center[FLEXIBLE_ARRAY_MEMBER];
} IvfflatList;

typedef struct IvfflatMetaPageData {
    uint32 magicNumber;
    uint32 version;
    uint16 dimensions;
    uint16 lists;
} IvfflatMetaPageData;
```

**K-means Algorithm**:
- Elkan's algorithm for efficiency
- Parallel clustering support
- Adaptive sampling for large datasets

### 4. Distance Functions

**SIMD Optimizations**:
```c
// Platform detection and dispatching
#ifdef __x86_64__
    if (pg_popcount_available())
        return vector_l2_distance_avx(a, b, dim);
#endif
#ifdef __aarch64__
    return vector_l2_distance_neon(a, b, dim);
#endif
    return vector_l2_distance_default(a, b, dim);
```

**Supported Metrics**:
- L2 distance (`<->`)
- Inner product (`<#>`)
- Cosine distance (`<=>`)
- L1 distance (`<+>`)
- Hamming distance (binary vectors)
- Jaccard distance (sparse vectors)

## PostgreSQL Integration

### 1. Access Method Interface

```c
// Index access method definition
IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);
amroutine->amstrategies = 5;  /* number of operators */
amroutine->amsupport = 3;     /* number of support functions */
amroutine->amcanorder = false;
amroutine->amcanunique = false;
amroutine->amcanmulticol = false;
amroutine->amoptionalkey = true;
amroutine->amindexnulls = false;
amroutine->amsearcharray = false;
amroutine->amsearchnulls = false;
amroutine->amstorage = false;
amroutine->amclusterable = false;
amroutine->ampredlocks = false;
amroutine->amcanparallel = true;
amroutine->amcaninclude = false;
amroutine->amkeytype = InvalidOid;
amroutine->ambuild = hnswbuild;
amroutine->ambuildempty = hnswbuildempty;
amroutine->aminsert = hnswinsert;
amroutine->ambulkdelete = hnswbulkdelete;
amroutine->amvacuumcleanup = hnswvacuumcleanup;
amroutine->amcanreturn = NULL;
amroutine->amcostestimate = hnswcostestimate;
amroutine->amoptions = hnswoptions;
amroutine->amproperty = hnswproperty;
amroutine->ambuildphasename = hnswbuildphasename;
amroutine->amvalidate = hnswvalidate;
amroutine->ambeginscan = hnswbeginscan;
amroutine->amrescan = hnswrescan;
amroutine->amgettuple = hnswgettuple;
amroutine->amgetbitmap = NULL;
amroutine->amendscan = hnswendscan;
amroutine->ammarkpos = NULL;
amroutine->amrestrpos = NULL;
amroutine->amestimateparallelscan = NULL;
amroutine->aminitparallelscan = NULL;
amroutine->amparallelrescan = NULL;
```

### 2. Buffer Management

```c
// Page access with PostgreSQL buffer manager
Buffer buffer = ReadBuffer(index, blkno);
LockBuffer(buffer, BUFFER_LOCK_SHARE);
Page page = BufferGetPage(buffer);

// Critical section for modifications
START_CRIT_SECTION();
MarkBufferDirty(buffer);
// ... modifications ...
END_CRIT_SECTION();

UnlockReleaseBuffer(buffer);
```

### 3. WAL Integration

```c
typedef struct HnswInsertWALData {
    uint16 m;
    uint16 ef;
    ItemPointerData heaptid;
    uint8 level;
    /* neighbors data follows */
} HnswInsertWALData;

// Write-ahead logging for crash recovery
XLogBeginInsert();
XLogRegisterData((char *) &walData, sizeof(HnswInsertWALData));
XLogRegisterBuffer(0, buffer, REGBUF_STANDARD);
recptr = XLogInsert(RM_GENERIC_ID, 0);
PageSetLSN(page, recptr);
```

### 4. Query Planning

```c
// Cost estimation for query planner
static void
hnswcostestimate(PlannerInfo *root, IndexPath *path, double loop_count,
                Cost *indexStartupCost, Cost *indexTotalCost,
                Selectivity *indexSelectivity, double *indexCorrelation,
                double *indexPages)
{
    // Estimate based on ef parameter and index size
    *indexStartupCost = 0;
    *indexTotalCost = path->path.rows * cpu_operator_cost * ef;
    *indexSelectivity = 1.0;
    *indexCorrelation = 0.0;
    *indexPages = index_pages;
}
```

## Concurrency Model

### 1. Lock Types

```c
// Different lock modes for different operations
LockBuffer(buffer, BUFFER_LOCK_SHARE);      // Read operations
LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);  // Write operations

// Page-level locking
LWLockAcquire(HnswLock, LW_EXCLUSIVE);
// ... critical section ...
LWLockRelease(HnswLock);
```

### 2. MVCC Compliance

- No in-place updates
- New versions for modifications
- Visibility checks for concurrent access
- VACUUM integration for cleanup

### 3. Parallel Operations

```c
// Parallel index build support
typedef struct HnswParallelBuildState {
    /* Shared state for parallel workers */
    int nworkers;
    double indtuples;
    ConditionVariable cv;
    int nparticipants;
    /* ... */
} HnswParallelBuildState;
```

## Memory Management

### 1. Memory Contexts

```c
// PostgreSQL memory context hierarchy
MemoryContext buildCtx = AllocSetContextCreate(CurrentMemoryContext,
                                              "Hnsw build context",
                                              ALLOCSET_DEFAULT_SIZES);
MemoryContext oldCtx = MemoryContextSwitchTo(buildCtx);

// Allocations in this context
void *data = palloc(size);

// Cleanup
MemoryContextSwitchTo(oldCtx);
MemoryContextDelete(buildCtx);
```

### 2. Work Memory Management

```c
// Respect work_mem and maintenance_work_mem
if (buildstate->memoryUsed > maintenance_work_mem * 1024L) {
    FlushPages(buildstate);
    ResetBuildState(buildstate);
}
```

## Performance Optimizations

### 1. Batch Processing

```c
// Batch insert for index build
#define HNSW_INSERT_BATCH_SIZE 1000
for (i = 0; i < HNSW_INSERT_BATCH_SIZE && i < ntuples; i++) {
    HnswInsertTuple(buildstate, &tuples[i]);
}
```

### 2. CPU Cache Optimization

```c
// Cache-friendly data layout
typedef struct HnswNeighborArray {
    int count;
    int neighbors[HNSW_MAX_M * 2];
} HnswNeighborArray;
```

### 3. Vectorization

```c
// Compiler hints for vectorization
#pragma omp simd
for (i = 0; i < dim; i++) {
    distance += (a[i] - b[i]) * (a[i] - b[i]);
}
```

## Configuration

### 1. Index Parameters

```sql
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

-- Runtime configuration
SET hnsw.ef_search = 40;
SET ivfflat.probes = 10;
```

### 2. GUC Variables

```c
// Define custom GUC variables
DefineCustomIntVariable("hnsw.ef_search",
                       "Sets the size of the dynamic candidate list",
                       NULL,
                       &hnsw_ef_search,
                       40, 1, 1000,
                       PGC_USERSET,
                       0,
                       NULL, NULL, NULL);
```

## Error Handling

### PostgreSQL Error Reporting

```c
if (vector_dim > HNSW_MAX_DIM)
    ereport(ERROR,
            (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
             errmsg("vector dimension %d exceeds maximum %d",
                    vector_dim, HNSW_MAX_DIM)));
```

## Extension Lifecycle

### 1. Installation

```sql
CREATE EXTENSION vector;
```

### 2. Upgrade

```sql
ALTER EXTENSION vector UPDATE TO '0.8.0';
```

### 3. Uninstall

```sql
DROP EXTENSION vector CASCADE;
```

## Summary

pgvector's architecture demonstrates:
1. **Deep Integration**: Leverages PostgreSQL's infrastructure
2. **Reliability**: WAL support, MVCC compliance, crash recovery
3. **Performance**: SIMD optimizations, parallel builds, efficient memory use
4. **Compatibility**: Works with existing PostgreSQL features
5. **Simplicity**: Standard SQL interface, familiar operational model

The extension model allows pgvector to provide vector search capabilities while maintaining all the benefits of PostgreSQL's mature database infrastructure.