# pgvector Filtering Strategies Analysis

## Overview
pgvector integrates filtering through PostgreSQL's powerful query planner and execution engine, combining vector operations with traditional SQL predicates.

## Filtering Approach

### 1. **SQL-Based Filtering**
- Leverages PostgreSQL's mature query optimization
- Filters are standard WHERE clauses
- Can combine with any PostgreSQL features (indexes, joins, CTEs)

### 2. **Query Examples**
```sql
-- Basic filtering with vector search
SELECT id, embedding <-> '[1,2,3]'::vector AS distance
FROM items
WHERE category = 'science' AND price < 100
ORDER BY embedding <-> '[1,2,3]'::vector
LIMIT 10;

-- Using partial index for filtering
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)
WHERE status = 'active';
```

### 3. **Implementation Details**

#### Query Execution Plans
pgvector can use different strategies depending on the query:

1. **Index Scan with Filter**
```sql
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM items 
WHERE category = 'tech' 
ORDER BY embedding <-> query_vec 
LIMIT 10;

-- Execution plan:
Limit
  -> Index Scan using items_embedding_idx on items
       Order By: (embedding <-> query_vec)
       Filter: (category = 'tech')
```

2. **Bitmap Heap Scan Combination**
```sql
-- When filter is selective
Limit
  -> Sort
       -> Bitmap Heap Scan on items
            -> BitmapAnd
                 -> Bitmap Index Scan on category_idx
                 -> Bitmap Index Scan on price_idx
```

### 4. **Filtering Strategies**

#### Pre-filtering with B-tree Indexes
```sql
-- Create supporting indexes
CREATE INDEX ON items (category);
CREATE INDEX ON items (price);
CREATE INDEX ON items (created_at);

-- Compound index for common filter combinations
CREATE INDEX ON items (category, status) WHERE deleted = false;
```

#### Partial HNSW Indexes
```sql
-- Build HNSW only for frequently accessed subset
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops)
WHERE status = 'published' AND category IN ('tech', 'science');
```

#### Filtered CTEs
```sql
-- Pre-filter in CTE
WITH filtered_items AS (
    SELECT * FROM items 
    WHERE category = 'tech' 
    AND created_at > '2023-01-01'
)
SELECT id, embedding <-> query_vec AS distance
FROM filtered_items
ORDER BY distance
LIMIT 10;
```

### 5. **Advanced Techniques**

#### Multi-Column Indexes
```sql
-- GIN index for JSONB + vector search
CREATE INDEX ON items USING gin (metadata);

SELECT * FROM items
WHERE metadata @> '{"lang": "en", "type": "article"}'
ORDER BY embedding <-> query_vec
LIMIT 10;
```

#### Join-Based Filtering
```sql
-- Complex filtering with joins
SELECT i.id, i.embedding <-> query_vec AS distance
FROM items i
JOIN categories c ON i.category_id = c.id
JOIN users u ON i.user_id = u.id
WHERE c.name = 'technology' 
  AND u.country = 'US'
  AND i.created_at > CURRENT_DATE - INTERVAL '30 days'
ORDER BY distance
LIMIT 10;
```

### 6. **Performance Optimizations**

#### Query Planner Hints
```sql
-- Force specific plan
SET enable_seqscan = off;
SET enable_indexscan = on;

-- Adjust planner statistics
ALTER TABLE items SET (autovacuum_analyze_scale_factor = 0.02);
```

#### Work Memory Tuning
```sql
-- Increase work memory for sorting
SET work_mem = '256MB';

-- For session or configuration
ALTER DATABASE mydb SET work_mem = '256MB';
```

### 7. **Performance Characteristics**

#### Advantages
- Leverages PostgreSQL's mature optimizer
- Can combine multiple index types efficiently
- Excellent for complex queries with joins
- Statistics-based query planning

#### Trade-offs
- HNSW traversal doesn't early-terminate on filters
- May scan more nodes than necessary
- Performance depends on filter selectivity

#### Optimization Strategies
1. **Selective Filters First**: Place most selective conditions first
2. **Partial Indexes**: Create HNSW indexes on filtered subsets
3. **Index Combination**: Use bitmap index scans for multiple conditions
4. **CTEs for Complex Logic**: Break down complex queries

### 8. **Monitoring and Analysis**

#### Query Analysis
```sql
-- Analyze filter effectiveness
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT * FROM items
WHERE category = 'tech'
ORDER BY embedding <-> query_vec
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes
WHERE tablename = 'items';
```

## Code References

### C Implementation
- `src/hnsw.c` - HNSW index scan with filter checks
- `src/ivfflat.c` - IVFFlat index with filtering

### Key Functions
```c
// hnsw.c - Filter checking during scan
static bool
CheckIndexFilter(HnswScanOpaque so, ItemPointer heaptid)
{
    // Checks additional quals during index scan
    return DatumGetBool(ExecQual(so->filter, so->econtext));
}
```

## Comparison Notes
- More flexible than purpose-built vector databases for complex queries
- Benefits from PostgreSQL's extensive optimization capabilities
- Better for applications that need rich relational queries alongside vector search
- Trade-off: Less optimized for pure vector workloads compared to specialized systems