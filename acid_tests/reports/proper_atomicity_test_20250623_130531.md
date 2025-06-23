# PROPER_ATOMICITY_TEST Test Report
Generated: 2025-06-23 13:05:31

## Summary

| Database | Atomicity | Behavior | Details |
|----------|-----------|----------|----------|

## Detailed Results

### pgvector

### qdrant

### milvus

### chroma


## Conclusions

### Observed Behaviors

### Key Findings

1. **Atomicity != Error Handling**: Some databases show 'atomic' behavior by rejecting entire batches due to validation errors, not true transactional atomicity
2. **Milvus Special Case**: Silently handles duplicates without errors, making it appear non-atomic when it's actually a design choice
3. **True ACID**: Only pgvector demonstrated actual transaction support with rollback capability

### Technical Characteristics Based on Test Results

**Transaction Support**:
- pgvector: Explicit SQL transactions with rollback capability
- Others: No transaction support (BASE model)

**Duplicate ID Handling**:
- Milvus: Silently deduplicates (first-write-wins)
- Qdrant/ChromaDB: Rejects entire batch on duplicate detection
- pgvector: Depends on constraint configuration

**Batch Validation Behavior**:
- ChromaDB/Milvus: Strict dimension validation (all-or-nothing)
- Qdrant: Partial insertion possible before validation error
- pgvector: Depends on transaction boundaries

**ID Format Requirements**:
- pgvector/Qdrant: Strict UUID format required
- Milvus/ChromaDB: Flexible string ID format
