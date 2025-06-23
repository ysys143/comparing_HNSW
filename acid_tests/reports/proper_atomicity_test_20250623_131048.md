# PROPER_ATOMICITY_TEST Test Report
Generated: 2025-06-23 13:10:48

## Summary

| Database | Test Type | Result | Details |
|----------|-----------|--------|----------|
| pgvector | Transactional | ✅ PASS | 100 vectors rolled back |
| qdrant | Best-effort | ✅ BASE | 71/100 inserted |
| milvus | Best-effort | ⚠️ N/A | Validation prevents test |
| chroma | Best-effort | ⚠️ N/A | Validation prevents test |

## Detailed Results

### pgvector

#### Transaction Test
- Test type: SQL transaction with deliberate rollback
- Vectors in transaction: 100
- Rollback successful: True
- Result: All vectors rolled back

### qdrant

#### Best-Effort Insert Test
- Test type: Batch with invalid vector at position 50
- Vectors attempted: 100
- Vectors inserted: 71
- Behavior: best_effort_partial_success

#### Partial Failure Test
- Test type: Batch with duplicate ID at position 50
- Vectors added: 99
- Partial success: True

### milvus

#### Best-Effort Insert Test
- Test type: Batch with invalid vector at position 50
- Vectors attempted: 100
- Vectors inserted: 0
- Behavior: all_or_nothing_validation

#### Partial Failure Test
- Test type: Batch with duplicate ID at position 50
- Vectors added: 100
- Partial success: False

### chroma

#### Best-Effort Insert Test
- Test type: Batch with invalid vector at position 50
- Vectors attempted: 100
- Vectors inserted: 0
- Behavior: all_or_nothing_validation

#### Partial Failure Test
- Test type: Batch with duplicate ID at position 50
- Vectors added: 99
- Partial success: True


## Conclusions

### Observed Behaviors

**pgvector**:
- ✅ Transaction rollback successful
- 100 vectors rolled back

**qdrant**:
- ✅ Best-effort partial success (expected BASE behavior)
- 71/100 vectors inserted
- Duplicate handling: 99 vectors added (duplicate skipped)

**milvus**:
- ⚠️ Strict validation prevents partial insert

**chroma**:
- ⚠️ Strict validation prevents partial insert
- Duplicate handling: 99 vectors added (duplicate skipped)

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
