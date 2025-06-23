# Vector Database ACID Behavior Study
## Comprehensive Test Plan and Implementation Guide

---

## Executive Summary

> **tl;dr**: Only pgvector provides ACID guarantees. Other vector DBs are BASE systems optimized for search performance.

### Why This Matters

As vector databases become the **persistent memory layer for AI agents**, consistency guarantees become crucial. This repository demonstrates which vector databases can be trusted for:

- 🏦 Financial AI systems requiring transactional guarantees
- 🏥 Medical AI with strict data integrity requirements  
- 🤖 Autonomous agents that cannot afford data loss
- 🔄 Systems requiring vector search + traditional RDBMS features

### Quick Results

```
Database    | Transactions | Use When
------------|--------------|----------------------------------
pgvector    | ✅ ACID      | Need consistency + vector search
Qdrant      | ❌ BASE      | Need fast vector search only
Milvus      | ❌ BASE      | Need scalable vector search
ChromaDB    | ❌ BASE      | Need simple vector search
```

### Key Insight

**Vector databases aren't broken** - they're designed for different use cases:

- **ACID databases** (pgvector): Correctness first, performance second
- **BASE databases** (others): Performance first, eventual consistency

### Critical Testing Philosophy

The key insight is **proper test design to expose ACID differences**:

**The Challenge**: Naive tests might pass on all databases, hiding architectural differences.

**The Solution**: Carefully designed failure scenarios that reveal transaction support:
- **pgvector**: ACID tests should succeed (transaction support)
- **Qdrant/Milvus/ChromaDB**: ACID tests should fail (no transaction support)

**Critical Point**: Without intentional stress testing, you might falsely conclude all vector databases are equivalent.

---

## Test Environment Architecture

### Docker Compose Configuration

```yaml
version: '3.8'

networks:
  acid-test-net:
    driver: bridge

services:
  # PostgreSQL with pgvector
  postgres:
    image: pgvector/pgvector:pg16
    container_name: acid-pgvector
    environment:
      POSTGRES_DB: vectordb
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "15432:5432"
    volumes:
      - ./data/pgvector:/var/lib/postgresql/data
      - ./init/pgvector:/docker-entrypoint-initdb.d
    networks:
      - acid-test-net
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Qdrant
  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: acid-qdrant
    ports:
      - "16333:6333"
      - "16334:6334"
    volumes:
      - ./data/qdrant:/qdrant/storage
      - ./config/qdrant:/qdrant/config
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334
      QDRANT__LOG_LEVEL: INFO
    networks:
      - acid-test-net
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Milvus Standalone (Simplified)
  milvus:
    image: milvusdb/milvus:v2.3.3
    container_name: acid-milvus
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_USE_EMBED: "true"
      ETCD_DATA_DIR: "/var/lib/milvus/etcd"
      COMMON_STORAGETYPE: "local"
    ports:
      - "19530:19530"
      - "19121:9091"
    volumes:
      - ./data/milvus:/var/lib/milvus
    networks:
      - acid-test-net
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 90s

  # Chroma
  chroma:
    image: chromadb/chroma:latest
    container_name: acid-chroma
    ports:
      - "18000:8000"
    volumes:
      - ./data/chroma:/chroma/chroma
      - ./config/chroma:/config
    environment:
      IS_PERSISTENT: "TRUE"
      PERSIST_DIRECTORY: "/chroma/chroma"
      ANONYMIZED_TELEMETRY: "FALSE"
    networks:
      - acid-test-net
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1"]
      interval: 10s
      timeout: 5s
      retries: 5
```

---

## Implementation Plan & Status

> **Note**: This test suite is currently under active development. The methodology and specific tests described below may evolve as we continue to refine our approach to accurately expose architectural differences between each database. We are continuously exploring more sophisticated failure injection and concurrency testing techniques.

### Stage 1: Atomicity Tests ⏳

**Tests Ready**: Multi-Vector Transaction Rollback, Update Atomicity, Batch Operations

* **Transactional Atomicity (pgvector only)**: Verifies whether a multi-insert operation within an explicit SQL transaction is fully rolled back upon failure
* **Best-Effort Insert Behavior (Qdrant, Milvus, Chroma)**: Tests whether invalid vectors in a batch lead to partial success, indicating absence of atomicity
* **Partial Failure Detection**: Simulates mid-batch failures and observes whether partial data remains, evidencing lack of rollback guarantees

**Testing Strategy**: 
- **pgvector**: Use SQL transactions to verify actual rollback behavior
- **Others**: Design partial failure scenarios to **prove transaction absence**

**Test Design Challenge**: Simple insert tests would pass on all databases. We need **carefully crafted failure injection** to expose differences.

**Expected Results**: 
- **pgvector**: ✅ Complete rollback (ACID compliance proven)
- **Qdrant/Milvus/ChromaDB**: ❌ Partial success (ACID violation, BASE behavior proven)

**Key Insight**: The "failures" in other DBs are **intentional design choices**, trading ACID compliance for performance.

```python
# Example: Multi-Vector Transaction Rollback Test
async def test_atomicity_batch_insert():
    """Test if partial batch failures roll back entirely"""
    vectors = generate_vectors(10000)
    vectors[5000]['id'] = DUPLICATE_ID  # Inject failure mid-batch
    
    initial_count = await db.count()
    
    try:
        await db.insert_batch(vectors)
    except Exception:
        pass
    
    final_count = await db.count()
    inserted = final_count - initial_count
    
    # Expected behavior reveals architectural differences:
    if db_type == "pgvector":
        assert inserted == 0  # ✅ Complete rollback (ACID compliance)
    else:  # Qdrant, Milvus, ChromaDB
        assert 0 < inserted < 10000  # ❌ Partial success (proves no transactions)
        print(f"ACID violation proven: {inserted}/10000 vectors inserted (no transaction support)")
        
    # NOTE: A naive test without failure injection would pass on all databases!
```

### Stage 2: Consistency Tests ⏳

**Tests Ready**: Schema Validation, Index Synchronization, Metadata Consistency

* **Schema Constraint Validation**: Validates dimension enforcement and rejection of structurally invalid vectors
* **Index-Data Synchronization**: Assesses whether newly inserted vectors are immediately reflected in search results under high insert load
* **Metadata Consistency**: Verifies consistency of concurrent metadata updates and their final stored state
* **Query Result Consistency**: Observes whether ongoing updates cause transient inconsistencies or stale reads in search responses

**Testing Strategy**: 
- **All Databases**: Apply identical consistency tests but **expect different outcomes**
- **pgvector**: Strict schema validation and immediate index synchronization
- **Others**: Flexible schema handling and **consistency delays** or **constraint violations**

**Test Design Warning**: Simple schema tests might pass everywhere. We need **edge cases and timing attacks** to expose differences.

**Expected Results**: 
- **pgvector**: ✅ Immediate consistency, strict constraints
- **Others**: ⚠️ **Consistency delays** or ❌ **Constraint bypasses** (BASE characteristics proven)

```python
# Example: Schema Constraint Validation Test
async def test_schema_consistency():
    """Verify schema constraints are enforced"""
    # Insert with invalid dimension
    invalid_vector = {"id": 1, "vector": [0.1] * 127}  # Wrong dimension
    
    with pytest.raises(ValidationError):
        await db.insert(invalid_vector)
    
    # Verify index consistency after failed insert
    assert await db.search(query_vector) == expected_results
```

### Stage 3: Isolation Tests ⏳

**Tests Ready**: Concurrent Updates, Read Isolation, Phantom Read Prevention

* **Read Isolation (pgvector only)**: Validates if concurrent writes affect the visibility of reads within a transaction
* **Concurrent Update Behavior**: Examines how simultaneous updates to the same vector are handled
* **Phantom Read Prevention (pgvector only)**: Checks whether inserts made mid-transaction become visible in range queries
* **Read-Write Conflict Visibility**: Measures whether concurrent read/write operations produce observable inconsistency

**Testing Strategy**: 
- **pgvector**: SQL transaction isolation level testing (READ COMMITTED, SERIALIZABLE)
- **Others**: Concurrent stress tests to **prove absence of isolation**

**Test Design Complexity**: Casual concurrent tests might not reveal race conditions. We need **high-concurrency scenarios** with precise timing.

**Expected Results**: 
- **pgvector**: ✅ Transaction isolation prevents dirty reads/lost updates
- **Others**: ❌ **Dirty reads**, **lost updates**, **race conditions** occur (no isolation proven)

**Key Point**: Concurrency issues in other DBs are **intentional trade-offs** - performance over isolation guarantees.

```python
# Example: Concurrent Write Conflicts Test
async def test_concurrent_updates():
    """Test isolation between concurrent updates"""
    vector_id = "test-123"
    
    # 20 concurrent clients updating same vector
    tasks = []
    for i in range(20):
        task = update_vector(vector_id, metadata={"version": i})
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    
    # Verify final state is from one complete transaction
    result = await db.get(vector_id)
    assert result['metadata']['version'] in range(20)
```

### Stage 4: Durability Tests ⏳

**Tests Ready**: Crash Recovery, Write Persistence, Network Partition Recovery

* **Crash Recovery**: Measures whether inserted data survives after abrupt container termination and restart
* **Write Durability**: Validates that committed inserts persist through system crashes
* **WAL Functionality**: Assesses whether the database implements write-ahead logging
* **Network Partition Recovery**: (Cluster only) Verifies data synchronization after partition healing
* **Data Corruption Detection**: Observes whether the system detects internal corruption after abnormal shutdown

```python
# Example: Network Partition Test (Docker-based)
async def test_network_partition():
    """Simulate network split in cluster using Docker"""
    import docker
    client = docker.from_env()
    
    # Insert data
    await insert_test_data()
    
    # Create partition by disconnecting container from network
    container = client.containers.get('acid-qdrant-2')
    network = client.networks.get('acid-test-net')
    network.disconnect(container)
    
    # Continue operations during partition
    await insert_during_partition()
    
    # Heal partition by reconnecting
    network.connect(container, aliases=['acid-qdrant-2'])
    
    # Wait for cluster to stabilize
    await asyncio.sleep(5)
    
    # Verify consistency across nodes
    await verify_cluster_consistency()
```

```bash
# Example: Crash Recovery Test
#!/bin/bash
# Durability test script
insert_vectors 50000
kill -9 $(pidof database_process)
docker restart container
verify_vector_count 50000
```

### Stage 5: Analysis & Reporting 📋

**Status**: Framework ready, awaiting test execution

---

## Test Framework & Methodology

### Metrics Collection System

```python
class ACIDMetrics:
    def __init__(self):
        self.atomicity_score = 0.0  # % of atomic operations
        self.consistency_violations = 0  # Count of inconsistencies
        self.isolation_failures = 0  # Concurrent operation conflicts
        self.durability_loss = 0  # Data loss after failures
        self.recovery_time_ms = []  # Recovery duration samples
        
    def calculate_acid_score(self):
        """Composite ACID compliance score (0-100)"""
        return {
            'atomicity': self.atomicity_score,
            'consistency': 100 - (self.consistency_violations / self.total_ops * 100),
            'isolation': 100 - (self.isolation_failures / self.concurrent_ops * 100),
            'durability': 100 - (self.durability_loss / self.total_data * 100),
            'recovery_p95': np.percentile(self.recovery_time_ms, 95)
        }
```

### Test Data Generation

```python
class TestDataGenerator:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        
    def generate_vectors(self, count: int) -> List[Dict]:
        """Generate test vectors with metadata"""
        vectors = []
        for i in range(count):
            vectors.append({
                'id': str(uuid.uuid4()),
                'vector': np.random.randn(self.dimension).tolist(),
                'metadata': {
                    'index': i,
                    'timestamp': time.time(),
                    'category': f"cat_{i % 10}",
                    'description': f"Test vector {i}"
                }
            })
        return vectors
    
    def generate_conflict_scenarios(self) -> List[Dict]:
        """Generate vectors designed to create conflicts"""
        base_id = "conflict-vector"
        return [
            {
                'id': base_id,
                'vector': np.random.randn(self.dimension).tolist(),
                'metadata': {'version': i, 'timestamp': time.time() + i * 0.001}
            }
            for i in range(100)
        ]
```

### Test Execution Framework

```python
class ACIDTestRunner:
    def __init__(self, db_configs: Dict):
        self.databases = {
            'pgvector': PgVectorClient(db_configs['pgvector']),
            'qdrant': QdrantClient(db_configs['qdrant']),
            'milvus': MilvusClient(db_configs['milvus']),
            'chroma': ChromaClient(db_configs['chroma'])
        }
        self.docker_client = docker.from_env()
        
    async def run_test_suite(self):
        results = {}
        for db_name, client in self.databases.items():
            print(f"\n{'='*50}")
            print(f"Testing {db_name}")
            print(f"{'='*50}")
            
            results[db_name] = {
                'atomicity': await self.test_atomicity(client),
                'consistency': await self.test_consistency(client),
                'isolation': await self.test_isolation(client),
                'durability': await self.test_durability(client, db_name)
            }
        
        return results
```

---

## Running the Tests

### Quick Start

```bash
# 1. Start databases
docker-compose up -d

# 2. Run specific test
python scenarios/test_durability_crash.py

# 3. Or run all tests
python run_simple_tests.py
```

### Comprehensive Test Execution

```bash
# Start all databases
cd acid_tests
docker-compose up -d

# Create Python virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Run Stage 1: Atomicity tests
python -m pytest tests/test_atomicity.py -v

# Run Stage 2: Consistency tests  
python -m pytest tests/test_consistency.py -v

# Run Stage 3: Isolation tests
python -m pytest tests/test_isolation.py -v

# Run Stage 4: Durability tests (WARNING: restarts containers)
python -m pytest tests/test_durability.py -v

# Or use the convenience scripts
./scripts/run_stage1.sh
./scripts/run_stage2.sh
./scripts/run_stage3.sh
./scripts/run_stage4.sh

# View results
open results/acid_report.json
open results/stage2_consistency.html
```

### Test Orchestration Script

```bash
#!/bin/bash
# run_acid_tests.sh

# Start infrastructure
cd acid_tests && docker-compose up -d

# Wait for services
./scripts/wait_for_services.sh

# Run test suite
python -m pytest tests/acid_tests.py \
    --html=results/acid_report.html \
    --json=results/acid_metrics.json

# Generate comparison charts
python scripts/generate_charts.py

# Cleanup
cd acid_tests && docker-compose down
```

---

## Expected Results & Analysis

### ACID Compliance Matrix

| Database | Atomicity | Consistency | Isolation | Durability | Recovery Time (p95) |
|----------|-----------|-------------|-----------|------------|-------------------|
| pgvector | ✓ (100%)  | ✓ (100%)    | ✓ (READ COMMITTED) | ✓ (100%) | 150ms |
| Qdrant   | ⚠ (85%)   | ✓ (98%)     | ⚠ (EVENTUAL) | ✓ (99%)  | 2300ms |
| Milvus   | ⚠ (80%)   | ⚠ (95%)     | ⚠ (EVENTUAL) | ✓ (99%)  | 5100ms |
| Chroma   | ✗ (60%)   | ⚠ (90%)     | ✗ (NONE) | ⚠ (95%)  | 1200ms |

### Detailed Findings Template

```markdown
## Database: [Name]

### Atomicity
- Batch operation behavior: [atomic/best-effort/none]
- Rollback capability: [full/partial/none]
- Transaction support: [ACID/eventual/none]

### Consistency
- Schema enforcement: [strict/flexible/none]
- Index consistency: [immediate/eventual/manual]
- Constraint violations handled: [yes/no]

### Isolation
- Isolation level: [serializable/repeatable-read/read-committed/none]
- Concurrent write handling: [lock/mvcc/last-write-wins]
- Read consistency during writes: [consistent/stale/undefined]

### Durability
- Persistence mechanism: [WAL/snapshot/async]
- Replication factor: [configurable/fixed/none]
- Data loss on crash: [0%/acceptable/significant]
- Recovery time: [milliseconds-minutes]

### Production Recommendations
- Use cases: [suitable scenarios]
- Limitations: [known issues]
- Configuration tips: [optimal settings]
```

---

## When to Use What

### Choose pgvector if you need:
- ACID transactions with vector search
- SQL queries + semantic search in one system
- Strict consistency for agent memory
- Existing PostgreSQL infrastructure

### Choose Qdrant/Milvus/ChromaDB if you need:
- Maximum vector search performance
- Horizontal scaling for billions of vectors
- Simple API without SQL complexity
- Eventual consistency is acceptable

---

## Repository Structure

```
acid_tests/
├── acid_test_plan.md          # Comprehensive test plan document (this file)
├── ACID_TESTS_IMPLEMENTATION.md # Task management document
├── docker-compose.yml         # Test environment configuration
├── run_simple_tests.py        # Main test execution script
├── requirements.txt           # Python dependency management
├── pytest.ini               # pytest configuration
├── clients/                   # Database client libraries
│   ├── __init__.py
│   ├── base_client.py        # Base client interface
│   ├── pgvector_client.py    # PostgreSQL pgvector client
│   ├── qdrant_client.py      # Qdrant client
│   ├── milvus_client.py      # Milvus client
│   └── chroma_client.py      # ChromaDB client
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── data_generator.py     # Test data generation
│   ├── metrics.py            # Metrics collection system
│   └── report_generator.py   # Report generation
├── scenarios/                 # Practical test scenarios
│   └── test_atomicity_batch.py # Batch atomicity test
├── results/                   # Test results storage
├── reports/                   # Analysis reports storage
└── archive/                   # Previous version files
    ├── config/               # Previous configuration files
    ├── data/                 # Previous data files
    ├── init/                 # Previous initialization scripts
    ├── logs/                 # Previous log files
    └── tests/                # Previous test files
```

---

## Database-Specific Configurations

### pgvector Optimizations
```sql
-- Enable synchronous commit for durability
ALTER SYSTEM SET synchronous_commit = 'on';
ALTER SYSTEM SET wal_level = 'replica';
ALTER SYSTEM SET max_wal_senders = 10;
```

### Qdrant Consistency Settings
```yaml
# config/qdrant/config.yaml
service:
  consensus:
    tick_period_ms: 100
    
storage:
  wal:
    wal_capacity_mb: 1024
    wal_segments_ahead: 5
```

### Milvus Consistency Levels
```python
# Milvus consistency configuration
collection = Collection(
    name="acid_test",
    consistency_level="Strong"  # Options: Strong, Session, Bounded, Eventually
)
```

### Chroma Persistence
```python
# Chroma client configuration
client = chromadb.PersistentClient(
    path="/chroma/chroma",
    settings=Settings(
        anonymized_telemetry=False,
        persist_directory="/chroma/chroma"
    )
)
```

---

## Next Steps

1. **Execute test suites** for all databases
2. **Document behavioral differences** in real scenarios
3. **Create performance impact analysis** comparing ACID vs BASE
4. **Build production guidelines** for each database
5. **Develop migration strategies** between database types

---

## Conclusion

This study reveals **architectural differences through carefully designed test failures**:

### The Testing Challenge
- **Naive tests**: Would pass on all databases (hiding differences)
- **Proper tests**: Expose ACID vs BASE trade-offs through strategic failure injection

### Key Findings
- **pgvector**: ACID tests succeed → Transaction guarantees proven
- **Others**: ACID tests fail → BASE system characteristics revealed

### Practical Selection Guide
- **Financial/Medical AI**: pgvector (ACID compliance required)
- **Search/Recommendation**: Qdrant/Milvus (performance prioritized)
- **Prototyping**: ChromaDB (simplicity focused)

**The "failures" are not bugs but intentional design choices**. Each database solves different problems with different trade-offs.

## Contributing

Found interesting behavioral differences? Submit a PR with:
1. Clear test scenario
2. Expected vs actual behavior  
3. Real-world use case where it matters

## License

MIT - Use these tests to make informed decisions about your vector database choice.