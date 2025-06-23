import pytest
import asyncio
import os
import sys
from pathlib import Path
from utils.metrics import MetricsCollector

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def docker_compose_file():
    """Path to docker-compose file"""
    return project_root / "docker-compose.yml"


@pytest.fixture(scope="session")
def docker_compose_project_name():
    """Docker compose project name"""
    return "acid_tests"


@pytest.fixture(scope="session")
def metrics_collector():
    """Create metrics collector for the test session"""
    collector = MetricsCollector(output_dir="results")
    yield collector
    # Save report at the end of session
    collector.save_report()
    collector.print_summary()


# Test configuration
TEST_CONFIG = {
    'vector_dimension': 384,
    'test_timeout': 300,  # 5 minutes
    'databases': {
        'pgvector': {
            'host': os.getenv('PGVECTOR_HOST', '172.30.0.10'),
            'port': int(os.getenv('PGVECTOR_PORT', '5432')),
            'user': 'postgres',
            'password': 'postgres',
            'database': 'vectordb'
        },
        'qdrant': {
            'host': os.getenv('QDRANT_HOST', '172.30.0.20'),
            'port': int(os.getenv('QDRANT_PORT', '6333')),
            'grpc_port': int(os.getenv('QDRANT_GRPC_PORT', '6334'))
        },
        'milvus': {
            'host': os.getenv('MILVUS_HOST', '172.30.0.30'),
            'port': int(os.getenv('MILVUS_PORT', '19530'))
        },
        'chroma': {
            'host': os.getenv('CHROMA_HOST', '172.30.0.40'),
            'port': int(os.getenv('CHROMA_PORT', '8000')),
            'auth_token': 'test-token'
        }
    }
}


@pytest.fixture
def test_config():
    """Get test configuration"""
    return TEST_CONFIG


# Markers for different test types
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "atomicity: Test atomicity properties"
    )
    config.addinivalue_line(
        "markers", "consistency: Test consistency properties"
    )
    config.addinivalue_line(
        "markers", "isolation: Test isolation properties"
    )
    config.addinivalue_line(
        "markers", "durability: Test durability properties"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "requires_docker: Tests that require Docker"
    )