#!/usr/bin/env python3
"""Test UUID generation to debug Qdrant issue"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.data_generator import DataGenerator

# Test default generation
print("Testing UUID generation:")
gen = DataGenerator(128)

# Test with default (pure UUID)
vectors1 = gen.generate_vectors(3)
print("\nDefault generation (use_pure_uuid=True):")
for vec in vectors1:
    print(f"  ID: {vec.id}")
    print(f"  Prefix in metadata: {vec.metadata.get('prefix')}")

# Test with prefix but pure UUID
vectors2 = gen.generate_vectors(3, prefix="test")
print("\nWith prefix='test' (use_pure_uuid=True):")
for vec in vectors2:
    print(f"  ID: {vec.id}")
    print(f"  Prefix in metadata: {vec.metadata.get('prefix')}")

# Test with use_pure_uuid=False
vectors3 = gen.generate_vectors(3, prefix="test", use_pure_uuid=False)
print("\nWith prefix='test' (use_pure_uuid=False):")
for vec in vectors3:
    print(f"  ID: {vec.id}")
    print(f"  Prefix in metadata: {vec.metadata.get('prefix')}")

# Check if valid UUID
import uuid
print("\nUUID validation:")
for vec in vectors1:
    try:
        uuid.UUID(vec.id)
        print(f"  {vec.id}: Valid UUID ✓")
    except ValueError:
        print(f"  {vec.id}: Invalid UUID ✗")