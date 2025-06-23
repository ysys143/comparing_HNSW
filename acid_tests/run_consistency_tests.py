#!/usr/bin/env python3
"""
Run all Stage 2: Consistency Tests
"""
import asyncio
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# Set environment variable for protobuf compatibility
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

async def run_test(test_file: str):
    """Run a single test file"""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, str(Path(__file__).parent / "scenarios" / test_file)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {test_file}: {str(e)}")
        return False

async def main():
    """Run all consistency tests"""
    print("Starting Stage 2: Consistency Tests")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List of consistency test files
    test_files = [
        "test_consistency_schema.py",
        "test_consistency_index_sync.py",
        "test_consistency_metadata.py",
        "test_consistency_query.py"
    ]
    
    results = {}
    
    # Run each test
    for test_file in test_files:
        success = await run_test(test_file)
        results[test_file] = "✅ PASS" if success else "❌ FAIL"
    
    # Summary
    print(f"\n{'='*60}")
    print("CONSISTENCY TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_file, status in results.items():
        print(f"{test_file:<35} {status}")
    
    passed = sum(1 for status in results.values() if "PASS" in status)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All consistency tests completed successfully!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    asyncio.run(main())