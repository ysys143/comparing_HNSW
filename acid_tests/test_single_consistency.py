#!/usr/bin/env python3
"""Test single consistency test to debug issues"""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from scenarios.test_consistency_schema import main

if __name__ == "__main__":
    asyncio.run(main())