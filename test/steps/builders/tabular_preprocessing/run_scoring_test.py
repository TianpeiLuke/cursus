#!/usr/bin/env python3
"""
Simple runner script for TabularPreprocessingStepBuilder scoring tests.

This script provides an easy way to run the comprehensive scoring tests
for the TabularPreprocessingStepBuilder.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from test_scoring import main
    
    if __name__ == "__main__":
        print("🎯 Starting TabularPreprocessingStepBuilder Scoring Tests...")
        print("=" * 70)
        main()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed and the src directory is accessible.")
    sys.exit(1)
except Exception as e:
    print(f"💥 Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
