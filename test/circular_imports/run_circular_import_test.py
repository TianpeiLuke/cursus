#!/usr/bin/env python3
"""
Simple runner script for the circular import test.

This script provides an easy way to run the circular import detection
test for the cursus package.
"""

import sys
import os

from .test_circular_imports import run_circular_import_tests

if __name__ == "__main__":
    print("Running Cursus Package Circular Import Test...")
    print("This test will systematically check all modules in the cursus package")
    print("for circular import dependencies.\n")

    success = run_circular_import_tests()

    if success:
        print("\n✅ No circular imports detected in the cursus package!")
    else:
        print("\n❌ Circular imports or other issues detected!")
        print("Please review the test output above for details.")

    sys.exit(0 if success else 1)
