#!/usr/bin/env python3
"""
Comprehensive Test Runner for Cursus Core Package

This program runs all tests that cover the core package components:
- assembler
- base  
- compiler
- config_fields (config_field in test directory)
- deps

It provides detailed reporting on test results, coverage analysis, and redundancy assessment.
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import ast
import re

# Note: sys.path setup is handled by conftest.py
# No manual path manipulation needed
# Note: project_root setup handled by conftest.py