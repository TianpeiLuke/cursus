"""Execution context for script testing."""

from typing import Dict, Any, Optional
import argparse
from pydantic import BaseModel

class ExecutionContext(BaseModel):
    """Context for script execution"""
    input_paths: Dict[str, str]
    output_paths: Dict[str, str]
    environ_vars: Dict[str, str]
    job_args: Optional[argparse.Namespace] = None
