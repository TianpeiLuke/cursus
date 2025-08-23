"""Execution context for script testing."""

from typing import Dict, Any, Optional
import argparse
from pydantic import BaseModel, ConfigDict

class ExecutionContext(BaseModel):
    """Context for script execution"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    input_paths: Dict[str, str]
    output_paths: Dict[str, str]
    environ_vars: Dict[str, str]
    job_args: Optional[argparse.Namespace] = None
