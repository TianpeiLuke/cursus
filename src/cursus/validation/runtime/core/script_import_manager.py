"""Script import manager for dynamically loading and executing scripts."""

import importlib.util
import sys
import time
import traceback
import os
from pathlib import Path
from typing import Callable, Dict, Any, Optional

try:
    import psutil
except ImportError:
    psutil = None

from ..utils.execution_context import ExecutionContext
from ..utils.result_models import ExecutionResult
from ..utils.error_handling import ScriptExecutionError, ScriptImportError

class ScriptImportManager:
    """Handles dynamic import and execution of pipeline scripts"""
    
    def __init__(self):
        """Initialize import manager"""
        self._imported_modules = {}
        self._script_cache = {}
    
    def import_script_main(self, script_path: str) -> Callable:
        """Dynamically import main function from script path"""
        
        if script_path in self._script_cache:
            return self._script_cache[script_path]
        
        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location("script_module", script_path)
            if spec is None or spec.loader is None:
                raise ScriptImportError(f"Cannot load script from {script_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules["script_module"] = module
            spec.loader.exec_module(module)
            
            # Get main function
            if not hasattr(module, 'main'):
                raise ScriptImportError(f"Script {script_path} does not have a 'main' function")
            
            main_func = getattr(module, 'main')
            
            # Cache for reuse
            self._script_cache[script_path] = main_func
            self._imported_modules[script_path] = module
            
            return main_func
            
        except Exception as e:
            # Convert to ScriptImportError for consistent error handling
            if not isinstance(e, ScriptImportError):
                raise ScriptImportError(f"Failed to import script {script_path}: {str(e)}")
            raise
    
    def execute_script_main(self, main_func: Callable, 
                           context: ExecutionContext) -> ExecutionResult:
        """Execute script main function with comprehensive error handling"""
        
        if not main_func:
            raise ScriptExecutionError("Main function cannot be None")
        
        if not context:
            raise ScriptExecutionError("Execution context cannot be None")
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Execute main function using Pydantic model_dump() method
            # to convert ExecutionContext to a dictionary
            context_dict = context.model_dump()
            result = main_func(
                input_paths=context_dict["input_paths"],
                output_paths=context_dict["output_paths"],
                environ_vars=context_dict["environ_vars"],
                job_args=context_dict["job_args"]
            )
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            return ExecutionResult(
                success=True,
                execution_time=end_time - start_time,
                memory_usage=max(end_memory - start_memory, 0),
                result_data=result,
                error_message=None
            )
            
        except Exception as e:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Convert to ScriptExecutionError for consistent error handling
            if not isinstance(e, ScriptExecutionError):
                execution_error = ScriptExecutionError(f"Script execution failed: {str(e)}")
                execution_error.__cause__ = e  # Preserve original exception
                raise execution_error
            raise
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in MB"""
        try:
            if psutil is None:
                return 0
                
            process = psutil.Process(os.getpid())
            return int(process.memory_info().rss / 1024 / 1024)  # Convert to MB
        except:
            return 0
