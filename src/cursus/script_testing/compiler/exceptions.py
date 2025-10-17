"""
Script Compilation Exceptions

Defines exception classes for script compilation errors.
Mirrors the exception patterns in cursus/core but targets script compilation.
"""

from typing import Optional, List, Any


class ScriptCompilationError(Exception):
    """
    Base exception for script compilation errors.
    
    Mirrors compilation errors in cursus/core but targets script compilation
    instead of pipeline compilation.
    """
    
    def __init__(
        self, 
        message: str, 
        node_name: Optional[str] = None,
        details: Optional[dict] = None
    ):
        """
        Initialize script compilation error.
        
        Args:
            message: Error message describing the compilation failure
            node_name: Optional DAG node name where error occurred
            details: Optional dictionary with additional error details
        """
        self.message = message
        self.node_name = node_name
        self.details = details or {}
        
        # Format error message with context
        if node_name:
            full_message = f"Script compilation failed for node '{node_name}': {message}"
        else:
            full_message = f"Script compilation failed: {message}"
            
        super().__init__(full_message)
    
    def __str__(self) -> str:
        """String representation of the error."""
        return self.args[0]


class ScriptDiscoveryError(ScriptCompilationError):
    """
    Exception raised when script discovery fails.
    
    This occurs when the compiler cannot locate the script file
    for a DAG node, either through step catalog or fallback discovery.
    """
    
    def __init__(
        self, 
        message: str, 
        node_name: str,
        attempted_paths: Optional[List[str]] = None,
        step_catalog_available: bool = False
    ):
        """
        Initialize script discovery error.
        
        Args:
            message: Error message describing the discovery failure
            node_name: DAG node name where discovery failed
            attempted_paths: List of paths that were attempted
            step_catalog_available: Whether step catalog was available for discovery
        """
        self.attempted_paths = attempted_paths or []
        self.step_catalog_available = step_catalog_available
        
        details = {
            "attempted_paths": self.attempted_paths,
            "step_catalog_available": step_catalog_available,
        }
        
        super().__init__(message, node_name, details)


class ScriptValidationError(ScriptCompilationError):
    """
    Exception raised when script validation fails.
    
    This occurs when a discovered script fails validation checks,
    such as missing main() function or invalid structure.
    """
    
    def __init__(
        self, 
        message: str, 
        node_name: str,
        script_path: str,
        validation_failures: Optional[List[str]] = None
    ):
        """
        Initialize script validation error.
        
        Args:
            message: Error message describing the validation failure
            node_name: DAG node name where validation failed
            script_path: Path to the script that failed validation
            validation_failures: List of specific validation failures
        """
        self.script_path = script_path
        self.validation_failures = validation_failures or []
        
        details = {
            "script_path": script_path,
            "validation_failures": self.validation_failures,
        }
        
        super().__init__(message, node_name, details)


class DAGValidationError(ScriptCompilationError):
    """
    Exception raised when DAG validation fails for script execution.
    
    This occurs when the DAG structure is invalid for script execution,
    such as cycles, missing nodes, or invalid dependencies.
    """
    
    def __init__(
        self, 
        message: str,
        dag_errors: Optional[List[str]] = None,
        dag_warnings: Optional[List[str]] = None
    ):
        """
        Initialize DAG validation error.
        
        Args:
            message: Error message describing the DAG validation failure
            dag_errors: List of DAG validation errors
            dag_warnings: List of DAG validation warnings
        """
        self.dag_errors = dag_errors or []
        self.dag_warnings = dag_warnings or []
        
        details = {
            "dag_errors": self.dag_errors,
            "dag_warnings": self.dag_warnings,
        }
        
        super().__init__(message, None, details)


class ExecutionPlanValidationError(ScriptCompilationError):
    """
    Exception raised when execution plan validation fails.
    
    This occurs when the generated execution plan is invalid,
    such as missing scripts, invalid execution order, or dependency issues.
    """
    
    def __init__(
        self, 
        message: str,
        plan_errors: Optional[List[str]] = None,
        plan_warnings: Optional[List[str]] = None
    ):
        """
        Initialize execution plan validation error.
        
        Args:
            message: Error message describing the plan validation failure
            plan_errors: List of execution plan validation errors
            plan_warnings: List of execution plan validation warnings
        """
        self.plan_errors = plan_errors or []
        self.plan_warnings = plan_warnings or []
        
        details = {
            "plan_errors": self.plan_errors,
            "plan_warnings": self.plan_warnings,
        }
        
        super().__init__(message, None, details)


class StepCatalogIntegrationError(ScriptCompilationError):
    """
    Exception raised when step catalog integration fails.
    
    This occurs when there are issues with step catalog operations
    during script compilation, such as catalog unavailability or
    contract loading failures.
    """
    
    def __init__(
        self, 
        message: str, 
        node_name: Optional[str] = None,
        catalog_operation: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize step catalog integration error.
        
        Args:
            message: Error message describing the integration failure
            node_name: Optional DAG node name where error occurred
            catalog_operation: Optional name of the catalog operation that failed
            original_error: Optional original exception that caused the failure
        """
        self.catalog_operation = catalog_operation
        self.original_error = original_error
        
        details = {
            "catalog_operation": catalog_operation,
            "original_error": str(original_error) if original_error else None,
        }
        
        super().__init__(message, node_name, details)


class InteractiveInputError(ScriptCompilationError):
    """
    Exception raised when interactive input collection fails.
    
    This occurs when there are issues with collecting user inputs
    for script execution, such as invalid input values or
    collection process failures.
    """
    
    def __init__(
        self, 
        message: str, 
        node_name: Optional[str] = None,
        input_type: Optional[str] = None,
        invalid_value: Optional[Any] = None
    ):
        """
        Initialize interactive input error.
        
        Args:
            message: Error message describing the input failure
            node_name: Optional DAG node name where error occurred
            input_type: Optional type of input that failed
            invalid_value: Optional invalid value that caused the failure
        """
        self.input_type = input_type
        self.invalid_value = invalid_value
        
        details = {
            "input_type": input_type,
            "invalid_value": str(invalid_value) if invalid_value is not None else None,
        }
        
        super().__init__(message, node_name, details)


def format_compilation_error(error: ScriptCompilationError) -> str:
    """
    Format a script compilation error for user-friendly display.
    
    Args:
        error: The script compilation error to format
        
    Returns:
        Formatted error message string
    """
    lines = [f"❌ {error.message}"]
    
    if error.node_name:
        lines.append(f"   Node: {error.node_name}")
    
    if isinstance(error, ScriptDiscoveryError):
        if error.attempted_paths:
            lines.append("   Attempted paths:")
            for path in error.attempted_paths:
                lines.append(f"     - {path}")
        lines.append(f"   Step catalog available: {error.step_catalog_available}")
    
    elif isinstance(error, ScriptValidationError):
        lines.append(f"   Script path: {error.script_path}")
        if error.validation_failures:
            lines.append("   Validation failures:")
            for failure in error.validation_failures:
                lines.append(f"     - {failure}")
    
    elif isinstance(error, DAGValidationError):
        if error.dag_errors:
            lines.append("   DAG errors:")
            for dag_error in error.dag_errors:
                lines.append(f"     - {dag_error}")
        if error.dag_warnings:
            lines.append("   DAG warnings:")
            for warning in error.dag_warnings:
                lines.append(f"     - {warning}")
    
    elif isinstance(error, ExecutionPlanValidationError):
        if error.plan_errors:
            lines.append("   Plan errors:")
            for plan_error in error.plan_errors:
                lines.append(f"     - {plan_error}")
        if error.plan_warnings:
            lines.append("   Plan warnings:")
            for warning in error.plan_warnings:
                lines.append(f"     - {warning}")
    
    elif isinstance(error, StepCatalogIntegrationError):
        if error.catalog_operation:
            lines.append(f"   Catalog operation: {error.catalog_operation}")
        if error.original_error:
            lines.append(f"   Original error: {error.original_error}")
    
    elif isinstance(error, InteractiveInputError):
        if error.input_type:
            lines.append(f"   Input type: {error.input_type}")
        if error.invalid_value is not None:
            lines.append(f"   Invalid value: {error.invalid_value}")
    
    return "\n".join(lines)
