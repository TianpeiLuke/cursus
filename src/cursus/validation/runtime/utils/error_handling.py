"""Error handling utilities for the runtime testing system."""

class ScriptExecutionError(Exception):
    """Exception raised during script execution."""
    pass

class ScriptImportError(Exception):
    """Exception raised when a script cannot be imported."""
    pass

class DataFlowError(Exception):
    """Exception raised for data flow issues between pipeline steps."""
    pass

class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass

class ValidationError(Exception):
    """Exception raised for validation failures."""
    pass
