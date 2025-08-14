"""
Script analysis data models for alignment validation.

Contains data models representing various elements found during
script analysis (imports, paths, arguments, etc.).
"""

from typing import List, Any, Optional
from pydantic import BaseModel, Field


class PathReference(BaseModel):
    """
    Represents a path reference found in script analysis.
    
    Attributes:
        path: The path string found
        line_number: Line number where the path was found
        context: Surrounding code context
        is_hardcoded: Whether this is a hardcoded path
        construction_method: How the path is constructed (e.g., 'os.path.join')
    """
    path: str
    line_number: int
    context: str
    is_hardcoded: bool = True
    construction_method: Optional[str] = None


class EnvVarAccess(BaseModel):
    """
    Represents environment variable access found in script analysis.
    
    Attributes:
        variable_name: Name of the environment variable
        line_number: Line number where the access was found
        context: Surrounding code context
        access_method: How the variable is accessed (e.g., 'os.environ', 'os.getenv')
        has_default: Whether a default value is provided
        default_value: The default value if provided
    """
    variable_name: str
    line_number: int
    context: str
    access_method: str
    has_default: bool = False
    default_value: Optional[str] = None


class ImportStatement(BaseModel):
    """
    Represents an import statement found in script analysis.
    
    Attributes:
        module_name: Name of the imported module
        import_alias: Alias used for the import (if any)
        line_number: Line number where the import was found
        is_from_import: Whether this is a 'from X import Y' statement
        imported_items: List of specific items imported (for from imports)
    """
    module_name: str
    import_alias: Optional[str]
    line_number: int
    is_from_import: bool = False
    imported_items: List[str] = Field(default_factory=list)


class ArgumentDefinition(BaseModel):
    """
    Represents a command-line argument definition found in script analysis.
    
    Attributes:
        argument_name: Name of the argument (without dashes)
        line_number: Line number where the argument was defined
        is_required: Whether the argument is required
        has_default: Whether the argument has a default value
        default_value: The default value if provided
        argument_type: Type of the argument (str, int, etc.)
        choices: Valid choices for the argument (if any)
    """
    argument_name: str
    line_number: int
    is_required: bool = False
    has_default: bool = False
    default_value: Optional[Any] = None
    argument_type: Optional[str] = None
    choices: Optional[List[str]] = None


class PathConstruction(BaseModel):
    """
    Represents a dynamic path construction found in script analysis.
    
    Attributes:
        base_path: The base path being constructed from
        construction_parts: Parts used in the construction
        line_number: Line number where the construction was found
        context: Surrounding code context
        method: Method used for construction (e.g., 'os.path.join', 'pathlib')
    """
    base_path: str
    construction_parts: List[str]
    line_number: int
    context: str
    method: str


class FileOperation(BaseModel):
    """
    Represents a file operation found in script analysis.
    
    Attributes:
        file_path: Path to the file being operated on
        operation_type: Type of operation (read, write, append, etc.)
        line_number: Line number where the operation was found
        context: Surrounding code context
        mode: File mode used (if specified)
        method: Method used for the operation (e.g., 'open', 'tarfile.open', 'pandas.read_csv')
    """
    file_path: str
    operation_type: str
    line_number: int
    context: str
    mode: Optional[str] = None
    method: Optional[str] = None
