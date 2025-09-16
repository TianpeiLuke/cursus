"""
Base Script Contract Classes

Defines the core ScriptContract class and validation framework for pipeline scripts.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Set, Union, Any
from pathlib import Path
import os
import ast
import logging

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Result of script contract validation"""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @classmethod
    def success(cls, message: str = "Validation passed") -> "ValidationResult":
        """Create a successful validation result"""
        return cls(is_valid=True)

    @classmethod
    def error(cls, errors: Union[str, List[str]]) -> "ValidationResult":
        """Create a failed validation result"""
        if isinstance(errors, str):
            errors = [errors]
        return cls(is_valid=False, errors=errors)

    @classmethod
    def combine(cls, results: List["ValidationResult"]) -> "ValidationResult":
        """Combine multiple validation results"""
        all_errors = []
        all_warnings = []

        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        return cls(
            is_valid=len(all_errors) == 0, errors=all_errors, warnings=all_warnings
        )

    def add_error(self, error: str) -> None:
        """Add an error to the result and mark as invalid"""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning to the result"""
        self.warnings.append(warning)


class AlignmentResult(ValidationResult):
    """Result of contract-specification alignment validation"""

    missing_outputs: List[str] = Field(default_factory=list)
    missing_inputs: List[str] = Field(default_factory=list)
    extra_outputs: List[str] = Field(default_factory=list)
    extra_inputs: List[str] = Field(default_factory=list)

    @classmethod
    def success(cls, message: str = "Alignment validation passed") -> "AlignmentResult":
        """Create a successful alignment result"""
        return cls(is_valid=True)

    @classmethod
    def error(
        cls,
        errors: Union[str, List[str]],
        missing_outputs: Optional[List[str]] = None,
        missing_inputs: Optional[List[str]] = None,
        extra_outputs: Optional[List[str]] = None,
        extra_inputs: Optional[List[str]] = None,
    ) -> "AlignmentResult":
        """Create a failed alignment result"""
        if isinstance(errors, str):
            errors = [errors]
        return cls(
            is_valid=False,
            errors=errors,
            missing_outputs=missing_outputs or [],
            missing_inputs=missing_inputs or [],
            extra_outputs=extra_outputs or [],
            extra_inputs=extra_inputs or [],
        )


class ScriptContract(BaseModel):
    """
    Script execution contract that defines explicit I/O, environment requirements, and CLI arguments
    """

    entry_point: str = Field(..., description="Script entry point filename")
    expected_input_paths: Dict[str, str] = Field(
        ..., description="Mapping of logical names to expected input paths"
    )
    expected_output_paths: Dict[str, str] = Field(
        ..., description="Mapping of logical names to expected output paths"
    )
    required_env_vars: List[str] = Field(
        ..., description="List of required environment variables"
    )
    optional_env_vars: Dict[str, str] = Field(
        default_factory=dict, description="Optional environment variables with defaults"
    )
    expected_arguments: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of argument names to container paths or values",
    )
    framework_requirements: Dict[str, str] = Field(
        default_factory=dict, description="Framework version requirements"
    )
    description: str = Field(
        default="", description="Human-readable description of the script"
    )

    @field_validator("entry_point")
    @classmethod
    def validate_entry_point(cls, v: str) -> str:
        """Validate entry point is a Python file"""
        if not v.endswith(".py"):
            raise ValueError("Entry point must be a Python file (.py)")
        return v

    @field_validator("expected_input_paths")
    @classmethod
    def validate_input_paths(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate input paths are absolute SageMaker paths"""
        for logical_name, path in v.items():
            if logical_name == "GeneratedPayloadSamples":
                if not path.startswith("/opt/ml/processing/"):
                    raise ValueError(
                        f"Input path for {logical_name} must start with /opt/ml/processing/, got: {path}"
                    )
            elif not path.startswith("/opt/ml/processing/input"):
                raise ValueError(
                    f"Input path for {logical_name} must start with /opt/ml/processing/input, got: {path}"
                )
        return v

    @field_validator("expected_output_paths")
    @classmethod
    def validate_output_paths(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate output paths are absolute SageMaker paths"""
        for logical_name, path in v.items():
            if not path.startswith("/opt/ml/processing/output"):
                raise ValueError(
                    f"Output path for {logical_name} must start with /opt/ml/processing/output, got: {path}"
                )
        return v

    @field_validator("expected_arguments")
    @classmethod
    def validate_arguments(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate argument names follow CLI conventions (kebab-case)"""
        for arg_name in v.keys():
            if not all(c.isalnum() or c == "-" for c in arg_name):
                raise ValueError(
                    f"Argument name contains invalid characters: {arg_name}"
                )
            if not arg_name.lower() == arg_name:
                raise ValueError(f"Argument name should be lowercase: {arg_name}")
        return v

    def validate_implementation(self, script_path: str) -> ValidationResult:
        """
        Validate that a script implementation matches this contract

        Args:
            script_path: Path to the script file to validate

        Returns:
            ValidationResult indicating whether the script complies with the contract
        """
        if not os.path.exists(script_path):
            return ValidationResult.error([f"Script file not found: {script_path}"])

        try:
            analyzer = ScriptAnalyzer(script_path)
            return self._validate_against_analysis(analyzer)
        except Exception as e:
            return ValidationResult.error([f"Error analyzing script: {str(e)}"])

    def _validate_against_analysis(
        self, analyzer: "ScriptAnalyzer"
    ) -> ValidationResult:
        """Validate contract against script analysis"""
        errors = []
        warnings = []

        # Validate input paths
        script_input_paths = analyzer.get_input_paths()
        for logical_name, expected_path in self.expected_input_paths.items():
            if expected_path not in script_input_paths:
                errors.append(
                    f"Script doesn't use expected input path: {expected_path} (for {logical_name})"
                )

        # Check for unexpected input paths
        expected_paths = set(self.expected_input_paths.values())
        unexpected_paths = script_input_paths - expected_paths
        for path in unexpected_paths:
            if path.startswith("/opt/ml/processing/input"):
                warnings.append(f"Script uses undeclared input path: {path}")

        # Validate output paths
        script_output_paths = analyzer.get_output_paths()
        for logical_name, expected_path in self.expected_output_paths.items():
            if expected_path not in script_output_paths:
                errors.append(
                    f"Script doesn't use expected output path: {expected_path} (for {logical_name})"
                )

        # Validate environment variables
        script_env_vars = analyzer.get_env_var_usage()
        missing_env_vars = set(self.required_env_vars) - script_env_vars
        if missing_env_vars:
            errors.append(
                f"Script missing required environment variables: {list(missing_env_vars)}"
            )

        # Validate arguments
        script_args = analyzer.get_argument_usage()
        for arg_name in self.expected_arguments.keys():
            if arg_name not in script_args:
                warnings.append(
                    f"Script doesn't seem to handle expected argument: --{arg_name}"
                )

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )


class ScriptAnalyzer:
    """
    Analyzes Python scripts to extract I/O patterns and environment variable usage
    """

    def __init__(self, script_path: str):
        self.script_path = script_path
        self._ast_tree: Optional[ast.AST] = None
        # Strategy 2 + 3: Early initialization with lazy loading flags
        self._input_paths: Set[str] = set()
        self._output_paths: Set[str] = set()
        self._env_vars: Set[str] = set()
        self._arguments: Set[str] = set()
        # Lazy loading flags to preserve original logic
        self._input_paths_loaded = False
        self._output_paths_loaded = False
        self._env_vars_loaded = False
        self._arguments_loaded = False

    @property
    def ast_tree(self) -> Any:
        """Lazy load and parse the script AST"""
        if self._ast_tree is None:
            with open(self.script_path, "r") as f:
                content = f.read()
            self._ast_tree = ast.parse(content)
        return self._ast_tree

    def get_input_paths(self) -> Set[str]:
        """Extract input paths used by the script"""
        # Strategy 2 + 3: Use lazy loading flag to preserve original logic
        if not self._input_paths_loaded:
            # Look for common input path patterns
            for node in ast.walk(self.ast_tree):
                # Look for string literals that look like input paths
                if isinstance(node, ast.Str):
                    if "/opt/ml/processing/input" in node.s:
                        self._input_paths.add(node.s)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if "/opt/ml/processing/input" in node.value:
                        self._input_paths.add(node.value)

                # Look for os.path.join calls with input paths
                if isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "os"
                        and node.func.attr == "path"
                        and hasattr(node.func, "attr")
                    ):
                        # This is a complex pattern, for now just look for string literals
                        pass
            
            self._input_paths_loaded = True

        return self._input_paths

    def get_output_paths(self) -> Set[str]:
        """Extract output paths used by the script"""
        # Strategy 2 + 3: Use lazy loading flag to preserve original logic
        if not self._output_paths_loaded:
            # Look for common output path patterns
            for node in ast.walk(self.ast_tree):
                # Look for string literals that look like output paths
                if isinstance(node, ast.Str):
                    if "/opt/ml/processing/output" in node.s:
                        self._output_paths.add(node.s)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if "/opt/ml/processing/output" in node.value:
                        self._output_paths.add(node.value)
            
            self._output_paths_loaded = True

        return self._output_paths

    def get_env_var_usage(self) -> Set[str]:
        """Extract environment variables accessed by the script"""
        # Strategy 2 + 3: Use lazy loading flag to preserve original logic
        if not self._env_vars_loaded:
            # Look for os.environ access patterns
            for node in ast.walk(self.ast_tree):
                # os.environ["VAR_NAME"]
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "os"
                    and node.value.attr == "environ"
                ):

                    if isinstance(node.slice, ast.Str):
                        self._env_vars.add(node.slice.s)
                    elif isinstance(node.slice, ast.Constant) and isinstance(
                        node.slice.value, str
                    ):
                        self._env_vars.add(node.slice.value)

                # os.environ.get("VAR_NAME")
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Attribute)
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "os"
                    and node.func.value.attr == "environ"
                    and node.func.attr == "get"
                ):

                    if node.args and isinstance(node.args[0], ast.Str):
                        self._env_vars.add(node.args[0].s)
                    elif (
                        node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    ):
                        self._env_vars.add(node.args[0].value)

                # os.getenv("VAR_NAME")
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "getenv"
                ):

                    if node.args and isinstance(node.args[0], ast.Str):
                        self._env_vars.add(node.args[0].s)
                    elif (
                        node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    ):
                        self._env_vars.add(node.args[0].value)
            
            self._env_vars_loaded = True

        return self._env_vars

    def get_argument_usage(self) -> Set[str]:
        """Extract command-line arguments used by the script"""
        # Strategy 2 + 3: Use lazy loading flag to preserve original logic
        if not self._arguments_loaded:
            # Look for argparse patterns
            for node in ast.walk(self.ast_tree):
                # Look for parser.add_argument calls
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                ):

                    # Check first argument for the argument name
                    if node.args and (
                        isinstance(node.args[0], ast.Str)
                        or (
                            isinstance(node.args[0], ast.Constant)
                            and isinstance(node.args[0].value, str)
                        )
                    ):
                        arg_name = (
                            node.args[0].s
                            if isinstance(node.args[0], ast.Str)
                            else node.args[0].value
                        )
                        # Strip leading dashes
                        if arg_name.startswith("--"):
                            self._arguments.add(arg_name[2:])
                        elif arg_name.startswith("-"):
                            self._arguments.add(arg_name[1:])
            
            self._arguments_loaded = True

        return self._arguments
