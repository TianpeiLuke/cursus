"""
Base Script Validation Classes

Defines ValidationResult/AlignmentResult and the ScriptAnalyzer (AST-based script
analysis) used by the alignment validation framework. The legacy ScriptContract
data class was removed — contract data now lives in the unified StepInterface
(core/base/step_interface.py), sourced from the .step.yaml interfaces.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Set, Union, Any
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
