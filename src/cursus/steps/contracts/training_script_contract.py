"""
Training Script Contract Classes

Specialized contract classes for training scripts that use different SageMaker path conventions
than processing scripts.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Set
from ...core.base.contract_base import ValidationResult, ScriptAnalyzer
import os
import ast


class TrainingScriptContract(BaseModel):
    """
    Training script execution contract that supports SageMaker training job path conventions
    """

    entry_point: str = Field(..., description="Script entry point filename")
    expected_input_paths: Dict[str, str] = Field(
        ..., description="Mapping of logical names to expected input paths"
    )
    expected_output_paths: Dict[str, str] = Field(
        ..., description="Mapping of logical names to expected output paths"
    )
    expected_arguments: Dict[str, str] = Field(
        default_factory=dict,
        description="Expected script arguments with default values",
    )
    required_env_vars: List[str] = Field(
        ..., description="List of required environment variables"
    )
    optional_env_vars: Dict[str, str] = Field(
        default_factory=dict, description="Optional environment variables with defaults"
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
        """Validate input paths are valid SageMaker training paths"""
        valid_prefixes = ["/opt/ml/input/data", "/opt/ml/input/config", "/opt/ml/code"]
        for logical_name, path in v.items():
            if not any(path.startswith(prefix) for prefix in valid_prefixes):
                raise ValueError(
                    f"Input path for {logical_name} must start with one of {valid_prefixes}, got: {path}"
                )
        return v

    @field_validator("expected_output_paths")
    @classmethod
    def validate_output_paths(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate output paths are valid SageMaker training paths"""
        valid_prefixes = ["/opt/ml/model", "/opt/ml/output/data", "/opt/ml/checkpoints"]
        for logical_name, path in v.items():
            if not any(path.startswith(prefix) for prefix in valid_prefixes):
                raise ValueError(
                    f"Output path for {logical_name} must start with one of {valid_prefixes}, got: {path}"
                )
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
            analyzer = TrainingScriptAnalyzer(script_path)
            return self._validate_against_analysis(analyzer)
        except Exception as e:
            return ValidationResult.error([f"Error analyzing script: {str(e)}"])

    def _validate_against_analysis(
        self, analyzer: "TrainingScriptAnalyzer"
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
            if any(
                path.startswith(prefix)
                for prefix in ["/opt/ml/input/data", "/opt/ml/input/config"]
            ):
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

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )


class TrainingScriptAnalyzer(ScriptAnalyzer):
    """
    Specialized analyzer for training scripts that looks for SageMaker training path patterns
    """

    def get_input_paths(self) -> Set[str]:
        """Extract input paths used by training scripts"""
        if self._input_paths is None:
            self._input_paths = set()

            # Look for training-specific input path patterns
            training_input_prefixes = ["/opt/ml/input/data", "/opt/ml/input/config"]

            for node in ast.walk(self.ast_tree):
                # Look for string literals that look like input paths
                if isinstance(node, ast.Str):
                    if any(prefix in node.s for prefix in training_input_prefixes):
                        self._input_paths.add(node.s)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if any(prefix in node.value for prefix in training_input_prefixes):
                        self._input_paths.add(node.value)

        return self._input_paths

    def get_output_paths(self) -> Set[str]:
        """Extract output paths used by training scripts"""
        if self._output_paths is None:
            self._output_paths = set()

            # Look for training-specific output path patterns
            training_output_prefixes = [
                "/opt/ml/model",
                "/opt/ml/output/data",
                "/opt/ml/checkpoints",
            ]

            for node in ast.walk(self.ast_tree):
                # Look for string literals that look like output paths
                if isinstance(node, ast.Str):
                    if any(prefix in node.s for prefix in training_output_prefixes):
                        self._output_paths.add(node.s)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if any(prefix in node.value for prefix in training_output_prefixes):
                        self._output_paths.add(node.value)

        return self._output_paths
