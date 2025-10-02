"""
Builder ↔ Configuration Alignment Tester

Validates alignment between step builders and their configuration requirements.
Ensures builders properly handle configuration fields and validation.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

from ..analyzer import StepCatalogAnalyzer
from ..patterns import PatternRecognizer


class BuilderConfigurationAlignmentTester:
    """
    Tests alignment between step builders and configuration requirements.

    Validates:
    - Configuration fields are properly handled
    - Required fields are validated
    - Default values are consistent
    - Configuration schema matches usage
    """

    def __init__(self, workspace_dirs: Optional[List[Path]] = None):
        """
        Initialize the builder-configuration alignment tester.

        Args:
            workspace_dirs: Optional list of workspace directories for workspace-aware discovery
        """
        # Initialize StepCatalog with workspace-aware discovery
        from ....step_catalog import StepCatalog
        self.step_catalog = StepCatalog(workspace_dirs=workspace_dirs)

        # Initialize unified analyzer with StepCatalog integration
        self.unified_analyzer = StepCatalogAnalyzer(self.step_catalog)
        self.pattern_recognizer = PatternRecognizer()

    def validate_all_builders(
        self, target_scripts: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate alignment for all builders or specified target scripts.

        Args:
            target_scripts: Specific scripts to validate (None for all)

        Returns:
            Dictionary mapping builder names to validation results
        """
        results = {}

        # Discover builders to validate
        if target_scripts:
            builders_to_validate = target_scripts
        else:
            builders_to_validate = self.step_catalog.list_available_builders()

        for builder_name in builders_to_validate:
            try:
                result = self.validate_builder(builder_name)
                results[builder_name] = result
            except Exception as e:
                results[builder_name] = {
                    "passed": False,
                    "error": str(e),
                    "issues": [
                        {
                            "severity": "CRITICAL",
                            "category": "validation_error",
                            "message": f"Failed to validate builder {builder_name}: {str(e)}",
                        }
                    ],
                }

        return results

    def validate_builder(self, builder_name: str) -> Dict[str, Any]:
        """
        Validate alignment for a specific builder.

        Args:
            builder_name: Name of the builder to validate

        Returns:
            Validation result dictionary
        """
        # Use StepCatalog to discover builder class directly
        try:
            builder_class = self.step_catalog.load_builder_class(builder_name)
            if not builder_class:
                return {
                    "passed": False,
                    "issues": [
                        {
                            "severity": "CRITICAL",
                            "category": "missing_builder",
                            "message": f"Builder class not found for {builder_name}",
                            "details": {
                                "builder_name": builder_name,
                                "discovery_method": "StepCatalog.load_builder_class()",
                            },
                            "recommendation": f"Create builder class for {builder_name} or check naming patterns",
                        }
                    ],
                }
        except Exception as e:
            return {
                "passed": False,
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "category": "builder_load_error",
                        "message": f"Failed to load builder class: {str(e)}",
                        "recommendation": "Fix Python syntax or builder class structure",
                    }
                ],
            }

        # Use StepCatalog to discover config class directly
        try:
            config_class = self.step_catalog.load_config_class(builder_name)
            if not config_class:
                return {
                    "passed": False,
                    "issues": [
                        {
                            "severity": "ERROR",
                            "category": "missing_configuration",
                            "message": f"Configuration class not found for {builder_name}",
                            "details": {
                                "builder_name": builder_name,
                                "discovery_method": "StepCatalog.load_config_class()",
                            },
                            "recommendation": f"Create configuration class for {builder_name} or check naming patterns",
                        }
                    ],
                }
        except Exception as e:
            return {
                "passed": False,
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "category": "config_load_error",
                        "message": f"Failed to load configuration class: {str(e)}",
                        "recommendation": "Fix Python syntax or configuration class structure",
                    }
                ],
            }

        # Analyze step using unified StepCatalogAnalyzer
        try:
            step_analysis = self.unified_analyzer.analyze_step(builder_name)
            if "error" in step_analysis:
                return {
                    "passed": False,
                    "issues": [
                        {
                            "severity": "CRITICAL",
                            "category": "step_analysis_error",
                            "message": f"Failed to analyze step: {step_analysis['error']}",
                            "recommendation": "Fix step structure or implementation",
                        }
                    ],
                }
            
            # Extract individual analysis components
            config_analysis = step_analysis.get("config_analysis", {})
            builder_analysis = step_analysis.get("builder_analysis", {})
            
            # Check for analysis errors
            if "error" in config_analysis:
                return {
                    "passed": False,
                    "issues": [
                        {
                            "severity": "CRITICAL",
                            "category": "config_analysis_error",
                            "message": f"Failed to analyze configuration: {config_analysis['error']}",
                            "recommendation": "Fix configuration class structure or implementation",
                        }
                    ],
                }
            
            if "error" in builder_analysis:
                return {
                    "passed": False,
                    "issues": [
                        {
                            "severity": "CRITICAL",
                            "category": "builder_analysis_error",
                            "message": f"Failed to analyze builder: {builder_analysis['error']}",
                            "recommendation": "Fix builder class structure or implementation",
                        }
                    ],
                }
                
        except Exception as e:
            return {
                "passed": False,
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "category": "unified_analysis_error",
                        "message": f"Failed to perform unified analysis: {str(e)}",
                        "recommendation": "Check step structure and StepCatalog integration",
                    }
                ],
            }

        # Perform alignment validation
        issues = []

        # Validate configuration field handling
        config_issues = self._validate_configuration_fields(
            builder_analysis, config_analysis, builder_name
        )
        issues.extend(config_issues)

        # Validate required field validation
        validation_issues = self._validate_required_fields(
            builder_analysis, config_analysis, builder_name
        )
        issues.extend(validation_issues)

        # Validate configuration import
        import_issues = self._validate_config_import(
            builder_analysis, config_analysis, builder_name
        )
        issues.extend(import_issues)

        # Determine overall pass/fail status
        has_critical_or_error = any(
            issue["severity"] in ["CRITICAL", "ERROR"] for issue in issues
        )

        return {
            "passed": not has_critical_or_error,
            "issues": issues,
            "builder_analysis": builder_analysis,
            "config_analysis": config_analysis,
        }

    def _validate_required_fields(
        self,
        builder_analysis: Dict[str, Any],
        specification: Dict[str, Any],
        builder_name: str,
    ) -> List[Dict[str, Any]]:
        """Validate that builder properly validates required fields."""
        issues = []

        config_schema = specification.get("configuration", {})
        required_fields = set(config_schema.get("required", []))

        # Check if builder has validation logic
        has_validation = len(builder_analysis.get("validation_calls", [])) > 0

        if required_fields and not has_validation:
            issues.append(
                {
                    "severity": "WARNING",
                    "category": "required_field_validation",
                    "message": f"Builder has required fields but no validation logic detected",
                    "details": {
                        "required_fields": list(required_fields),
                        "builder": builder_name,
                    },
                    "recommendation": "Add validation logic for required configuration fields",
                }
            )

        return issues

    def _validate_default_values(
        self,
        builder_analysis: Dict[str, Any],
        specification: Dict[str, Any],
        builder_name: str,
    ) -> List[Dict[str, Any]]:
        """Validate that default values are consistent between builder and specification."""
        issues = []

        config_schema = specification.get("configuration", {})
        spec_defaults = {}

        # Extract default values from specification
        for field_name, field_spec in config_schema.get("fields", {}).items():
            if "default" in field_spec:
                spec_defaults[field_name] = field_spec["default"]

        # Get default assignments from builder
        builder_defaults = set()
        for assignment in builder_analysis.get("default_assignments", []):
            builder_defaults.add(assignment["field_name"])

        # Check for specification defaults not handled in builder
        for field_name, default_value in spec_defaults.items():
            if field_name not in builder_defaults:
                issues.append(
                    {
                        "severity": "INFO",
                        "category": "default_values",
                        "message": f"Specification defines default for {field_name} but builder does not set it",
                        "details": {
                            "field_name": field_name,
                            "spec_default": default_value,
                            "builder": builder_name,
                        },
                        "recommendation": f"Consider setting default value for {field_name} in builder",
                    }
                )

        return issues

    def _validate_config_import(
        self,
        builder_analysis: Dict[str, Any],
        config_analysis: Dict[str, Any],
        builder_name: str,
    ) -> List[Dict[str, Any]]:
        """Validate that builder properly imports and uses configuration."""
        issues = []

        # Check if builder imports the configuration class
        config_class_name = config_analysis.get("class_name", "")

        # Look for import statements in builder (this is a simplified check)
        # In a real implementation, we'd parse import statements from the AST
        has_config_import = any(
            class_def["class_name"]
            == config_class_name.replace("Config", "StepBuilder")
            for class_def in builder_analysis.get("class_definitions", [])
        )

        if not has_config_import:
            issues.append(
                {
                    "severity": "INFO",
                    "category": "config_import",
                    "message": f"Builder may not be properly importing configuration class {config_class_name}",
                    "details": {
                        "config_class": config_class_name,
                        "builder": builder_name,
                    },
                    "recommendation": f"Ensure builder imports and uses {config_class_name}",
                }
            )

        return issues

    def _validate_configuration_fields(
        self,
        builder_analysis: Dict[str, Any],
        config_analysis: Dict[str, Any],
        builder_name: str,
    ) -> List[Dict[str, Any]]:
        """Validate that builder properly handles configuration fields."""
        issues = []

        # Get configuration fields from analysis (now includes inherited fields)
        config_fields = set(config_analysis.get("fields", {}).keys())
        # Handle both list and set types for required_fields
        required_fields_raw = config_analysis.get("required_fields", [])
        if isinstance(required_fields_raw, list):
            required_fields = set(required_fields_raw)
        else:
            required_fields = set(required_fields_raw)

        # Get fields accessed in builder
        accessed_fields = set()
        for access in builder_analysis.get("config_accesses", []):
            accessed_fields.add(access["field_name"])

        # Apply pattern-aware filtering to reduce false positives
        filtered_issues = []

        # Check for accessed fields not in configuration
        undeclared_fields = accessed_fields - config_fields
        for field_name in undeclared_fields:
            # Apply architectural pattern recognition
            if not self._is_acceptable_pattern(
                field_name, builder_name, "undeclared_access"
            ):
                filtered_issues.append(
                    {
                        "severity": "ERROR",
                        "category": "configuration_fields",
                        "message": f"Builder accesses undeclared configuration field: {field_name}",
                        "details": {"field_name": field_name, "builder": builder_name},
                        "recommendation": f"Add {field_name} to configuration class or remove from builder",
                    }
                )

        # Check for required fields not accessed
        unaccessed_required = required_fields - accessed_fields
        for field_name in unaccessed_required:
            # Apply pattern-aware filtering
            if not self._is_acceptable_pattern(
                field_name, builder_name, "unaccessed_required"
            ):
                filtered_issues.append(
                    {
                        "severity": "WARNING",
                        "category": "configuration_fields",
                        "message": f"Required configuration field not accessed in builder: {field_name}",
                        "details": {"field_name": field_name, "builder": builder_name},
                        "recommendation": f"Access required field {field_name} in builder or make it optional",
                    }
                )

        return filtered_issues

    def _is_acceptable_pattern(
        self, field_name: str, builder_name: str, issue_type: str
    ) -> bool:
        """
        Determine if a configuration field issue represents an acceptable architectural pattern.

        Uses the extracted PatternRecognizer component for consistent pattern recognition.

        Args:
            field_name: Name of the configuration field
            builder_name: Name of the builder
            issue_type: Type of issue ('undeclared_access', 'unaccessed_required')

        Returns:
            True if this is an acceptable pattern (should be filtered out)
        """
        return self.pattern_recognizer.is_acceptable_pattern(
            field_name, builder_name, issue_type
        )

    def _validate_required_fields(
        self,
        builder_analysis: Dict[str, Any],
        config_analysis: Dict[str, Any],
        builder_name: str,
    ) -> List[Dict[str, Any]]:
        """Validate that builder properly validates required fields."""
        issues = []

        required_fields = set(config_analysis.get("required_fields", []))

        # Check if builder has validation logic
        has_validation = len(builder_analysis.get("validation_calls", [])) > 0

        if required_fields and not has_validation:
            issues.append(
                {
                    "severity": "INFO",
                    "category": "required_field_validation",
                    "message": f"Builder has required fields but no explicit validation logic detected",
                    "details": {
                        "required_fields": list(required_fields),
                        "builder": builder_name,
                    },
                    "recommendation": "Consider adding explicit validation logic for required configuration fields",
                }
            )

        return issues
