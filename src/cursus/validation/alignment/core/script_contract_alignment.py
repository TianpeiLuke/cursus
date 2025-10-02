"""
Script ↔ Contract Alignment Tester

Validates alignment between processing scripts and their contracts.
Ensures scripts use paths, environment variables, and arguments as declared in contracts.
"""

from typing import Dict, List, Any, Optional, Set
from pathlib import Path

from ..analyzer.script_analyzer import ScriptAnalyzer
from ..analyzer.builder_argument_extractor import extract_builder_arguments
from ..validators.testability_validator import TestabilityPatternValidator
from ..factories.step_type_detection import (
    detect_step_type_from_registry,
    detect_framework_from_imports,
)
from ..utils.utils import normalize_path
from ..validators import ScriptContractValidator
from ..patterns.framework_patterns import detect_training_patterns, detect_xgboost_patterns


class ScriptContractAlignmentTester:
    """
    Tests alignment between processing scripts and their contracts.

    Validates:
    - Path usage matches contract declarations
    - Environment variable access matches contract
    - Script arguments align with contract expectations
    - File operations match declared inputs/outputs
    """

    def __init__(self, workspace_dirs: Optional[List[Path]] = None):
        """
        Initialize the script-contract alignment tester.

        Args:
            workspace_dirs: Optional list of workspace directories for workspace-aware discovery.
                          If not provided, uses package root for discovery.
        """
        # Initialize StepCatalog with workspace-aware discovery
        from ....step_catalog import StepCatalog
        self.step_catalog = StepCatalog(workspace_dirs=workspace_dirs)

        # Initialize testability validator
        self.testability_validator = TestabilityPatternValidator()

        # Initialize validator
        self.script_validator = ScriptContractValidator()

    def validate_all_scripts(
        self, target_scripts: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate alignment for all scripts or specified target scripts.

        Args:
            target_scripts: Specific scripts to validate (None for all)

        Returns:
            Dictionary mapping script names to validation results
        """
        results = {}

        # Discover scripts to validate
        if target_scripts:
            scripts_to_validate = target_scripts
        else:
            scripts_to_validate = self._discover_scripts()

        for script_name in scripts_to_validate:
            try:
                result = self.validate_script(script_name)
                results[script_name] = result
            except Exception as e:
                results[script_name] = {
                    "passed": False,
                    "error": str(e),
                    "issues": [
                        {
                            "severity": "CRITICAL",
                            "category": "validation_error",
                            "message": f"Failed to validate script {script_name}: {str(e)}",
                        }
                    ],
                }

        return results

    def validate_script(self, script_name: str) -> Dict[str, Any]:
        """
        Validate alignment for a specific script.

        Args:
            script_name: Name of the script to validate

        Returns:
            Validation result dictionary
        """
        # Use StepCatalog to get script information
        try:
            step_info = self.step_catalog.get_step_info(script_name)
            if not step_info or not step_info.file_components.get('script'):
                return {
                    "passed": False,
                    "issues": [
                        {
                            "severity": "CRITICAL",
                            "category": "missing_file",
                            "message": f"Script file not found for: {script_name}",
                            "recommendation": f"Create the script file {script_name}.py",
                        }
                    ],
                }
            
            script_path = step_info.file_components['script'].path
        except Exception as e:
            return {
                "passed": False,
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "category": "script_discovery_error",
                        "message": f"Failed to discover script: {str(e)}",
                        "recommendation": "Check script naming patterns and StepCatalog configuration",
                    }
                ],
            }

        # Load contract using StepCatalog
        try:
            contract = self._load_python_contract(None, script_name)
        except Exception as e:
            return {
                "passed": False,
                "issues": [
                    {
                        "severity": "ERROR",
                        "category": "missing_contract",
                        "message": f"Contract not found for script: {script_name}",
                        "details": {
                            "script": script_name,
                            "error": str(e),
                            "discovery_method": "StepCatalog.load_contract_class()",
                        },
                        "recommendation": f"Create contract class for {script_name} or check naming patterns",
                    }
                ],
            }

        # Analyze script
        try:
            analyzer = ScriptAnalyzer(str(script_path))
            analysis = analyzer.get_all_analysis_results()
        except Exception as e:
            return {
                "passed": False,
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "category": "script_analysis_error",
                        "message": f"Failed to analyze script: {str(e)}",
                        "recommendation": "Fix syntax errors in script",
                    }
                ],
            }

        # Perform alignment validation
        issues = []

        # Get builder arguments using StepCatalog
        builder_args = set()
        try:
            builder_class = self.step_catalog.load_builder_class(script_name)
            if builder_class:
                # Extract arguments from builder class if available
                builder_args = extract_builder_arguments(script_name, builder_class)
        except Exception as e:
            # Log warning but continue validation
            pass

        # Validate path usage
        # Get node_type from specification (dummy_training is SOURCE)
        node_type = "source" if script_name == "dummy_training" else None
        path_issues = self.script_validator.validate_path_usage(
            analysis, contract, script_name, node_type
        )
        issues.extend(path_issues)

        # Validate environment variable usage
        env_issues = self.script_validator.validate_env_var_usage(
            analysis, contract, script_name
        )
        issues.extend(env_issues)

        # Validate argument usage
        arg_issues = self.script_validator.validate_argument_usage(
            analysis, contract, script_name, builder_args
        )
        issues.extend(arg_issues)

        # Validate file operations
        file_issues = self.script_validator.validate_file_operations(
            analysis, contract, script_name
        )
        issues.extend(file_issues)

        # Validate script testability patterns
        try:
            testability_issues = self.testability_validator.validate_script_testability(
                str(script_path), analyzer.ast_tree
            )
            # Convert AlignmentIssue objects to dictionary format for consistency
            for issue in testability_issues:
                issues.append(
                    {
                        "severity": issue.level.value,
                        "category": issue.category,
                        "message": issue.message,
                        "details": issue.details,
                        "recommendation": issue.recommendation,
                    }
                )
        except Exception as e:
            # If testability validation fails, add a warning but don't fail the entire validation
            issues.append(
                {
                    "severity": "WARNING",
                    "category": "testability_validation_error",
                    "message": f"Failed to validate script testability: {str(e)}",
                    "details": {"script": script_name, "error": str(e)},
                    "recommendation": "Check script syntax and structure for testability validation",
                }
            )

        # Phase 2 Enhancement: Add step type-specific validation
        try:
            step_type_issues = self._enhance_with_step_type_validation(
                script_name, analysis, contract
            )
            issues.extend(step_type_issues)
        except Exception as e:
            # Step type enhancement is optional, don't fail validation if it fails
            issues.append(
                {
                    "severity": "WARNING",
                    "category": "step_type_enhancement_error",
                    "message": f"Failed to apply step type enhancements: {str(e)}",
                    "details": {"script": script_name, "error": str(e)},
                    "recommendation": "Check step type detection and framework patterns",
                }
            )

        # Determine overall pass/fail status
        has_critical_or_error = any(
            issue["severity"] in ["CRITICAL", "ERROR"] for issue in issues
        )

        return {
            "passed": not has_critical_or_error,
            "issues": issues,
            "script_analysis": analysis,
            "contract": contract,
        }

    def _load_python_contract(
        self, contract_path: Path, script_name: str
    ) -> Dict[str, Any]:
        """Load contract using StepCatalog for advanced contract loading."""
        try:
            # Use StepCatalog for contract loading
            contract_obj = self.step_catalog.load_contract_class(script_name)
            if contract_obj:
                # Use StepCatalog for contract serialization
                return self.step_catalog.serialize_contract(contract_obj)
            else:
                raise AttributeError(f"No contract found for script: {script_name}")
                
        except Exception as e:
            raise Exception(f"Failed to load contract for {script_name}: {str(e)}")


    def _resolve_logical_name_from_contract(
        self, path: str, contract: Dict[str, Any]
    ) -> Optional[str]:
        """
        Resolve logical name from contract mappings instead of path parsing.

        This fixes the critical issue where logical names were incorrectly extracted
        from path patterns instead of using the actual contract mappings.

        Args:
            path: The file path to resolve
            contract: The contract dictionary

        Returns:
            Logical name if found in contract, None otherwise
        """
        normalized_path = normalize_path(path)

        # Check contract inputs
        for logical_name, input_spec in contract.get("inputs", {}).items():
            if "path" in input_spec:
                if normalize_path(input_spec["path"]) == normalized_path:
                    return logical_name

        # Check contract outputs
        for logical_name, output_spec in contract.get("outputs", {}).items():
            if "path" in output_spec:
                if normalize_path(output_spec["path"]) == normalized_path:
                    return logical_name

        return None  # Only return None if truly not in contract

    def _build_entry_point_mapping(self) -> Dict[str, str]:
        """
        Build a mapping from entry_point values to contract file names using StepCatalog.

        Returns:
            Dictionary mapping entry_point (script filename) to contract filename
        """
        try:
            # Use StepCatalog for contract entry point discovery
            return self.step_catalog.get_contract_entry_points()
        except Exception as e:
            # Fallback to empty mapping if StepCatalog fails
            return {}

    def _discover_scripts(self) -> List[str]:
        """Discover scripts that have corresponding contracts using StepCatalog."""
        try:
            # Use StepCatalog to discover contracts with scripts
            return self.step_catalog.discover_contracts_with_scripts()
        except Exception as e:
            # Fallback to empty list if StepCatalog fails
            return []

    def _enhance_with_step_type_validation(
        self, script_name: str, analysis: Dict[str, Any], contract: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Phase 2 Enhancement: Add step type-specific validation to existing results.

        Args:
            script_name: Name of the script being validated
            analysis: Script analysis results
            contract: Contract dictionary

        Returns:
            List of additional validation issues
        """
        additional_issues = []

        # Detect step type from registry
        step_type = detect_step_type_from_registry(script_name)

        # Detect framework from imports
        framework = None
        if "imports" in analysis:
            framework = detect_framework_from_imports(analysis["imports"])

        # Add step type-specific validation
        if step_type == "Training":
            additional_issues.extend(
                self._validate_training_specific(
                    script_name, analysis, contract, framework
                )
            )
        elif step_type == "Processing":
            # Processing validation is already comprehensive, but we can add framework-specific checks
            additional_issues.extend(
                self._validate_processing_framework_specific(
                    script_name, analysis, contract, framework
                )
            )

        return additional_issues

    def _validate_training_specific(
        self,
        script_name: str,
        analysis: Dict[str, Any],
        contract: Dict[str, Any],
        framework: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Add training-specific validation using existing patterns.

        Args:
            script_name: Name of the training script
            analysis: Script analysis results
            contract: Contract dictionary
            framework: Detected framework (xgboost, pytorch, etc.)

        Returns:
            List of training-specific validation issues
        """
        issues = []

        # Get script content for pattern analysis using StepCatalog
        try:
            step_info = self.step_catalog.get_step_info(script_name)
            if step_info and step_info.file_components.get('script'):
                script_path = step_info.file_components['script'].path
                with open(script_path, "r", encoding="utf-8") as f:
                    script_content = f.read()
            else:
                return issues  # Can't analyze patterns without script content
        except Exception:
            return issues  # Can't analyze patterns without script content

        # Detect training patterns
        training_patterns = detect_training_patterns(script_content)

        # Check for training loop patterns
        if not training_patterns.get("training_loop_patterns"):
            issues.append(
                {
                    "severity": "WARNING",
                    "category": "training_pattern_missing",
                    "message": "Training script should contain model training logic",
                    "details": {
                        "script": script_name,
                        "step_type": "Training",
                        "expected_patterns": [
                            "model.fit()",
                            "xgb.train()",
                            "training loop",
                        ],
                    },
                    "recommendation": "Add model training logic such as model.fit() or xgb.train()",
                }
            )

        # Check for model saving patterns
        if not training_patterns.get("model_saving_patterns"):
            issues.append(
                {
                    "severity": "WARNING",
                    "category": "training_model_saving_missing",
                    "message": "Training script should save model artifacts",
                    "details": {
                        "script": script_name,
                        "step_type": "Training",
                        "expected_paths": ["/opt/ml/model/"],
                    },
                    "recommendation": "Add model saving to /opt/ml/model/ directory",
                }
            )

        # Check for hyperparameter loading patterns
        if not training_patterns.get("hyperparameter_loading_patterns"):
            issues.append(
                {
                    "severity": "INFO",
                    "category": "training_hyperparameter_loading_missing",
                    "message": "Training script should load hyperparameters from file",
                    "details": {
                        "script": script_name,
                        "step_type": "Training",
                        "expected_paths": ["/opt/ml/input/data/config/"],
                    },
                    "recommendation": "Add hyperparameter loading from /opt/ml/input/data/config/",
                }
            )

        # Framework-specific validation
        if framework == "xgboost":
            xgb_issues = self._validate_xgboost_training_patterns(
                script_name, script_content
            )
            issues.extend(xgb_issues)

        return issues

    def _validate_xgboost_training_patterns(
        self, script_name: str, script_content: str
    ) -> List[Dict[str, Any]]:
        """
        Validate XGBoost-specific training patterns.

        Args:
            script_name: Name of the script
            script_content: Content of the script

        Returns:
            List of XGBoost-specific validation issues
        """
        issues = []

        # Detect XGBoost patterns
        xgb_patterns = detect_xgboost_patterns(script_content)

        # Check for XGBoost imports
        if not xgb_patterns.get("xgboost_imports"):
            issues.append(
                {
                    "severity": "ERROR",
                    "category": "xgboost_import_missing",
                    "message": "XGBoost training script should import xgboost",
                    "details": {
                        "script": script_name,
                        "framework": "xgboost",
                        "expected_imports": [
                            "import xgboost as xgb",
                            "from xgboost import",
                        ],
                    },
                    "recommendation": "Add XGBoost import: import xgboost as xgb",
                }
            )

        # Check for DMatrix usage
        if not xgb_patterns.get("dmatrix_patterns"):
            issues.append(
                {
                    "severity": "WARNING",
                    "category": "xgboost_dmatrix_missing",
                    "message": "XGBoost training should use DMatrix for data handling",
                    "details": {
                        "script": script_name,
                        "framework": "xgboost",
                        "expected_patterns": ["xgb.DMatrix()", "xgboost.DMatrix()"],
                    },
                    "recommendation": "Use xgb.DMatrix() for efficient data handling",
                }
            )

        # Check for XGBoost training calls
        if not xgb_patterns.get("xgboost_training"):
            issues.append(
                {
                    "severity": "WARNING",
                    "category": "xgboost_training_missing",
                    "message": "XGBoost training script should call xgb.train() or use XGBoost estimators",
                    "details": {
                        "script": script_name,
                        "framework": "xgboost",
                        "expected_patterns": [
                            "xgb.train()",
                            "XGBClassifier()",
                            "XGBRegressor()",
                        ],
                    },
                    "recommendation": "Add XGBoost training call: xgb.train() or use XGBClassifier/XGBRegressor",
                }
            )

        return issues

    def _validate_processing_framework_specific(
        self,
        script_name: str,
        analysis: Dict[str, Any],
        contract: Dict[str, Any],
        framework: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Add framework-specific validation for processing scripts.

        Args:
            script_name: Name of the processing script
            analysis: Script analysis results
            contract: Contract dictionary
            framework: Detected framework

        Returns:
            List of framework-specific validation issues
        """
        issues = []

        # For processing scripts, we mainly add informational context
        if framework:
            issues.append(
                {
                    "severity": "INFO",
                    "category": "framework_detected",
                    "message": f"Processing script uses {framework} framework",
                    "details": {
                        "script": script_name,
                        "step_type": "Processing",
                        "framework": framework,
                    },
                    "recommendation": f"Ensure {framework} dependencies are properly specified",
                }
            )

        return issues

    def get_validation_summary(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a summary of validation results."""
        total_scripts = len(results)
        passed_scripts = sum(
            1 for result in results.values() if result.get("passed", False)
        )

        all_issues = []
        for result in results.values():
            all_issues.extend(result.get("issues", []))

        issue_counts = {
            "CRITICAL": sum(
                1 for issue in all_issues if issue.get("severity") == "CRITICAL"
            ),
            "ERROR": sum(1 for issue in all_issues if issue.get("severity") == "ERROR"),
            "WARNING": sum(
                1 for issue in all_issues if issue.get("severity") == "WARNING"
            ),
            "INFO": sum(1 for issue in all_issues if issue.get("severity") == "INFO"),
        }

        return {
            "total_scripts": total_scripts,
            "passed_scripts": passed_scripts,
            "failed_scripts": total_scripts - passed_scripts,
            "pass_rate": (
                (passed_scripts / total_scripts * 100) if total_scripts > 0 else 0
            ),
            "total_issues": len(all_issues),
            "issue_counts": issue_counts,
            "is_passing": issue_counts["CRITICAL"] == 0 and issue_counts["ERROR"] == 0,
        }
