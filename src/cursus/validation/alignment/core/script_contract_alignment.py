"""
Script ↔ Contract Alignment Tester

Validates alignment between processing scripts and their contracts.
Ensures scripts use paths, environment variables, and arguments as declared in contracts.
"""

from typing import Dict, List, Any, Optional, Set
from pathlib import Path

# Note: These imports reference removed modules - functionality needs to be replaced
# from ..analyzer.script_analyzer import ScriptAnalyzer
# from ..analyzer.builder_argument_extractor import extract_builder_arguments  
# from ..validators.testability_validator import TestabilityPatternValidator
from ....registry.step_names import (
    get_sagemaker_step_type,
    get_canonical_name_from_file_name,
)
from ....step_catalog import StepCatalog
from ..utils.utils import normalize_path
# Note: These imports reference removed modules - functionality needs to be replaced
# from ..validators import ScriptContractValidator
# from ..patterns.framework_patterns import detect_training_patterns, detect_xgboost_patterns


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
        # Store workspace directories
        self.workspace_dirs = workspace_dirs
        
        # Initialize StepCatalog with workspace-aware discovery
        from ....step_catalog import StepCatalog
        self.step_catalog = StepCatalog(workspace_dirs=workspace_dirs)

        # Note: These validators were removed during consolidation
        # self.testability_validator = TestabilityPatternValidator()
        # self.script_validator = ScriptContractValidator()
        
        # TODO: Replace with consolidated validation logic
        self.testability_validator = None
        self.script_validator = None

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

        # RESTORED: Use consolidated validation logic with restored ScriptAnalyzer
        from ..analyzer.script_analyzer import ScriptAnalyzer
        
        # Use restored ScriptAnalyzer for contract alignment validation
        try:
            analyzer = ScriptAnalyzer(str(script_path))
        except Exception as e:
            return {
                "passed": False,
                "issues": [
                    {
                        "severity": "ERROR",
                        "category": "script_analysis_error",
                        "message": f"Failed to analyze script: {str(e)}",
                        "details": {
                            "script": script_name,
                            "script_path": str(script_path),
                            "error": str(e),
                        },
                        "recommendation": "Check script syntax and ensure it can be parsed",
                    }
                ],
            }
        
        # Validate main function signature
        main_function_result = analyzer.validate_main_function_signature()
        if not main_function_result.get("has_main"):
            issues = [{
                "severity": "CRITICAL",
                "category": "missing_main_function",
                "message": "Script must define main function with standard signature",
                "details": {
                    "script": script_name,
                    "expected_signature": "def main(input_paths, output_paths, environ_vars, job_args)"
                },
                "recommendation": "Add main function with standard signature"
            }]
        elif not main_function_result.get("signature_valid"):
            issues = [{
                "severity": "ERROR",
                "category": "invalid_main_signature",
                "message": "Main function signature does not match expected format",
                "details": {
                    "script": script_name,
                    "actual_params": main_function_result.get("actual_params", []),
                    "expected_params": main_function_result.get("expected_params", []),
                    "issues": main_function_result.get("issues", [])
                },
                "recommendation": "Fix main function signature to match: def main(input_paths, output_paths, environ_vars, job_args)"
            }]
        else:
            issues = []
        
        # Extract parameter usage
        parameter_usage = analyzer.extract_parameter_usage()
        
        # Validate contract alignment
        alignment_issues = analyzer.validate_contract_alignment(contract)
        issues.extend(alignment_issues)
        
        # Create analysis results
        analysis = {
            "main_function": main_function_result,
            "parameter_usage": parameter_usage,
            "contract_alignment": {
                "total_issues": len(alignment_issues),
                "error_count": len([i for i in alignment_issues if i["severity"] == "ERROR"]),
                "warning_count": len([i for i in alignment_issues if i["severity"] == "WARNING"])
            }
        }

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

        # Detect step type from registry using registry functions instead of redundant factories
        try:
            canonical_name = get_canonical_name_from_file_name(script_name)
            step_type = get_sagemaker_step_type(canonical_name)
        except (ValueError, Exception):
            step_type = "Processing"  # Default fallback

        # Detect framework using StepCatalog instead of redundant function
        framework = None
        try:
            framework = self.step_catalog.detect_framework(script_name)
        except (ValueError, Exception):
            framework = None

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

        # TODO: Replace with consolidated pattern detection
        # training_patterns = detect_training_patterns(script_content)
        training_patterns = {}  # Placeholder until pattern detection is restored

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

        # TODO: Replace with consolidated pattern detection
        # xgb_patterns = detect_xgboost_patterns(script_content)
        xgb_patterns = {}  # Placeholder until pattern detection is restored

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
