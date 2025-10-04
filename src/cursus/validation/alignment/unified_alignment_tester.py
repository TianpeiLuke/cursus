"""
Enhanced Unified Alignment Tester

This module provides a configuration-driven unified interface for testing alignment 
across all validation levels in the cursus framework. It orchestrates validation 
based on step-type-aware configuration rules.

Enhanced with:
- Configuration-driven validation level control
- Step-type-aware validation
- Priority-based validation system
- Consolidated discovery methods
- Method interface focus
"""

from typing import Dict, Any, List, Optional, Set
import logging
from pathlib import Path

from .config import (
    ValidationLevel,
    get_validation_ruleset,
    get_enabled_validation_levels,
    is_step_type_excluded,
    is_validation_level_enabled,
    validate_step_type_configuration
)
from .core.level_validators import LevelValidators
from ...step_catalog import StepCatalog
from ...registry.step_names import get_sagemaker_step_type

logger = logging.getLogger(__name__)


class UnifiedAlignmentTester:
    """
    Enhanced Unified Alignment Tester with configuration-driven validation.
    
    This class orchestrates validation across all levels based on step-type-aware
    configuration rules, providing dramatic performance improvements through
    validation level skipping.
    """
    
    def __init__(self, workspace_dirs: Optional[List[str]] = None, **kwargs):
        """
        Initialize the enhanced unified alignment tester.
        
        Args:
            workspace_dirs: Optional list of workspace directories to search.
                           If None, only discovers package internal steps.
            **kwargs: Additional configuration options (preserved for backward compatibility)
        """
        self.workspace_dirs = workspace_dirs
        self.validation_config = {}  # Will be loaded from configuration system
        
        # Initialize step catalog - key for discovery methods
        self.step_catalog = StepCatalog(workspace_dirs=workspace_dirs)
        
        # Validate configuration on initialization
        config_issues = validate_step_type_configuration()
        if config_issues:
            logger.warning(f"Configuration issues found: {config_issues}")
            # Don't raise error, just log warnings to maintain backward compatibility
        
        # Initialize level validators (replaces 4 separate level testers)
        self.level_validators = LevelValidators(workspace_dirs)
        
        # Preserve legacy kwargs for backward compatibility
        self.legacy_kwargs = kwargs
        
        workspace_count = len(self.workspace_dirs) if self.workspace_dirs else 0
        logger.info(f"Initialized Enhanced UnifiedAlignmentTester with {workspace_count} workspace directories")
    
    def run_full_validation(self, target_scripts: Optional[List[str]] = None, 
                          skip_levels: Optional[Set[int]] = None) -> Dict[str, Any]:
        """
        Enhanced run_full_validation with configuration-driven approach.
        
        Args:
            target_scripts: Optional list of specific scripts to validate
            skip_levels: Optional set of validation levels to skip (legacy support)
            
        Returns:
            Dictionary containing validation results
        """
        logger.info("Starting configuration-driven full validation")
        
        if target_scripts:
            results = {}
            for script_name in target_scripts:
                results[script_name] = self.run_validation_for_step(script_name)
            return results
        else:
            return self.run_validation_for_all_steps()
    
    def run_validation_for_step(self, step_name: str) -> Dict[str, Any]:
        """
        Run validation for a specific step based on its ruleset.
        
        Args:
            step_name: Name of the step to validate
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Running configuration-driven validation for step: {step_name}")
        
        # Get step type from registry
        sagemaker_step_type = get_sagemaker_step_type(step_name)
        
        # Get validation ruleset
        ruleset = get_validation_ruleset(sagemaker_step_type)
        
        # Check if step type is excluded
        if is_step_type_excluded(sagemaker_step_type):
            return self._handle_excluded_step(step_name, sagemaker_step_type, ruleset)
        
        # Run enabled validation levels
        return self._run_enabled_validation_levels(step_name, sagemaker_step_type, ruleset)
    
    def run_validation_for_all_steps(self) -> Dict[str, Any]:
        """
        Run validation for all discovered steps.
        
        Returns:
            Dictionary containing validation results for all steps
        """
        logger.info("Running configuration-driven validation for all steps")
        
        # Discover all steps using consolidated discovery method
        discovered_steps = self._discover_all_steps()
        
        results = {}
        for step_name in discovered_steps:
            results[step_name] = self.run_validation_for_step(step_name)
        
        return results
    
    def _discover_all_steps(self) -> List[str]:
        """
        Discover ALL pipeline steps for comprehensive validation.
        
        The step-type-specific ruleset controls which validation levels
        are applied to each step type, enabling comprehensive coverage
        while skipping inappropriate validation levels.
        
        Returns:
            List of all concrete pipeline steps (21 steps)
        """
        logger.info("Discovering all steps using step catalog")
        
        try:
            # COMPREHENSIVE: Use list_available_steps() for all concrete steps
            all_steps = self.step_catalog.list_available_steps()
            
            logger.info(f"Discovered {len(all_steps)} steps for comprehensive validation")
            return all_steps
            
        except Exception as e:
            logger.error(f"Failed to discover steps: {str(e)}")
            return []
    
    def _has_script_file(self, step_name: str) -> bool:
        """
        Simple file existence validation - no complex validator classes.
        
        Args:
            step_name: Step name to validate
            
        Returns:
            True if step has script file, False otherwise
        """
        try:
            step_info = self.step_catalog.get_step_info(step_name)
            return (step_info is not None and 
                    step_info.file_components.get('script') is not None)
        except Exception as e:
            logger.debug(f"File validation failed for {step_name}: {e}")
            return False
    
    def _run_validation_level(self, step_name: str, level: ValidationLevel, ruleset) -> Dict[str, Any]:
        """
        Run a specific validation level (replaces 4 separate level methods).
        
        Args:
            step_name: Name of the step to validate
            level: Validation level to run
            ruleset: Validation ruleset for the step type
            
        Returns:
            Dictionary containing validation results for the level
        """
        logger.debug(f"Running Level {level.value} validation for {step_name}")
        
        try:
            if level == ValidationLevel.SCRIPT_CONTRACT:
                return self.level_validators.run_level_1_validation(step_name)
            elif level == ValidationLevel.CONTRACT_SPEC:
                return self.level_validators.run_level_2_validation(step_name)
            elif level == ValidationLevel.SPEC_DEPENDENCY:
                return self.level_validators.run_level_3_validation(step_name)  # Universal
            elif level == ValidationLevel.BUILDER_CONFIG:
                validator_class = ruleset.level_4_validator_class if ruleset else None
                return self.level_validators.run_level_4_validation(step_name, validator_class)
            else:
                raise ValueError(f"Invalid validation level: {level}")
                
        except Exception as e:
            logger.error(f"Level {level.value} validation failed for {step_name}: {str(e)}")
            return {
                "level": level.value,
                "step_name": step_name,
                "status": "ERROR",
                "error": str(e)
            }
    
    def _run_enabled_validation_levels(self, step_name: str, sagemaker_step_type: str, ruleset) -> Dict[str, Any]:
        """
        Run all enabled validation levels for a step.
        
        Args:
            step_name: Name of the step to validate
            sagemaker_step_type: SageMaker step type
            ruleset: Validation ruleset for the step type
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            "step_name": step_name,
            "sagemaker_step_type": sagemaker_step_type,
            "category": ruleset.category.value if ruleset else "unknown",
            "enabled_levels": [level.value for level in ruleset.enabled_levels] if ruleset else [],
            "validation_results": {}
        }
        
        if not ruleset:
            results["status"] = "ERROR"
            results["error"] = f"No validation ruleset found for step type: {sagemaker_step_type}"
            return results
        
        # Run only enabled validation levels (key performance optimization)
        for level in ValidationLevel:
            if level in ruleset.enabled_levels:
                level_result = self._run_validation_level(step_name, level, ruleset)
                results["validation_results"][f"level_{level.value}"] = level_result
            else:
                # Log skipped levels for transparency
                logger.debug(f"Skipping Level {level.value} for {step_name} (not enabled for {sagemaker_step_type})")
        
        # Determine overall status
        has_errors = any(
            level_result.get("status") == "ERROR" 
            for level_result in results["validation_results"].values()
        )
        results["overall_status"] = "FAILED" if has_errors else "PASSED"
        
        return results
    
    def _handle_excluded_step(self, step_name: str, sagemaker_step_type: str, ruleset) -> Dict[str, Any]:
        """
        Handle excluded step types.
        
        Args:
            step_name: Name of the step
            sagemaker_step_type: SageMaker step type
            ruleset: Validation ruleset for the step type
            
        Returns:
            Dictionary indicating step is excluded
        """
        logger.info(f"Step {step_name} excluded from validation (type: {sagemaker_step_type})")
        
        return {
            "step_name": step_name,
            "sagemaker_step_type": sagemaker_step_type,
            "status": "EXCLUDED",
            "reason": ruleset.skip_reason if ruleset else f"Step type {sagemaker_step_type} is excluded",
            "category": ruleset.category.value if ruleset else "excluded"
        }
    
    # Preserve existing API methods for backward compatibility
    def validate_specific_script(self, step_name: str, 
                                skip_levels: Optional[Set[int]] = None) -> Dict[str, Any]:
        """
        Validate a specific script - maintained for backward compatibility.
        
        Args:
            step_name: Name of the step to validate
            skip_levels: Optional set of validation levels to skip (ignored in new system)
            
        Returns:
            Dictionary containing validation results
        """
        if skip_levels:
            logger.warning("skip_levels parameter is deprecated. Use configuration-driven validation instead.")
        
        return self.run_validation_for_step(step_name)
    
    def discover_scripts(self) -> List[str]:
        """
        Discover scripts - maintained for backward compatibility.
        
        Returns:
            List of discovered script names (only steps with actual script files)
        """
        logger.info("Discovering scripts (steps with script files) using step catalog")
        
        try:
            # Get all available steps
            all_steps = self.step_catalog.list_available_steps()
            
            # Filter to only include steps that have script files
            scripts_with_files = []
            for step_name in all_steps:
                if self._has_script_file(step_name):
                    scripts_with_files.append(step_name)
                else:
                    logger.debug(f"Skipping {step_name} - no script file found")
            
            logger.info(f"Discovered {len(scripts_with_files)} scripts with files out of {len(all_steps)} total steps")
            return scripts_with_files
            
        except Exception as e:
            logger.error(f"Failed to discover scripts: {str(e)}")
            return []
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get validation summary - enhanced with step-type-aware metrics.
        
        Returns:
            Dictionary containing validation summary
        """
        logger.info("Generating enhanced validation summary")
        
        # Run validation for all steps
        all_results = self.run_validation_for_all_steps()
        
        # Generate enhanced summary statistics
        total_steps = len(all_results)
        passed_steps = 0
        failed_steps = 0
        excluded_steps = 0
        step_type_breakdown = {}
        
        for step_name, result in all_results.items():
            step_type = result.get("sagemaker_step_type", "unknown")
            
            # Count by step type
            if step_type not in step_type_breakdown:
                step_type_breakdown[step_type] = {"total": 0, "passed": 0, "failed": 0, "excluded": 0}
            step_type_breakdown[step_type]["total"] += 1
            
            # Count overall status
            status = result.get("overall_status") or result.get("status")
            if status == "EXCLUDED":
                excluded_steps += 1
                step_type_breakdown[step_type]["excluded"] += 1
            elif status == "PASSED":
                passed_steps += 1
                step_type_breakdown[step_type]["passed"] += 1
            else:
                failed_steps += 1
                step_type_breakdown[step_type]["failed"] += 1
        
        return {
            "total_steps": total_steps,
            "passed_steps": passed_steps,
            "failed_steps": failed_steps,
            "excluded_steps": excluded_steps,
            "pass_rate": passed_steps / (total_steps - excluded_steps) if (total_steps - excluded_steps) > 0 else 0,
            "step_type_breakdown": step_type_breakdown,
            "configuration_driven": True,
            "detailed_results": all_results
        }
    
    def export_report(self, format: str = "json", output_path: Optional[str] = None) -> str:
        """
        Export validation report - enhanced with configuration insights.
        
        Args:
            format: Report format ("json" or "html")
            output_path: Optional output file path
            
        Returns:
            Report content as string
        """
        logger.info(f"Exporting enhanced report in {format} format")
        
        summary = self.get_validation_summary()
        
        if format == "json":
            import json
            report_content = json.dumps(summary, indent=2, default=str)
        else:
            # Enhanced text format with step type breakdown
            report_content = f"Enhanced Validation Summary:\n"
            report_content += f"Total Steps: {summary['total_steps']}\n"
            report_content += f"Passed: {summary['passed_steps']}\n"
            report_content += f"Failed: {summary['failed_steps']}\n"
            report_content += f"Excluded: {summary['excluded_steps']}\n"
            report_content += f"Pass Rate: {summary['pass_rate']:.2%}\n\n"
            
            report_content += "Step Type Breakdown:\n"
            for step_type, breakdown in summary['step_type_breakdown'].items():
                report_content += f"  {step_type}: {breakdown['passed']}/{breakdown['total']} passed"
                if breakdown['excluded'] > 0:
                    report_content += f" ({breakdown['excluded']} excluded)"
                report_content += "\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Enhanced report exported to {output_path}")
        
        return report_content
    
    def print_summary(self):
        """Print enhanced validation summary to console."""
        summary = self.get_validation_summary()
        
        print("\n" + "="*60)
        print("ENHANCED VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Passed: {summary['passed_steps']}")
        print(f"Failed: {summary['failed_steps']}")
        print(f"Excluded: {summary['excluded_steps']}")
        print(f"Pass Rate: {summary['pass_rate']:.2%}")
        print(f"Configuration-Driven: {summary['configuration_driven']}")
        
        print("\nStep Type Breakdown:")
        for step_type, breakdown in summary['step_type_breakdown'].items():
            status_str = f"{breakdown['passed']}/{breakdown['total']} passed"
            if breakdown['excluded'] > 0:
                status_str += f" ({breakdown['excluded']} excluded)"
            print(f"  {step_type}: {status_str}")
        
        print("="*60 + "\n")
    
    def get_critical_issues(self) -> List[Dict[str, Any]]:
        """
        Get critical validation issues - step-type-aware critical issue analysis.
        
        Returns:
            List of critical issues
        """
        logger.info("Identifying critical issues with step-type awareness")
        
        all_results = self.run_validation_for_all_steps()
        critical_issues = []
        
        for step_name, result in all_results.items():
            step_type = result.get("sagemaker_step_type", "unknown")
            validation_results = result.get("validation_results", {})
            
            for level, level_result in validation_results.items():
                if level_result.get("status") == "ERROR":
                    critical_issues.append({
                        "step_name": step_name,
                        "step_type": step_type,
                        "level": level,
                        "error": level_result.get("error", "Unknown error"),
                        "category": result.get("category", "unknown")
                    })
        
        return critical_issues
    
    def get_step_info_from_catalog(self, step_name: str) -> Optional[Any]:
        """
        Get step information from step catalog - maintained for backward compatibility.
        
        Args:
            step_name: Name of the step
            
        Returns:
            StepInfo object or None if not found
        """
        try:
            return self.step_catalog.get_step_info(step_name)
        except Exception as e:
            logger.error(f"Error getting step info for {step_name}: {str(e)}")
            return None
    
    def get_component_path_from_catalog(self, step_name: str, component_type: str) -> Optional[Path]:
        """
        Get component file path from step catalog - maintained for backward compatibility.
        
        Args:
            step_name: Name of the step
            component_type: Type of component ('script', 'contract', 'spec', 'builder', 'config')
            
        Returns:
            Path to component file or None if not found
        """
        try:
            step_info = self.get_step_info_from_catalog(step_name)
            if step_info and step_info.file_components.get(component_type):
                return step_info.file_components[component_type].path
            return None
        except Exception as e:
            logger.error(f"Error getting {component_type} path for {step_name}: {str(e)}")
            return None
    
    def validate_cross_workspace_compatibility(self, step_names: List[str]) -> Dict[str, Any]:
        """
        Validate compatibility across workspace components - simplified with configuration.
        
        Args:
            step_names: List of step names to validate
            
        Returns:
            Compatibility validation results
        """
        results = {
            "compatible": True,
            "issues": [],
            "step_type_distribution": {},
            "recommendations": []
        }
        
        try:
            # Group steps by step type instead of workspace for simpler analysis
            step_type_groups = {}
            for step_name in step_names:
                try:
                    step_type = get_sagemaker_step_type(step_name)
                    if step_type not in step_type_groups:
                        step_type_groups[step_type] = []
                    step_type_groups[step_type].append(step_name)
                except Exception as e:
                    logger.warning(f"Could not determine step type for {step_name}: {str(e)}")
            
            results["step_type_distribution"] = step_type_groups
            
            # Check for potential compatibility issues based on step types
            if len(step_type_groups) > 3:
                results["recommendations"].append(
                    f"Multiple step types detected ({len(step_type_groups)}). Ensure validation levels are appropriate."
                )
            
        except Exception as e:
            results["issues"].append(f"Cross-workspace validation error: {str(e)}")
            results["compatible"] = False
        
        return results
