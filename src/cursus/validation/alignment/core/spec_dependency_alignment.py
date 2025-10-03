"""
Specification ↔ Dependencies Alignment Tester

Validates alignment between step specifications and their dependency declarations.
Ensures dependency chains are consistent and resolvable.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .level3_validation_config import Level3ValidationConfig, ValidationMode
from ..validators import DependencyValidator
from ....core.deps.factory import create_pipeline_components
from ....core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
)
from ....registry.step_names import (
    get_step_name_from_spec_type,
    get_canonical_name_from_file_name,
)

logger = logging.getLogger(__name__)


class SpecificationDependencyAlignmentTester:
    """
    Tests alignment between step specifications and their dependencies.

    Validates:
    - Dependency chains are consistent
    - All dependencies can be resolved
    - No circular dependencies exist
    - Data types match across dependency chains
    """

    def __init__(
        self, validation_config: Level3ValidationConfig = None, workspace_dirs: Optional[List[Path]] = None
    ):
        """
        Initialize the specification-dependency alignment tester.

        Args:
            validation_config: Configuration for validation thresholds and behavior
            workspace_dirs: Optional list of workspace directories for workspace-aware discovery
        """
        self.config = (
            validation_config or Level3ValidationConfig.create_relaxed_config()
        )

        # Store workspace directories
        self.workspace_dirs = workspace_dirs
        
        # Initialize StepCatalog with workspace-aware discovery
        from ....step_catalog import StepCatalog
        self.step_catalog = StepCatalog(workspace_dirs=workspace_dirs)
        self.dependency_validator = DependencyValidator(self.config)


        # Initialize dependency resolver components
        self.pipeline_components = create_pipeline_components("level3_validation")
        self.dependency_resolver = self.pipeline_components["resolver"]
        self.spec_registry = self.pipeline_components["registry"]

        # Log configuration
        threshold_desc = self.config.get_threshold_description()
        logger.info(
            f"Level 3 validation initialized with {threshold_desc['mode']} mode"
        )
        logger.debug(f"Thresholds: {threshold_desc['thresholds']}")

    def validate_all_specifications(
        self, target_scripts: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate alignment for all specifications or specified target scripts.
        
        This method uses StepCatalog's bulk loading for efficiency.

        Args:
            target_scripts: Specific scripts to validate (None for all)

        Returns:
            Dictionary mapping specification names to validation results
        """
        results = {}

        # Load all specifications at once for efficiency
        try:
            all_specs = self.step_catalog.load_all_specifications()
        except Exception as e:
            logger.error(f"Failed to load specifications via StepCatalog: {e}")
            # Fallback to individual loading
            return self._validate_all_specifications_fallback(target_scripts)

        # Filter to target scripts if specified
        if target_scripts:
            specs_to_validate = {name: spec for name, spec in all_specs.items() 
                               if name in target_scripts}
        else:
            specs_to_validate = all_specs

        # Validate each specification using the object-based method
        for spec_name, spec_dict in specs_to_validate.items():
            try:
                result = self.validate_specification_object(spec_dict, spec_name)
                results[spec_name] = result
            except Exception as e:
                results[spec_name] = {
                    "passed": False,
                    "error": str(e),
                    "issues": [
                        {
                            "severity": "CRITICAL",
                            "category": "validation_error",
                            "message": f"Failed to validate specification {spec_name}: {str(e)}",
                        }
                    ],
                }

        return results
    
    def _validate_all_specifications_fallback(
        self, target_scripts: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Fallback method using individual specification loading."""
        results = {}
        
        # Discover specifications to validate
        if target_scripts:
            specs_to_validate = target_scripts
        else:
            specs_to_validate = self.step_catalog.list_steps_with_specs()

        for spec_name in specs_to_validate:
            try:
                result = self.validate_specification(spec_name)
                results[spec_name] = result
            except Exception as e:
                results[spec_name] = {
                    "passed": False,
                    "error": str(e),
                    "issues": [
                        {
                            "severity": "CRITICAL",
                            "category": "validation_error",
                            "message": f"Failed to validate specification {spec_name}: {str(e)}",
                        }
                    ],
                }

        return results

    def validate_specification(self, spec_name: str) -> Dict[str, Any]:
        """
        Validate alignment for a specific specification.

        Args:
            spec_name: Name of the specification to validate

        Returns:
            Validation result dictionary
        """
        # Load specification using StepCatalog with built-in error handling
        try:
            spec_obj = self.step_catalog.load_spec_class(spec_name)
        except Exception as e:
            return {
                "passed": False,
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "category": "spec_loading_error",
                        "message": f"Failed to load specification for {spec_name}: {str(e)}",
                        "details": {
                            "spec_name": spec_name,
                            "error": str(e),
                        },
                        "recommendation": f"Check specification file for {spec_name} or StepCatalog configuration",
                    }
                ],
            }
        
        if not spec_obj:
            return self._create_missing_spec_error(spec_name)
        
        # Serialize specification
        try:
            specification = self.step_catalog.serialize_spec(spec_obj)
        except Exception as e:
            return self._create_serialization_error(spec_name, str(e))
        
        # Perform validation using the simplified validation method
        return self.validate_specification_object(specification, spec_name)
    
    def validate_specification_object(self, specification: Dict[str, Any], spec_name: str = None) -> Dict[str, Any]:
        """
        Validate a pre-loaded specification object.
        
        Args:
            specification: Serialized specification dictionary
            spec_name: Optional specification name for context
            
        Returns:
            Validation result dictionary
        """
        # Load all specifications for dependency resolution (cached by StepCatalog)
        all_specs = self._load_all_specifications()
        
        # Perform alignment validation
        issues = []
        
        # Validate dependency resolution
        resolution_issues = self._validate_dependency_resolution(
            specification, all_specs, spec_name or "unknown"
        )
        issues.extend(resolution_issues)
        
        # Validate circular dependencies
        circular_issues = self._validate_circular_dependencies(
            specification, all_specs, spec_name or "unknown"
        )
        issues.extend(circular_issues)
        
        # Validate data type consistency
        type_issues = self._validate_dependency_data_types(
            specification, all_specs, spec_name or "unknown"
        )
        issues.extend(type_issues)
        
        # Determine overall pass/fail status
        has_critical_or_error = any(
            issue["severity"] in ["CRITICAL", "ERROR"] for issue in issues
        )
        
        return {
            "passed": not has_critical_or_error,
            "issues": issues,
            "specification": specification,
        }
    
    def _create_missing_spec_error(self, spec_name: str) -> Dict[str, Any]:
        """Create standardized error response for missing specifications."""
        return {
            "passed": False,
            "issues": [
                {
                    "severity": "CRITICAL",
                    "category": "spec_not_found",
                    "message": f"No specification found for {spec_name} via StepCatalog",
                    "details": {
                        "spec_name": spec_name,
                        "discovery_method": "StepCatalog.load_spec_class()",
                    },
                    "recommendation": f"Create specification for {spec_name} or check StepCatalog configuration",
                }
            ],
        }
    
    def _create_serialization_error(self, spec_name: str, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response for serialization failures."""
        return {
            "passed": False,
            "issues": [
                {
                    "severity": "CRITICAL",
                    "category": "spec_serialization_error",
                    "message": f"Failed to serialize specification for {spec_name}: {error_msg}",
                    "details": {
                        "spec_name": spec_name,
                        "error": error_msg,
                    },
                    "recommendation": "Fix specification structure or StepCatalog serialization",
                }
            ],
        }

    def _validate_dependency_resolution(
        self,
        specification: Dict[str, Any],
        all_specs: Dict[str, Dict[str, Any]],
        spec_name: str,
    ) -> List[Dict[str, Any]]:
        """Enhanced dependency validation with compatibility scoring using extracted component."""
        return self.dependency_validator.validate_dependency_resolution(
            specification, all_specs, spec_name
        )

    def _generate_compatibility_recommendation(
        self, dep_name: str, best_candidate: Dict
    ) -> str:
        """Generate specific recommendations based on compatibility analysis."""
        if "score_breakdown" not in best_candidate:
            return f"Review dependency specification for {dep_name} and output specification for {best_candidate['output_name']}"

        score_breakdown = best_candidate["score_breakdown"]
        recommendations = []

        if score_breakdown.get("type_compatibility", 0) < 0.2:
            recommendations.append(
                f"Consider changing dependency type or output type for better compatibility"
            )

        if score_breakdown.get("semantic_similarity", 0) < 0.15:
            recommendations.append(
                f"Consider renaming '{dep_name}' or adding aliases to improve semantic matching"
            )

        if score_breakdown.get("source_compatibility", 0) < 0.05:
            recommendations.append(
                f"Add '{best_candidate['provider_step']}' to compatible_sources for {dep_name}"
            )

        if score_breakdown.get("data_type_compatibility", 0) < 0.1:
            recommendations.append(
                f"Align data types between dependency and output specifications"
            )

        if not recommendations:
            recommendations.append(
                f"Review dependency specification for {dep_name} and output specification for {best_candidate['output_name']}"
            )

        return "; ".join(recommendations)

    def _get_available_canonical_step_names(
        self, all_specs: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Get available canonical step names using the registry as single source of truth.

        This method queries the production registry to get the authoritative list of
        canonical step names, ensuring alignment with production dependency resolution.

        Args:
            all_specs: Dictionary of all loaded specifications

        Returns:
            List of canonical step names from the production registry
        """
        from ....registry.step_names import get_all_step_names

        # Get canonical step names from the production registry (single source of truth)
        canonical_names = get_all_step_names()

        logger.debug(f"Available canonical step names from registry: {canonical_names}")
        return canonical_names


    def _populate_resolver_registry(self, all_specs: Dict[str, Dict[str, Any]]):
        """Populate the dependency resolver registry with all specifications using canonical names."""
        for spec_name, spec_dict in all_specs.items():
            try:
                # Convert file-based spec name to canonical step name using registry
                canonical_name = get_canonical_name_from_file_name(spec_name)

                # Convert dict back to StepSpecification object
                step_spec = self._dict_to_step_specification(spec_dict)

                # Register with canonical name
                self.dependency_resolver.register_specification(
                    canonical_name, step_spec
                )
                logger.debug(
                    f"Registered specification: '{spec_name}' as canonical '{canonical_name}'"
                )

            except Exception as e:
                logger.warning(f"Failed to register {spec_name} with resolver: {e}")

    def _dict_to_step_specification(
        self, spec_dict: Dict[str, Any]
    ) -> StepSpecification:
        """Convert specification dictionary back to StepSpecification object."""
        # Convert dependencies
        dependencies = {}
        for dep in spec_dict.get("dependencies", []):
            # Create DependencySpec using keyword arguments
            dep_data = {
                "logical_name": dep["logical_name"],
                "dependency_type": dep[
                    "dependency_type"
                ],  # Keep as string, validator will convert
                "required": dep["required"],
                "compatible_sources": dep.get("compatible_sources", []),
                "data_type": dep["data_type"],
                "description": dep.get("description", ""),
                "semantic_keywords": dep.get("semantic_keywords", []),
            }
            dep_spec = DependencySpec(**dep_data)
            dependencies[dep["logical_name"]] = dep_spec

        # Convert outputs
        outputs = {}
        for out in spec_dict.get("outputs", []):
            # Create OutputSpec using keyword arguments
            out_data = {
                "logical_name": out["logical_name"],
                "output_type": out[
                    "output_type"
                ],  # Keep as string, validator will convert
                "property_path": out["property_path"],
                "data_type": out["data_type"],
                "description": out.get("description", ""),
                "aliases": out.get("aliases", []),
            }
            out_spec = OutputSpec(**out_data)
            outputs[out["logical_name"]] = out_spec

        # Create StepSpecification using keyword arguments
        spec_data = {
            "step_type": spec_dict["step_type"],
            "node_type": spec_dict[
                "node_type"
            ],  # Keep as string, validator will convert
            "dependencies": dependencies,
            "outputs": outputs,
        }
        return StepSpecification(**spec_data)

    def get_dependency_resolution_report(
        self, all_specs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate detailed dependency resolution report using production resolver."""
        self._populate_resolver_registry(all_specs)

        # Convert file-based spec names to canonical names for the report
        available_steps = []
        for spec_name in all_specs.keys():
            try:
                canonical_name = get_canonical_name_from_file_name(spec_name)
                available_steps.append(canonical_name)
            except Exception as e:
                logger.warning(f"Could not get canonical name for {spec_name}: {e}")
                available_steps.append(spec_name)  # Fallback to file name

        return self.dependency_resolver.get_resolution_report(available_steps)

    def _is_compatible_output(
        self, required_logical_name: str, output_logical_name: str
    ) -> bool:
        """Check if an output logical name is compatible with a required logical name using flexible matching."""
        if not required_logical_name or not output_logical_name:
            return False

        # Exact match
        if required_logical_name == output_logical_name:
            return True

        # Common data input/output patterns
        data_patterns = {
            "data_input": [
                "processed_data",
                "training_data",
                "input_data",
                "data",
                "model_input_data",
            ],
            "input_data": [
                "processed_data",
                "training_data",
                "data_input",
                "data",
                "model_input_data",
            ],
            "training_data": [
                "processed_data",
                "data_input",
                "input_data",
                "data",
                "model_input_data",
            ],
            "processed_data": [
                "data_input",
                "input_data",
                "training_data",
                "data",
                "model_input_data",
            ],
            "model_input_data": [
                "processed_data",
                "data_input",
                "input_data",
                "training_data",
                "data",
            ],
            "data": [
                "processed_data",
                "data_input",
                "input_data",
                "training_data",
                "model_input_data",
            ],
        }

        # Check if required name has compatible patterns
        compatible_outputs = data_patterns.get(required_logical_name.lower(), [])
        if output_logical_name.lower() in compatible_outputs:
            return True

        # Check reverse mapping
        for pattern_key, pattern_values in data_patterns.items():
            if (
                output_logical_name.lower() == pattern_key
                and required_logical_name.lower() in pattern_values
            ):
                return True

        return False

    def _validate_circular_dependencies(
        self,
        specification: Dict[str, Any],
        all_specs: Dict[str, Dict[str, Any]],
        spec_name: str,
    ) -> List[Dict[str, Any]]:
        """Validate that no circular dependencies exist using extracted component."""
        return self.dependency_validator.validate_circular_dependencies(
            specification, all_specs, spec_name
        )

    def _validate_dependency_data_types(
        self,
        specification: Dict[str, Any],
        all_specs: Dict[str, Dict[str, Any]],
        spec_name: str,
    ) -> List[Dict[str, Any]]:
        """Validate data type consistency across dependency chains using extracted component."""
        return self.dependency_validator.validate_dependency_data_types(
            specification, all_specs, spec_name
        )



    def _load_all_specifications(self) -> Dict[str, Dict[str, Any]]:
        """Load all specification files using StepCatalog's load_all_specifications method."""
        try:
            # Use StepCatalog's dedicated load_all_specifications method
            all_specs = self.step_catalog.load_all_specifications()
            
            if all_specs:
                logger.info(f"Loaded {len(all_specs)} specifications using StepCatalog.load_all_specifications()")
                return all_specs
            else:
                logger.warning("StepCatalog.load_all_specifications() returned empty results")
                # Fallback to legacy file system scanning
                logger.warning("Falling back to legacy file system scanning")
                return self._load_all_specifications_legacy()
            
        except Exception as e:
            logger.error(f"StepCatalog.load_all_specifications() failed: {e}")
            
            # Fallback to legacy file system scanning if StepCatalog fails
            logger.warning("Falling back to legacy file system scanning")
            return self._load_all_specifications_legacy()
    
    def _load_all_specifications_legacy(self) -> Dict[str, Dict[str, Any]]:
        """Legacy fallback method - returns empty dict since we rely on StepCatalog."""
        logger.warning("Legacy fallback called - StepCatalog should handle all specification loading")
        return {}
