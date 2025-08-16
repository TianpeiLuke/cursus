"""
CreateModel Step Validation Test Orchestrator

This module provides the main orchestrator for CreateModel step validation,
integrating all 4 levels of testing:
- Level 1: Interface Tests (CreateModelInterfaceTests)
- Level 2: Specification Tests (CreateModelSpecificationTests)  
- Level 3: Path Mapping Tests (CreateModelPathMappingTests)
- Level 4: Integration Tests (CreateModelIntegrationTests)

The orchestrator provides convenience functions for running CreateModel-specific
validation suites and generating comprehensive reports.
"""

from typing import Dict, Any, List, Optional, Union
import logging

from .createmodel_interface_tests import CreateModelInterfaceTests
from .createmodel_specification_tests import CreateModelSpecificationTests
from ..step_creation_tests import StepCreationTests
from .createmodel_integration_tests import CreateModelIntegrationTests

logger = logging.getLogger(__name__)


class CreateModelStepBuilderTest:
    """Main orchestrator for CreateModel step validation testing."""
    
    def __init__(self, builder_instance, config: Dict[str, Any]):
        """
        Initialize CreateModel test orchestrator.
        
        Args:
            builder_instance: The CreateModel step builder instance to test
            config: Configuration dictionary for testing
        """
        self.builder_instance = builder_instance
        self.config = config
        self.step_type = "CreateModel"
        
        # Initialize all test levels
        self.interface_tests = CreateModelInterfaceTests(builder_instance, config)
        self.specification_tests = CreateModelSpecificationTests(builder_instance, config)
        self.path_mapping_tests = CreateModelPathMappingTests(builder_instance, config)
        self.integration_tests = CreateModelIntegrationTests(builder_instance, config)
        
        logger.info(f"Initialized CreateModel test orchestrator for builder: {type(builder_instance).__name__}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all CreateModel validation tests across all 4 levels.
        
        Returns:
            Dict containing comprehensive test results
        """
        logger.info("Running complete CreateModel validation test suite")
        
        results = {
            "step_type": self.step_type,
            "builder_type": type(self.builder_instance).__name__,
            "test_summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "overall_passed": True
            },
            "level_results": {}
        }
        
        # Run Level 1: Interface Tests
        logger.info("Running Level 1: CreateModel Interface Tests")
        level1_results = self.interface_tests.run_all_tests()
        results["level_results"]["level_1_interface"] = level1_results
        self._update_summary(results["test_summary"], level1_results)
        
        # Run Level 2: Specification Tests
        logger.info("Running Level 2: CreateModel Specification Tests")
        level2_results = self.specification_tests.run_all_tests()
        results["level_results"]["level_2_specification"] = level2_results
        self._update_summary(results["test_summary"], level2_results)
        
        # Run Level 3: Path Mapping Tests
        logger.info("Running Level 3: CreateModel Path Mapping Tests")
        level3_results = self.path_mapping_tests.run_all_tests()
        results["level_results"]["level_3_path_mapping"] = level3_results
        self._update_summary(results["test_summary"], level3_results)
        
        # Run Level 4: Integration Tests
        logger.info("Running Level 4: CreateModel Integration Tests")
        level4_results = self.integration_tests.run_all_tests()
        results["level_results"]["level_4_integration"] = level4_results
        self._update_summary(results["test_summary"], level4_results)
        
        # Finalize summary
        results["test_summary"]["overall_passed"] = (
            results["test_summary"]["failed_tests"] == 0
        )
        
        logger.info(f"CreateModel validation complete. "
                   f"Passed: {results['test_summary']['passed_tests']}, "
                   f"Failed: {results['test_summary']['failed_tests']}")
        
        return results
    
    def run_interface_tests(self) -> Dict[str, Any]:
        """Run only Level 1 CreateModel interface tests."""
        logger.info("Running CreateModel Interface Tests (Level 1)")
        return self.interface_tests.run_all_tests()
    
    def run_specification_tests(self) -> Dict[str, Any]:
        """Run only Level 2 CreateModel specification tests."""
        logger.info("Running CreateModel Specification Tests (Level 2)")
        return self.specification_tests.run_all_tests()
    
    def run_path_mapping_tests(self) -> Dict[str, Any]:
        """Run only Level 3 CreateModel path mapping tests."""
        logger.info("Running CreateModel Path Mapping Tests (Level 3)")
        return self.path_mapping_tests.run_all_tests()
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run only Level 4 CreateModel integration tests."""
        logger.info("Running CreateModel Integration Tests (Level 4)")
        return self.integration_tests.run_all_tests()
    
    def run_framework_specific_tests(self, framework: str) -> Dict[str, Any]:
        """
        Run CreateModel tests specific to a particular ML framework.
        
        Args:
            framework: The ML framework to test ('pytorch', 'xgboost', 'tensorflow', 'sklearn')
            
        Returns:
            Dict containing framework-specific test results
        """
        logger.info(f"Running CreateModel tests for framework: {framework}")
        
        results = {
            "step_type": self.step_type,
            "framework": framework,
            "builder_type": type(self.builder_instance).__name__,
            "framework_tests": {}
        }
        
        # Run framework-specific interface tests
        if hasattr(self.interface_tests, f'test_{framework}_specific_methods'):
            framework_interface = getattr(self.interface_tests, f'test_{framework}_specific_methods')()
            results["framework_tests"]["interface"] = framework_interface
        
        # Run framework-specific specification tests
        if hasattr(self.specification_tests, f'test_{framework}_configuration'):
            framework_spec = getattr(self.specification_tests, f'test_{framework}_configuration')()
            results["framework_tests"]["specification"] = framework_spec
        
        # Run framework-specific path mapping tests
        if hasattr(self.path_mapping_tests, f'test_{framework}_model_paths'):
            framework_paths = getattr(self.path_mapping_tests, f'test_{framework}_model_paths')()
            results["framework_tests"]["path_mapping"] = framework_paths
        
        # Run framework-specific integration tests
        if hasattr(self.integration_tests, f'test_{framework}_deployment_pattern'):
            framework_integration = getattr(self.integration_tests, f'test_{framework}_deployment_pattern')()
            results["framework_tests"]["integration"] = framework_integration
        
        return results
    
    def run_deployment_readiness_tests(self) -> Dict[str, Any]:
        """
        Run CreateModel deployment readiness validation tests.
        
        Returns:
            Dict containing deployment readiness test results
        """
        logger.info("Running CreateModel deployment readiness tests")
        
        results = {
            "step_type": self.step_type,
            "test_type": "deployment_readiness",
            "builder_type": type(self.builder_instance).__name__,
            "readiness_tests": {}
        }
        
        # Run container configuration tests
        if hasattr(self.specification_tests, 'test_container_configuration_validation'):
            container_test = self.specification_tests.test_container_configuration_validation()
            results["readiness_tests"]["container_configuration"] = container_test
        
        # Run model artifact validation
        if hasattr(self.path_mapping_tests, 'test_model_artifact_path_mapping'):
            artifact_test = self.path_mapping_tests.test_model_artifact_path_mapping()
            results["readiness_tests"]["model_artifacts"] = artifact_test
        
        # Run inference endpoint preparation
        if hasattr(self.integration_tests, 'test_inference_endpoint_preparation'):
            endpoint_test = self.integration_tests.test_inference_endpoint_preparation()
            results["readiness_tests"]["endpoint_preparation"] = endpoint_test
        
        # Run production deployment readiness
        if hasattr(self.integration_tests, 'test_production_deployment_readiness'):
            production_test = self.integration_tests.test_production_deployment_readiness()
            results["readiness_tests"]["production_readiness"] = production_test
        
        return results
    
    def run_model_registry_tests(self) -> Dict[str, Any]:
        """
        Run CreateModel registry integration tests.
        
        Returns:
            Dict containing model registry test results
        """
        logger.info("Running CreateModel model registry tests")
        
        results = {
            "step_type": self.step_type,
            "test_type": "model_registry",
            "builder_type": type(self.builder_instance).__name__,
            "registry_tests": {}
        }
        
        # Run model registry specification tests
        if hasattr(self.specification_tests, 'test_model_registry_specification'):
            registry_spec = self.specification_tests.test_model_registry_specification()
            results["registry_tests"]["specification"] = registry_spec
        
        # Run model registry path integration
        if hasattr(self.path_mapping_tests, 'test_model_registry_path_integration'):
            registry_paths = self.path_mapping_tests.test_model_registry_path_integration()
            results["registry_tests"]["path_integration"] = registry_paths
        
        # Run model registry workflow integration
        if hasattr(self.integration_tests, 'test_model_registry_integration_workflow'):
            registry_workflow = self.integration_tests.test_model_registry_integration_workflow()
            results["registry_tests"]["workflow_integration"] = registry_workflow
        
        # Run model versioning integration
        if hasattr(self.integration_tests, 'test_model_versioning_integration'):
            versioning_test = self.integration_tests.test_model_versioning_integration()
            results["registry_tests"]["versioning"] = versioning_test
        
        return results
    
    def run_multi_container_tests(self) -> Dict[str, Any]:
        """
        Run CreateModel multi-container deployment tests.
        
        Returns:
            Dict containing multi-container test results
        """
        logger.info("Running CreateModel multi-container tests")
        
        results = {
            "step_type": self.step_type,
            "test_type": "multi_container",
            "builder_type": type(self.builder_instance).__name__,
            "multi_container_tests": {}
        }
        
        # Run multi-container specification tests
        if hasattr(self.specification_tests, 'test_multi_container_specification'):
            multi_spec = self.specification_tests.test_multi_container_specification()
            results["multi_container_tests"]["specification"] = multi_spec
        
        # Run multi-container deployment tests
        if hasattr(self.integration_tests, 'test_multi_container_model_deployment'):
            multi_deployment = self.integration_tests.test_multi_container_model_deployment()
            results["multi_container_tests"]["deployment"] = multi_deployment
        
        return results
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """
        Run CreateModel performance optimization tests.
        
        Returns:
            Dict containing performance test results
        """
        logger.info("Running CreateModel performance tests")
        
        results = {
            "step_type": self.step_type,
            "test_type": "performance",
            "builder_type": type(self.builder_instance).__name__,
            "performance_tests": {}
        }
        
        # Run container optimization tests
        if hasattr(self.integration_tests, 'test_container_optimization_validation'):
            optimization_test = self.integration_tests.test_container_optimization_validation()
            results["performance_tests"]["container_optimization"] = optimization_test
        
        # Run inference performance tests
        if hasattr(self.path_mapping_tests, 'test_inference_environment_path_mapping'):
            inference_test = self.path_mapping_tests.test_inference_environment_path_mapping()
            results["performance_tests"]["inference_environment"] = inference_test
        
        return results
    
    def generate_createmodel_report(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive CreateModel validation report.
        
        Args:
            test_results: Results from running CreateModel tests
            
        Returns:
            Dict containing formatted report
        """
        logger.info("Generating CreateModel validation report")
        
        report = {
            "report_type": "CreateModel Validation Report",
            "step_type": self.step_type,
            "builder_type": type(self.builder_instance).__name__,
            "timestamp": self._get_timestamp(),
            "summary": self._generate_summary(test_results),
            "detailed_results": test_results,
            "recommendations": self._generate_recommendations(test_results),
            "framework_analysis": self._analyze_framework_compatibility(test_results),
            "deployment_readiness": self._assess_deployment_readiness(test_results)
        }
        
        return report
    
    def get_createmodel_test_coverage(self) -> Dict[str, Any]:
        """
        Get CreateModel test coverage information.
        
        Returns:
            Dict containing test coverage details
        """
        coverage = {
            "step_type": self.step_type,
            "coverage_analysis": {
                "level_1_interface": {
                    "total_tests": len(self.interface_tests.get_step_type_specific_tests()),
                    "test_categories": [
                        "model_creation_methods",
                        "container_configuration",
                        "deployment_preparation",
                        "framework_specific_methods"
                    ]
                },
                "level_2_specification": {
                    "total_tests": len(self.specification_tests.get_step_type_specific_tests()),
                    "test_categories": [
                        "container_validation",
                        "framework_configuration",
                        "inference_environment",
                        "model_registry_integration"
                    ]
                },
                "level_3_path_mapping": {
                    "total_tests": len(self.path_mapping_tests.get_step_type_specific_tests()),
                    "test_categories": [
                        "model_artifact_paths",
                        "container_image_paths",
                        "inference_code_paths",
                        "deployment_configuration_paths"
                    ]
                },
                "level_4_integration": {
                    "total_tests": len(self.integration_tests.get_step_type_specific_tests()),
                    "test_categories": [
                        "complete_step_creation",
                        "framework_deployment_patterns",
                        "model_registry_workflows",
                        "production_readiness"
                    ]
                }
            },
            "framework_support": [
                "pytorch",
                "xgboost", 
                "tensorflow",
                "sklearn",
                "custom"
            ],
            "deployment_patterns": [
                "single_container",
                "multi_container",
                "model_registry_integration",
                "endpoint_deployment"
            ]
        }
        
        total_tests = sum(
            level["total_tests"] 
            for level in coverage["coverage_analysis"].values()
        )
        coverage["total_test_count"] = total_tests
        
        return coverage
    
    # Helper methods
    
    def _update_summary(self, summary: Dict[str, Any], level_results: Dict[str, Any]) -> None:
        """Update test summary with level results."""
        if "test_results" in level_results:
            for test_result in level_results["test_results"]:
                summary["total_tests"] += 1
                if test_result.get("passed", False):
                    summary["passed_tests"] += 1
                else:
                    summary["failed_tests"] += 1
    
    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test results summary."""
        if "test_summary" in test_results:
            return test_results["test_summary"]
        
        return {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "overall_passed": True
        }
    
    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Analyze failed tests and generate recommendations
        if "level_results" in test_results:
            for level_name, level_result in test_results["level_results"].items():
                if "test_results" in level_result:
                    for test in level_result["test_results"]:
                        if not test.get("passed", True) and "errors" in test:
                            for error in test["errors"]:
                                if "container" in error.lower():
                                    recommendations.append("Review container configuration and image specifications")
                                elif "model" in error.lower():
                                    recommendations.append("Validate model artifact paths and accessibility")
                                elif "framework" in error.lower():
                                    recommendations.append("Check framework-specific configuration requirements")
                                elif "deployment" in error.lower():
                                    recommendations.append("Verify deployment readiness and endpoint configuration")
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("All CreateModel validation tests passed successfully")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _analyze_framework_compatibility(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze framework compatibility from test results."""
        framework_analysis = {
            "detected_framework": "unknown",
            "compatibility_status": "unknown",
            "framework_specific_issues": []
        }
        
        # Try to detect framework from test results
        if "level_results" in test_results:
            for level_result in test_results["level_results"].values():
                if "test_results" in level_result:
                    for test in level_result["test_results"]:
                        if "details" in test and "framework" in test["details"]:
                            framework_analysis["detected_framework"] = test["details"]["framework"]
                            break
        
        return framework_analysis
    
    def _assess_deployment_readiness(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess deployment readiness from test results."""
        readiness = {
            "ready_for_deployment": True,
            "readiness_score": 100,
            "blocking_issues": [],
            "warnings": []
        }
        
        # Analyze test results for deployment readiness
        if "test_summary" in test_results:
            if test_results["test_summary"]["failed_tests"] > 0:
                readiness["ready_for_deployment"] = False
                readiness["readiness_score"] = max(0, 100 - (test_results["test_summary"]["failed_tests"] * 10))
        
        return readiness
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for reporting."""
        from datetime import datetime
        return datetime.now().isoformat()


# Convenience functions for CreateModel testing

def run_createmodel_validation(builder_instance, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to run complete CreateModel validation.
    
    Args:
        builder_instance: CreateModel step builder instance
        config: Optional configuration dictionary
        
    Returns:
        Dict containing complete validation results
    """
    if config is None:
        config = {}
    
    test_orchestrator = CreateModelTest(builder_instance, config)
    return test_orchestrator.run_all_tests()


def run_createmodel_framework_tests(builder_instance, framework: str, 
                                   config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to run framework-specific CreateModel tests.
    
    Args:
        builder_instance: CreateModel step builder instance
        framework: ML framework to test
        config: Optional configuration dictionary
        
    Returns:
        Dict containing framework-specific test results
    """
    if config is None:
        config = {}
    
    test_orchestrator = CreateModelTest(builder_instance, config)
    return test_orchestrator.run_framework_specific_tests(framework)


def generate_createmodel_report(builder_instance, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to generate CreateModel validation report.
    
    Args:
        builder_instance: CreateModel step builder instance
        config: Optional configuration dictionary
        
    Returns:
        Dict containing comprehensive validation report
    """
    if config is None:
        config = {}
    
    test_orchestrator = CreateModelTest(builder_instance, config)
    test_results = test_orchestrator.run_all_tests()
    return test_orchestrator.generate_createmodel_report(test_results)
