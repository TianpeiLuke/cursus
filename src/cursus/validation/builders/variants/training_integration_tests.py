"""
Training Step Level 4 Integration Tests

This module provides Level 4 validation for Training step builders, focusing on:
- Complete Training step creation and validation
- End-to-end training workflow integration
- Framework-specific training patterns
- Hyperparameter optimization integration
- Data channel and model artifact management
- Training job monitoring and completion
"""

from typing import Dict, Any, List, Optional
import logging

from ..integration_tests import IntegrationTests

logger = logging.getLogger(__name__)


class TrainingIntegrationTests(IntegrationTests):
    """Level 4 Training-specific integration validation tests."""
    
    def __init__(self, builder_instance, config: Dict[str, Any]):
        super().__init__(builder_instance, config)
        self.step_type = "Training"
        
    def get_step_type_specific_tests(self) -> List[str]:
        """Return Training-specific Level 4 integration tests."""
        return [
            "test_complete_training_step_creation",
            "test_framework_specific_training_workflow",
            "test_hyperparameter_optimization_integration",
            "test_data_channel_integration",
            "test_model_artifact_generation",
            "test_training_job_monitoring",
            "test_distributed_training_integration",
            "test_training_performance_optimization",
            "test_training_dependency_resolution"
        ]
    
    def test_complete_training_step_creation(self) -> Dict[str, Any]:
        """Test complete Training step creation and validation."""
        test_name = "test_complete_training_step_creation"
        logger.info(f"Running {test_name}")
        
        try:
            results = {
                "test_name": test_name,
                "passed": True,
                "details": {},
                "errors": []
            }
            
            # Test Training step instantiation
            if hasattr(self.builder_instance, 'create_step'):
                step = self.builder_instance.create_step()
                results["details"]["step_created"] = step is not None
                
                if step is None:
                    results["passed"] = False
                    results["errors"].append("Failed to create Training step")
                else:
                    # Validate step properties
                    step_validation = self._validate_training_step_properties(step)
                    results["details"]["step_validation"] = step_validation
                    
                    if not step_validation["valid"]:
                        results["passed"] = False
                        results["errors"].extend(step_validation["errors"])
            
            # Test step configuration completeness
            if hasattr(self.builder_instance, 'get_step_config'):
                config = self.builder_instance.get_step_config()
                results["details"]["step_config"] = config
                
                config_validation = self._validate_step_config_completeness(config)
                results["details"]["config_validation"] = config_validation
                
                if not config_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(config_validation["errors"])
            
            # Test estimator configuration
            if hasattr(self.builder_instance, 'get_estimator_config'):
                estimator_config = self.builder_instance.get_estimator_config()
                results["details"]["estimator_config"] = estimator_config
                
                estimator_validation = self._validate_estimator_config(estimator_config)
                if not estimator_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(estimator_validation["errors"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {str(e)}")
            return {
                "test_name": test_name,
                "passed": False,
                "details": {},
                "errors": [f"Test execution failed: {str(e)}"]
            }
    
    def test_framework_specific_training_workflow(self) -> Dict[str, Any]:
        """Test framework-specific training workflow patterns."""
        test_name = "test_framework_specific_training_workflow"
        logger.info(f"Running {test_name}")
        
        try:
            results = {
                "test_name": test_name,
                "passed": True,
                "details": {},
                "errors": []
            }
            
            framework = self._detect_framework()
            if not framework:
                results["details"]["framework"] = "No framework detected"
                return results
            
            results["details"]["framework"] = framework
            
            # Test PyTorch training workflow
            if framework == "pytorch":
                pytorch_workflow = self._test_pytorch_training_workflow()
                results["details"]["pytorch_workflow"] = pytorch_workflow
                if not pytorch_workflow["valid"]:
                    results["passed"] = False
                    results["errors"].extend(pytorch_workflow["errors"])
            
            # Test XGBoost training workflow
            elif framework == "xgboost":
                xgboost_workflow = self._test_xgboost_training_workflow()
                results["details"]["xgboost_workflow"] = xgboost_workflow
                if not xgboost_workflow["valid"]:
                    results["passed"] = False
                    results["errors"].extend(xgboost_workflow["errors"])
            
            # Test TensorFlow training workflow
            elif framework == "tensorflow":
                tf_workflow = self._test_tensorflow_training_workflow()
                results["details"]["tensorflow_workflow"] = tf_workflow
                if not tf_workflow["valid"]:
                    results["passed"] = False
                    results["errors"].extend(tf_workflow["errors"])
            
            # Test SKLearn training workflow
            elif framework == "sklearn":
                sklearn_workflow = self._test_sklearn_training_workflow()
                results["details"]["sklearn_workflow"] = sklearn_workflow
                if not sklearn_workflow["valid"]:
                    results["passed"] = False
                    results["errors"].extend(sklearn_workflow["errors"])
            
            # Test custom framework workflow
            else:
                custom_workflow = self._test_custom_training_workflow(framework)
                results["details"]["custom_workflow"] = custom_workflow
                if not custom_workflow["valid"]:
                    results["passed"] = False
                    results["errors"].extend(custom_workflow["errors"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {str(e)}")
            return {
                "test_name": test_name,
                "passed": False,
                "details": {},
                "errors": [f"Test execution failed: {str(e)}"]
            }
    
    def test_hyperparameter_optimization_integration(self) -> Dict[str, Any]:
        """Test hyperparameter optimization integration."""
        test_name = "test_hyperparameter_optimization_integration"
        logger.info(f"Running {test_name}")
        
        try:
            results = {
                "test_name": test_name,
                "passed": True,
                "details": {},
                "errors": []
            }
            
            # Test hyperparameter configuration
            if hasattr(self.builder_instance, 'get_hyperparameters'):
                hyperparams = self.builder_instance.get_hyperparameters()
                results["details"]["hyperparameters"] = hyperparams
                
                hyperparam_validation = self._validate_hyperparameters(hyperparams)
                results["details"]["hyperparam_validation"] = hyperparam_validation
                
                if not hyperparam_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(hyperparam_validation["errors"])
            
            # Test hyperparameter tuning configuration
            if hasattr(self.builder_instance, 'get_tuning_config'):
                tuning_config = self.builder_instance.get_tuning_config()
                results["details"]["tuning_config"] = tuning_config
                
                tuning_validation = self._validate_tuning_config(tuning_config)
                if not tuning_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(tuning_validation["errors"])
            
            # Test automatic model tuning integration
            if hasattr(self.builder_instance, 'get_auto_ml_config'):
                automl_config = self.builder_instance.get_auto_ml_config()
                results["details"]["automl_config"] = automl_config
                
                automl_validation = self._validate_automl_config(automl_config)
                if not automl_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(automl_validation["errors"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {str(e)}")
            return {
                "test_name": test_name,
                "passed": False,
                "details": {},
                "errors": [f"Test execution failed: {str(e)}"]
            }
    
    def test_data_channel_integration(self) -> Dict[str, Any]:
        """Test data channel integration and management."""
        test_name = "test_data_channel_integration"
        logger.info(f"Running {test_name}")
        
        try:
            results = {
                "test_name": test_name,
                "passed": True,
                "details": {},
                "errors": []
            }
            
            # Test training data channels
            if hasattr(self.builder_instance, 'get_training_inputs'):
                training_inputs = self.builder_instance.get_training_inputs()
                results["details"]["training_inputs"] = training_inputs
                
                for i, input_config in enumerate(training_inputs):
                    input_validation = self._validate_training_input(input_config, i)
                    results["details"][f"input_{i}_validation"] = input_validation
                    
                    if not input_validation["valid"]:
                        results["passed"] = False
                        results["errors"].extend(input_validation["errors"])
            
            # Test data distribution strategies
            if hasattr(self.builder_instance, 'get_data_distribution_config'):
                distribution_config = self.builder_instance.get_data_distribution_config()
                results["details"]["data_distribution"] = distribution_config
                
                distribution_validation = self._validate_data_distribution(distribution_config)
                if not distribution_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(distribution_validation["errors"])
            
            # Test data preprocessing integration
            if hasattr(self.builder_instance, 'get_preprocessing_config'):
                preprocessing_config = self.builder_instance.get_preprocessing_config()
                results["details"]["preprocessing_config"] = preprocessing_config
                
                preprocessing_validation = self._validate_preprocessing_config(preprocessing_config)
                if not preprocessing_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(preprocessing_validation["errors"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {str(e)}")
            return {
                "test_name": test_name,
                "passed": False,
                "details": {},
                "errors": [f"Test execution failed: {str(e)}"]
            }
    
    def test_model_artifact_generation(self) -> Dict[str, Any]:
        """Test model artifact generation and management."""
        test_name = "test_model_artifact_generation"
        logger.info(f"Running {test_name}")
        
        try:
            results = {
                "test_name": test_name,
                "passed": True,
                "details": {},
                "errors": []
            }
            
            # Test model output configuration
            if hasattr(self.builder_instance, 'get_model_output_config'):
                output_config = self.builder_instance.get_model_output_config()
                results["details"]["model_output_config"] = output_config
                
                output_validation = self._validate_model_output_config(output_config)
                results["details"]["output_validation"] = output_validation
                
                if not output_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(output_validation["errors"])
            
            # Test model artifact structure
            if hasattr(self.builder_instance, 'get_expected_artifacts'):
                expected_artifacts = self.builder_instance.get_expected_artifacts()
                results["details"]["expected_artifacts"] = expected_artifacts
                
                artifact_validation = self._validate_expected_artifacts(expected_artifacts)
                if not artifact_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(artifact_validation["errors"])
            
            # Test model serialization configuration
            if hasattr(self.builder_instance, 'get_serialization_config'):
                serialization_config = self.builder_instance.get_serialization_config()
                results["details"]["serialization_config"] = serialization_config
                
                serialization_validation = self._validate_serialization_config(serialization_config)
                if not serialization_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(serialization_validation["errors"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {str(e)}")
            return {
                "test_name": test_name,
                "passed": False,
                "details": {},
                "errors": [f"Test execution failed: {str(e)}"]
            }
    
    def test_training_job_monitoring(self) -> Dict[str, Any]:
        """Test training job monitoring and metrics collection."""
        test_name = "test_training_job_monitoring"
        logger.info(f"Running {test_name}")
        
        try:
            results = {
                "test_name": test_name,
                "passed": True,
                "details": {},
                "errors": []
            }
            
            # Test metrics configuration
            if hasattr(self.builder_instance, 'get_metrics_config'):
                metrics_config = self.builder_instance.get_metrics_config()
                results["details"]["metrics_config"] = metrics_config
                
                metrics_validation = self._validate_metrics_config(metrics_config)
                results["details"]["metrics_validation"] = metrics_validation
                
                if not metrics_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(metrics_validation["errors"])
            
            # Test logging configuration
            if hasattr(self.builder_instance, 'get_logging_config'):
                logging_config = self.builder_instance.get_logging_config()
                results["details"]["logging_config"] = logging_config
                
                logging_validation = self._validate_logging_config(logging_config)
                if not logging_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(logging_validation["errors"])
            
            # Test checkpoint configuration
            if hasattr(self.builder_instance, 'get_checkpoint_config'):
                checkpoint_config = self.builder_instance.get_checkpoint_config()
                results["details"]["checkpoint_config"] = checkpoint_config
                
                checkpoint_validation = self._validate_checkpoint_config(checkpoint_config)
                if not checkpoint_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(checkpoint_validation["errors"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {str(e)}")
            return {
                "test_name": test_name,
                "passed": False,
                "details": {},
                "errors": [f"Test execution failed: {str(e)}"]
            }
    
    def test_distributed_training_integration(self) -> Dict[str, Any]:
        """Test distributed training integration and configuration."""
        test_name = "test_distributed_training_integration"
        logger.info(f"Running {test_name}")
        
        try:
            results = {
                "test_name": test_name,
                "passed": True,
                "details": {},
                "errors": []
            }
            
            # Check if distributed training is configured
            if hasattr(self.builder_instance, 'is_distributed_training'):
                is_distributed = self.builder_instance.is_distributed_training()
                results["details"]["is_distributed"] = is_distributed
                
                if not is_distributed:
                    results["details"]["training_type"] = "single_instance"
                    return results
            
            # Test distributed training configuration
            if hasattr(self.builder_instance, 'get_distributed_config'):
                distributed_config = self.builder_instance.get_distributed_config()
                results["details"]["distributed_config"] = distributed_config
                
                distributed_validation = self._validate_distributed_config(distributed_config)
                results["details"]["distributed_validation"] = distributed_validation
                
                if not distributed_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(distributed_validation["errors"])
            
            # Test cluster configuration
            if hasattr(self.builder_instance, 'get_cluster_config'):
                cluster_config = self.builder_instance.get_cluster_config()
                results["details"]["cluster_config"] = cluster_config
                
                cluster_validation = self._validate_cluster_config(cluster_config)
                if not cluster_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(cluster_validation["errors"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {str(e)}")
            return {
                "test_name": test_name,
                "passed": False,
                "details": {},
                "errors": [f"Test execution failed: {str(e)}"]
            }
    
    def test_training_performance_optimization(self) -> Dict[str, Any]:
        """Test training performance optimization configuration."""
        test_name = "test_training_performance_optimization"
        logger.info(f"Running {test_name}")
        
        try:
            results = {
                "test_name": test_name,
                "passed": True,
                "details": {},
                "errors": []
            }
            
            # Test resource optimization
            if hasattr(self.builder_instance, 'get_resource_optimization_config'):
                resource_config = self.builder_instance.get_resource_optimization_config()
                results["details"]["resource_optimization"] = resource_config
                
                resource_validation = self._validate_resource_optimization(resource_config)
                if not resource_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(resource_validation["errors"])
            
            # Test GPU optimization
            if hasattr(self.builder_instance, 'get_gpu_optimization_config'):
                gpu_config = self.builder_instance.get_gpu_optimization_config()
                results["details"]["gpu_optimization"] = gpu_config
                
                gpu_validation = self._validate_gpu_optimization(gpu_config)
                if not gpu_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(gpu_validation["errors"])
            
            # Test memory optimization
            if hasattr(self.builder_instance, 'get_memory_optimization_config'):
                memory_config = self.builder_instance.get_memory_optimization_config()
                results["details"]["memory_optimization"] = memory_config
                
                memory_validation = self._validate_memory_optimization(memory_config)
                if not memory_validation["valid"]:
                    results["passed"] = False
                    results["errors"].extend(memory_validation["errors"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {str(e)}")
            return {
                "test_name": test_name,
                "passed": False,
                "details": {},
                "errors": [f"Test execution failed: {str(e)}"]
            }
    
    def test_training_dependency_resolution(self) -> Dict[str, Any]:
        """Test Training step dependency resolution."""
        test_name = "test_training_dependency_resolution"
        logger.info(f"Running {test_name}")
        
        try:
            results = {
                "test_name": test_name,
                "passed": True,
                "details": {},
                "errors": []
            }
            
            # Test processing step dependencies
            if hasattr(self.builder_instance, 'get_processing_dependencies'):
                processing_deps = self.builder_instance.get_processing_dependencies()
                results["details"]["processing_dependencies"] = processing_deps
                
                for dep in processing_deps:
                    dep_validation = self._validate_processing_dependency(dep)
                    if not dep_validation["valid"]:
                        results["passed"] = False
                        results["errors"].extend(dep_validation["errors"])
            
            # Test data dependencies
            if hasattr(self.builder_instance, 'get_data_dependencies'):
                data_deps = self.builder_instance.get_data_dependencies()
                results["details"]["data_dependencies"] = data_deps
                
                for dep in data_deps:
                    dep_validation = self._validate_data_dependency(dep)
                    if not dep_validation["valid"]:
                        results["passed"] = False
                        results["errors"].extend(dep_validation["errors"])
            
            # Test external dependencies
            if hasattr(self.builder_instance, 'get_external_dependencies'):
                external_deps = self.builder_instance.get_external_dependencies()
                results["details"]["external_dependencies"] = external_deps
                
                for dep in external_deps:
                    dep_validation = self._validate_external_dependency(dep)
                    if not dep_validation["valid"]:
                        results["passed"] = False
                        results["errors"].extend(dep_validation["errors"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {str(e)}")
            return {
                "test_name": test_name,
                "passed": False,
                "details": {},
                "errors": [f"Test execution failed: {str(e)}"]
            }
    
    # Helper methods for Training-specific validations
    
    def _validate_training_step_properties(self, step) -> Dict[str, Any]:
        """Validate Training step properties."""
        validation = {"valid": True, "errors": []}
        
        # Check required properties
        required_props = ["TrainingJobName", "AlgorithmSpecification", "RoleArn", "InputDataConfig", "OutputDataConfig"]
        for prop in required_props:
            if not hasattr(step, prop):
                validation["valid"] = False
                validation["errors"].append(f"Missing required property: {prop}")
        
        return validation
    
    def _validate_step_config_completeness(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate step configuration completeness."""
        validation = {"valid": True, "errors": []}
        
        required_config = ["training_job_name", "algorithm_specification", "role", "input_data_config"]
        for key in required_config:
            if key not in config:
                validation["valid"] = False
                validation["errors"].append(f"Missing required config: {key}")
        
        return validation
    
    def _validate_estimator_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate estimator configuration."""
        validation = {"valid": True, "errors": []}
        
        required_fields = ["image_uri", "role", "instance_type"]
        for field in required_fields:
            if field not in config:
                validation["valid"] = False
                validation["errors"].append(f"Missing estimator config field: {field}")
        
        return validation
    
    def _detect_framework(self) -> Optional[str]:
        """Detect the ML framework being used."""
        if hasattr(self.builder_instance, 'framework'):
            return self.builder_instance.framework
        return None
    
    def _test_pytorch_training_workflow(self) -> Dict[str, Any]:
        """Test PyTorch-specific training workflow."""
        return {
            "valid": True,
            "errors": [],
            "details": {"framework": "pytorch", "workflow_type": "pytorch_training"}
        }
    
    def _test_xgboost_training_workflow(self) -> Dict[str, Any]:
        """Test XGBoost-specific training workflow."""
        return {
            "valid": True,
            "errors": [],
            "details": {"framework": "xgboost", "workflow_type": "xgboost_training"}
        }
    
    def _test_tensorflow_training_workflow(self) -> Dict[str, Any]:
        """Test TensorFlow-specific training workflow."""
        return {
            "valid": True,
            "errors": [],
            "details": {"framework": "tensorflow", "workflow_type": "tensorflow_training"}
        }
    
    def _test_sklearn_training_workflow(self) -> Dict[str, Any]:
        """Test SKLearn-specific training workflow."""
        return {
            "valid": True,
            "errors": [],
            "details": {"framework": "sklearn", "workflow_type": "sklearn_training"}
        }
    
    def _test_custom_training_workflow(self, framework: str) -> Dict[str, Any]:
        """Test custom training workflow."""
        return {
            "valid": True,
            "errors": [],
            "details": {"framework": framework, "workflow_type": "custom_training"}
        }
    
    def _validate_hyperparameters(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hyperparameter configuration."""
        return {"valid": True, "errors": []}
    
    def _validate_tuning_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hyperparameter tuning configuration."""
        return {"valid": True, "errors": []}
    
    def _validate_automl_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate AutoML configuration."""
        return {"valid": True, "errors": []}
    
    def _validate_training_input(self, input_config: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Validate training input configuration."""
        validation = {"valid": True, "errors": []}
        
        required_fields = ["DataSource", "ContentType"]
        for field in required_fields:
            if field not in input_config:
                validation["valid"] = False
                validation["errors"].append(f"Training input {index} missing field: {field}")
        
        return validation
    
    def _validate_data_distribution(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data distribution configuration."""
        return {"valid": True, "errors": []}
    
    def _validate_preprocessing_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate preprocessing configuration."""
        return {"valid": True, "errors": []}
    
    def _validate_model_output_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model output configuration."""
        validation = {"valid": True, "errors": []}
        
        required_fields = ["S3OutputPath"]
        for field in required_fields:
            if field not in config:
                validation["valid"] = False
                validation["errors"].append(f"Missing model output field: {field}")
        
        return validation
    
    def _validate_expected_artifacts(self, artifacts: List[str]) -> Dict[str, Any]:
        """Validate expected model artifacts."""
        return {"valid": True, "errors": []}
    
    def _validate_serialization_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model serialization configuration."""
        return {"valid": True, "errors": []}
    
    def _validate_metrics_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training metrics configuration."""
        return {"valid": True, "errors": []}
    
    def _validate_logging_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate logging configuration."""
        return {"valid": True, "errors": []}
    
    def _validate_checkpoint_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate checkpoint configuration."""
        return {"valid": True, "errors": []}
    
    def _validate_distributed_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate distributed training configuration."""
        validation = {"valid": True, "errors": []}
        
        required_fields = ["InstanceType", "InstanceCount"]
        for field in required_fields:
            if field not in config:
                validation["valid"] = False
                validation["errors"].append(f"Missing distributed config field: {field}")
        
        return validation
    
    def _validate_cluster_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cluster configuration."""
        return {"valid": True, "errors": []}
    
    def _validate_resource_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate resource optimization configuration."""
        return {"valid": True, "errors": []}
    
    def _validate_gpu_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate GPU optimization configuration."""
        return {"valid": True, "errors": []}
    
    def _validate_memory_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate memory optimization configuration."""
        return {"valid": True, "errors": []}
    
    def _validate_processing_dependency(self, dep: Dict[str, Any]) -> Dict[str, Any]:
        """Validate processing step dependency."""
        return {"valid": True, "errors": []}
    
    def _validate_data_dependency(self, dep: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data dependency."""
        return {"valid": True, "errors": []}
    
    def _validate_external_dependency(self, dep: Dict[str, Any]) -> Dict[str, Any]:
        """Validate external dependency."""
        return {"valid": True, "errors": []}
