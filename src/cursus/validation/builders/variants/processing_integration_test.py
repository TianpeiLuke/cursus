"""
Level 4 Processing Integration Tests - Enhanced Processing-Specific Integration Validation.

These tests focus on Processing step-specific system integration and end-to-end functionality:
- Step creation patterns (Pattern A vs Pattern B) validation
- Framework-specific step creation and configuration
- Dependency resolution and input extraction
- Step name generation and consistency
- Cache configuration and step dependencies
"""

from typing import Dict, List, Any
from ..integration_tests import IntegrationTests


class ProcessingIntegrationTests(IntegrationTests):
    """
    Enhanced Level 4 integration tests for Processing step builders.
    
    Extends the generic integration tests with Processing-specific validations
    based on the patterns identified in Processing Step Builder Patterns analysis.
    """
    
    def get_step_type_specific_tests(self) -> List[str]:
        """Return Processing-specific integration test methods."""
        return [
            "test_step_creation_pattern_execution",
            "test_framework_specific_step_creation",
            "test_processing_dependency_resolution",
            "test_step_name_generation",
            "test_cache_configuration",
            "test_step_dependencies_handling",
            "test_end_to_end_step_creation",
            "test_specification_attachment"
        ]
    
    def _configure_step_type_mocks(self) -> None:
        """Configure Processing step-specific mock objects."""
        # Get Processing-specific mocks from factory
        self.step_type_mocks = self.mock_factory.create_step_type_mocks()
        
        # Log Processing step info if verbose
        if self.verbose:
            framework = self.step_info.get('framework', 'Unknown')
            pattern = self.step_info.get('step_creation_pattern', 'Unknown')
            self._log(f"Processing integration tests - Framework: {framework}, Pattern: {pattern}")
    
    def _validate_step_type_requirements(self) -> Dict[str, Any]:
        """Validate Processing step-specific requirements."""
        validation_results = {
            "is_processing_step": self.step_info.get("sagemaker_step_type") == "Processing",
            "has_step_creation_pattern": self.step_info.get("step_creation_pattern") is not None,
            "framework_detected": self.step_info.get("framework") is not None,
            "can_create_steps": True  # Will be validated in tests
        }
        
        return validation_results
    
    # Processing-specific integration tests
    
    def test_step_creation_pattern_execution(self) -> None:
        """Test step creation pattern execution (Pattern A vs Pattern B)."""
        self._log("Testing step creation pattern execution...")
        
        pattern = self.step_info.get('step_creation_pattern')
        framework = self.step_info.get('framework')
        
        if not pattern:
            # Infer pattern from framework
            if framework == 'sklearn':
                pattern = 'pattern_a'
            elif framework == 'xgboost':
                pattern = 'pattern_b'
        
        try:
            builder = self._create_builder_instance()
            
            # Test step creation with minimal inputs
            test_inputs = {}
            test_outputs = {}
            dependencies = []
            
            step = builder.create_step(
                inputs=test_inputs,
                outputs=test_outputs,
                dependencies=dependencies,
                enable_caching=True
            )
            
            # Validate step creation based on pattern
            if pattern == 'pattern_a':
                self._validate_pattern_a_step(step, builder)
            elif pattern == 'pattern_b':
                self._validate_pattern_b_step(step, builder)
            else:
                self._validate_generic_processing_step(step, builder)
                
        except Exception as e:
            self._assert(False, f"Step creation pattern execution test failed: {str(e)}")
    
    def test_framework_specific_step_creation(self) -> None:
        """Test framework-specific step creation and configuration."""
        self._log("Testing framework-specific step creation...")
        
        framework = self.step_info.get('framework')
        if not framework:
            self._log("No framework detected, skipping framework-specific test")
            return
        
        try:
            builder = self._create_builder_instance()
            
            # Create step with framework-specific configuration
            step = builder.create_step(
                inputs={},
                outputs={},
                dependencies=[],
                enable_caching=True
            )
            
            if framework == 'sklearn':
                self._validate_sklearn_step(step, builder)
            elif framework == 'xgboost':
                self._validate_xgboost_step(step, builder)
            else:
                self._log(f"Unknown framework: {framework}, performing generic validation")
                self._validate_generic_processing_step(step, builder)
                
        except Exception as e:
            self._assert(False, f"Framework-specific step creation test failed: {str(e)}")
    
    def test_processing_dependency_resolution(self) -> None:
        """Test Processing-specific dependency resolution."""
        self._log("Testing processing dependency resolution...")
        
        try:
            builder = self._create_builder_instance()
            
            # Test dependency extraction if method exists
            if hasattr(builder, 'extract_inputs_from_dependencies'):
                # Create mock dependencies
                mock_dependencies = []
                
                extracted_inputs = builder.extract_inputs_from_dependencies(mock_dependencies)
                self._assert(
                    isinstance(extracted_inputs, dict),
                    "Extracted inputs must be a dictionary"
                )
                
                self._log(f"Dependency resolution successful: {len(extracted_inputs)} inputs extracted")
            
            # Test step creation with dependencies
            test_dependencies = []  # Mock dependencies
            step = builder.create_step(
                inputs={},
                outputs={},
                dependencies=test_dependencies,
                enable_caching=True
            )
            
            # Validate that step handles dependencies correctly
            self._validate_step_dependencies(step, test_dependencies)
            
        except Exception as e:
            self._assert(False, f"Processing dependency resolution test failed: {str(e)}")
    
    def test_step_name_generation(self) -> None:
        """Test step name generation and consistency."""
        self._log("Testing step name generation...")
        
        try:
            builder = self._create_builder_instance()
            
            # Test step name generation method
            step_name = builder._get_step_name()
            
            self._assert(
                isinstance(step_name, str) and len(step_name) > 0,
                "Step name must be a non-empty string"
            )
            
            # Test step name consistency
            step_name_2 = builder._get_step_name()
            self._assert(
                step_name == step_name_2,
                "Step name generation must be consistent"
            )
            
            # Test that created step uses the generated name
            step = builder.create_step(
                inputs={},
                outputs={},
                dependencies=[],
                enable_caching=True
            )
            
            if hasattr(step, 'name'):
                actual_step_name = step.name
                self._assert(
                    actual_step_name == step_name,
                    f"Step should use generated name: expected {step_name}, got {actual_step_name}"
                )
            
            self._log(f"Step name generation successful: {step_name}")
            
        except Exception as e:
            self._assert(False, f"Step name generation test failed: {str(e)}")
    
    def test_cache_configuration(self) -> None:
        """Test cache configuration for Processing steps."""
        self._log("Testing cache configuration...")
        
        try:
            builder = self._create_builder_instance()
            
            # Test cache configuration method if available
            if hasattr(builder, '_get_cache_config'):
                # Test with caching enabled
                cache_config_enabled = builder._get_cache_config(True)
                if cache_config_enabled is not None:
                    self._log(f"Cache config (enabled): {cache_config_enabled}")
                
                # Test with caching disabled
                cache_config_disabled = builder._get_cache_config(False)
                if cache_config_disabled is not None:
                    self._log(f"Cache config (disabled): {cache_config_disabled}")
            
            # Test step creation with different cache settings
            step_cached = builder.create_step(
                inputs={},
                outputs={},
                dependencies=[],
                enable_caching=True
            )
            
            step_uncached = builder.create_step(
                inputs={},
                outputs={},
                dependencies=[],
                enable_caching=False
            )
            
            # Validate cache configuration in steps
            self._validate_step_cache_config(step_cached, True)
            self._validate_step_cache_config(step_uncached, False)
            
        except Exception as e:
            self._assert(False, f"Cache configuration test failed: {str(e)}")
    
    def test_step_dependencies_handling(self) -> None:
        """Test step dependencies handling."""
        self._log("Testing step dependencies handling...")
        
        try:
            builder = self._create_builder_instance()
            
            # Create mock dependencies
            mock_dependencies = []  # In real scenario, these would be SageMaker steps
            
            # Test step creation with dependencies
            step = builder.create_step(
                inputs={},
                outputs={},
                dependencies=mock_dependencies,
                enable_caching=True
            )
            
            # Validate that step correctly handles dependencies
            if hasattr(step, 'depends_on'):
                depends_on = step.depends_on
                self._assert(
                    depends_on == mock_dependencies,
                    f"Step dependencies should match input: expected {mock_dependencies}, got {depends_on}"
                )
            
            self._log("Step dependencies handling successful")
            
        except Exception as e:
            self._assert(False, f"Step dependencies handling test failed: {str(e)}")
    
    def test_end_to_end_step_creation(self) -> None:
        """Test end-to-end step creation with realistic inputs."""
        self._log("Testing end-to-end step creation...")
        
        try:
            builder = self._create_builder_instance()
            
            # Create realistic test inputs based on specification
            test_inputs = {}
            test_outputs = {}
            
            if builder.spec:
                # Create inputs based on specification
                for dep_name, dep_spec in builder.spec.dependencies.items():
                    logical_name = dep_spec.logical_name
                    test_inputs[logical_name] = f"s3://test-bucket/input/{logical_name}"
                
                # Create outputs based on specification
                for out_name, out_spec in builder.spec.outputs.items():
                    logical_name = out_spec.logical_name
                    test_outputs[logical_name] = f"s3://test-bucket/output/{logical_name}"
            
            # Create step with realistic configuration
            step = builder.create_step(
                inputs=test_inputs,
                outputs=test_outputs,
                dependencies=[],
                enable_caching=True
            )
            
            # Comprehensive step validation
            self._validate_complete_processing_step(step, builder, test_inputs, test_outputs)
            
            self._log("End-to-end step creation successful")
            
        except Exception as e:
            self._assert(False, f"End-to-end step creation test failed: {str(e)}")
    
    def test_specification_attachment(self) -> None:
        """Test specification attachment to created steps."""
        self._log("Testing specification attachment...")
        
        try:
            builder = self._create_builder_instance()
            
            # Create step
            step = builder.create_step(
                inputs={},
                outputs={},
                dependencies=[],
                enable_caching=True
            )
            
            # Check if specification is attached to step
            if hasattr(step, '_spec'):
                attached_spec = step._spec
                
                if attached_spec is not None:
                    self._assert(
                        attached_spec == builder.spec,
                        "Attached specification should match builder specification"
                    )
                    self._log("Specification successfully attached to step")
                else:
                    self._log("No specification attached (acceptable if no spec available)")
            else:
                self._log("Step does not support specification attachment")
            
        except Exception as e:
            self._assert(False, f"Specification attachment test failed: {str(e)}")
    
    # Helper methods for step validation
    
    def _validate_pattern_a_step(self, step, builder) -> None:
        """Validate Pattern A (Direct ProcessingStep creation) step."""
        self._log("Validating Pattern A step...")
        
        # Pattern A creates ProcessingStep directly
        step_type = type(step).__name__
        self._assert(
            'ProcessingStep' in step_type,
            f"Pattern A should create ProcessingStep, got {step_type}"
        )
        
        # Pattern A should have processor attribute
        if hasattr(step, 'processor'):
            processor = step.processor
            self._log(f"Pattern A processor: {type(processor).__name__}")
        
        # Pattern A should have code attribute
        if hasattr(step, 'code'):
            code = step.code
            self._log(f"Pattern A code: {code}")
    
    def _validate_pattern_b_step(self, step, builder) -> None:
        """Validate Pattern B (processor.run() + step_args) step."""
        self._log("Validating Pattern B step...")
        
        # Pattern B creates ProcessingStep with step_args
        step_type = type(step).__name__
        self._assert(
            'ProcessingStep' in step_type,
            f"Pattern B should create ProcessingStep, got {step_type}"
        )
        
        # Pattern B should have step_args attribute
        if hasattr(step, 'step_args'):
            step_args = step.step_args
            self._log(f"Pattern B step_args: {type(step_args)}")
        
        # Pattern B typically doesn't have direct processor attribute
        if not hasattr(step, 'processor'):
            self._log("Pattern B step correctly uses step_args instead of direct processor")
    
    def _validate_generic_processing_step(self, step, builder) -> None:
        """Validate generic Processing step."""
        self._log("Validating generic processing step...")
        
        # Basic ProcessingStep validation
        step_type = type(step).__name__
        self._assert(
            'ProcessingStep' in step_type,
            f"Should create ProcessingStep, got {step_type}"
        )
        
        # Check basic step attributes
        if hasattr(step, 'name'):
            self._log(f"Step name: {step.name}")
        
        if hasattr(step, 'depends_on'):
            self._log(f"Step dependencies: {step.depends_on}")
    
    def _validate_sklearn_step(self, step, builder) -> None:
        """Validate SKLearn-specific step."""
        self._log("Validating SKLearn step...")
        
        # SKLearn steps typically use Pattern A
        self._validate_pattern_a_step(step, builder)
        
        # Additional SKLearn-specific validations
        if hasattr(step, 'processor'):
            processor_type = type(step.processor).__name__
            if 'SKLearn' in processor_type:
                self._log(f"SKLearn processor confirmed: {processor_type}")
    
    def _validate_xgboost_step(self, step, builder) -> None:
        """Validate XGBoost-specific step."""
        self._log("Validating XGBoost step...")
        
        # XGBoost steps typically use Pattern B
        self._validate_pattern_b_step(step, builder)
        
        # Additional XGBoost-specific validations
        # XGBoost steps may have framework-specific attributes
        self._log("XGBoost step validation completed")
    
    def _validate_step_dependencies(self, step, expected_dependencies) -> None:
        """Validate step dependencies."""
        if hasattr(step, 'depends_on'):
            actual_dependencies = step.depends_on
            self._assert(
                actual_dependencies == expected_dependencies,
                f"Step dependencies mismatch: expected {expected_dependencies}, got {actual_dependencies}"
            )
        else:
            self._log("Step does not have depends_on attribute")
    
    def _validate_step_cache_config(self, step, cache_enabled: bool) -> None:
        """Validate step cache configuration."""
        if hasattr(step, 'cache_config'):
            cache_config = step.cache_config
            
            if cache_enabled:
                self._assert(
                    cache_config is not None,
                    "Cache config should be set when caching is enabled"
                )
            else:
                # Cache config may be None or disabled
                self._log(f"Cache config for disabled caching: {cache_config}")
        else:
            self._log("Step does not have cache_config attribute")
    
    def _validate_complete_processing_step(self, step, builder, test_inputs: Dict, test_outputs: Dict) -> None:
        """Validate complete Processing step with all components."""
        self._log("Validating complete processing step...")
        
        # Basic step validation
        self._validate_generic_processing_step(step, builder)
        
        # Validate step name
        if hasattr(step, 'name'):
            step_name = step.name
            self._assert(
                isinstance(step_name, str) and len(step_name) > 0,
                "Step must have a valid name"
            )
        
        # Validate inputs/outputs if step has them
        if hasattr(step, 'inputs'):
            inputs = step.inputs
            if inputs:
                self._assert(
                    isinstance(inputs, list),
                    "Step inputs must be a list"
                )
                self._log(f"Step has {len(inputs)} inputs")
        
        if hasattr(step, 'outputs'):
            outputs = step.outputs
            if outputs:
                self._assert(
                    isinstance(outputs, list),
                    "Step outputs must be a list"
                )
                self._log(f"Step has {len(outputs)} outputs")
        
        # Validate job arguments if step has them
        if hasattr(step, 'job_arguments'):
            job_args = step.job_arguments
            if job_args:
                self._assert(
                    isinstance(job_args, list),
                    "Job arguments must be a list"
                )
                self._log(f"Step has {len(job_args)} job arguments")
        
        self._log("Complete processing step validation successful")
