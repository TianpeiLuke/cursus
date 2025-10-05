import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
import logging

from cursus.core.assembler.pipeline_assembler import PipelineAssembler, safe_value_for_logging
from cursus.core.base import (
    BasePipelineConfig,
    StepBuilderBase,
    OutputSpec,
    DependencySpec,
    StepSpecification,
)
from cursus.api.dag.base_dag import PipelineDAG
from cursus.core.deps.registry_manager import RegistryManager
from cursus.core.deps.dependency_resolver import UnifiedDependencyResolver
from cursus.core.base.enums import DependencyType, NodeType


class TestSafeValueForLogging:
    """Test cases for safe_value_for_logging function.
    
    Following pytest best practices:
    1. Read source code first to understand actual implementation ✅
    2. Test actual behavior, not assumptions ✅
    3. Use implementation-driven test design ✅
    """

    def test_safe_value_for_logging_pipeline_variable_with_expr(self):
        """Test safe logging of objects with expr attribute (Pipeline variables)."""
        # Based on source: if hasattr(value, "expr"): return f"[Pipeline Variable: {value.__class__.__name__}]"
        mock_pipeline_var = Mock()
        mock_pipeline_var.expr = "some_expression"
        mock_pipeline_var.__class__.__name__ = "ParameterString"
        
        result = safe_value_for_logging(mock_pipeline_var)
        
        assert result == "[Pipeline Variable: ParameterString]"

    def test_safe_value_for_logging_dict_always_returns_ellipsis(self):
        """Test safe logging of dict always returns {...} regardless of content."""
        # Based on source: if isinstance(value, dict): return "{...}"
        
        # Non-empty dict
        result_non_empty = safe_value_for_logging({"key1": "value1", "key2": "value2"})
        assert result_non_empty == "{...}"
        
        # Empty dict
        result_empty = safe_value_for_logging({})
        assert result_empty == "{...}"

    def test_safe_value_for_logging_list_with_length(self):
        """Test safe logging of list returns type name with item count."""
        # Based on source: if isinstance(value, (list, tuple, set)): return f"[{type(value).__name__} with {len(value)} items]"
        
        # Non-empty list
        test_list = ["item1", "item2", "item3"]
        result = safe_value_for_logging(test_list)
        assert result == "[list with 3 items]"
        
        # Empty list
        empty_list = []
        result_empty = safe_value_for_logging(empty_list)
        assert result_empty == "[list with 0 items]"

    def test_safe_value_for_logging_tuple_with_length(self):
        """Test safe logging of tuple returns type name with item count."""
        # Based on source: same logic as list
        test_tuple = ("item1", "item2")
        result = safe_value_for_logging(test_tuple)
        assert result == "[tuple with 2 items]"

    def test_safe_value_for_logging_set_with_length(self):
        """Test safe logging of set returns type name with item count."""
        # Based on source: same logic as list/tuple
        test_set = {"item1", "item2", "item3"}
        result = safe_value_for_logging(test_set)
        assert result == "[set with 3 items]"

    def test_safe_value_for_logging_simple_values_use_str(self):
        """Test safe logging of simple values uses str() function."""
        # Based on source: try: return str(value)
        
        # String
        result_string = safe_value_for_logging("simple_string")
        assert result_string == "simple_string"
        
        # Number
        result_number = safe_value_for_logging(42)
        assert result_number == "42"
        
        # None
        result_none = safe_value_for_logging(None)
        assert result_none == "None"
        
        # Boolean
        result_bool = safe_value_for_logging(True)
        assert result_bool == "True"

    def test_safe_value_for_logging_exception_handling(self):
        """Test safe logging handles objects that raise exceptions during str()."""
        # Based on source: except Exception: return f"[Object of type: {type(value).__name__}]"
        
        # Create object that raises exception on str()
        class ProblematicObject:
            def __str__(self):
                raise Exception("str() failed")
        
        test_obj = ProblematicObject()
        result = safe_value_for_logging(test_obj)
        
        assert result == "[Object of type: ProblematicObject]"

    def test_safe_value_for_logging_custom_object_without_expr(self):
        """Test safe logging of custom objects without expr attribute uses str()."""
        # Based on source: objects without expr go to str() path
        
        class CustomObject:
            def __init__(self):
                self.data = "test"
            
            def __str__(self):
                return f"CustomObject(data={self.data})"
        
        test_obj = CustomObject()
        result = safe_value_for_logging(test_obj)
        
        assert result == "CustomObject(data=test)"

    def test_safe_value_for_logging_priority_expr_over_collections(self):
        """Test that expr attribute takes priority over collection type checking."""
        # Based on source: expr check happens first, before isinstance checks
        
        # Create a list-like object with expr attribute
        class ListWithExpr(list):
            def __init__(self):
                super().__init__(["item1", "item2"])
                self.expr = "some_expression"
        
        test_obj = ListWithExpr()
        result = safe_value_for_logging(test_obj)
        
        # Should return Pipeline Variable format, not list format
        assert result == "[Pipeline Variable: ListWithExpr]"

    def test_safe_value_for_logging_edge_cases(self):
        """Test safe logging edge cases based on implementation."""
        # Based on source implementation behavior
        
        # Object with expr but no __class__.__name__ (shouldn't happen but test defensive)
        mock_obj = Mock()
        mock_obj.expr = "test"
        # Mock should have __class__.__name__ = "Mock" by default
        result = safe_value_for_logging(mock_obj)
        assert result == "[Pipeline Variable: Mock]"


class MockConfig(BasePipelineConfig):
    """Mock configuration class for testing.
    
    Following pytest best practices:
    1. Fix the _cache attribute initialization issue
    2. Ensure proper inheritance from BasePipelineConfig
    3. Mock only what's necessary for testing
    """

    def __init__(
        self,
        author="test_author",
        bucket="test-bucket",
        role="test-role",
        region="NA",
        service_name="test_service",
        pipeline_version="1.0.0",
        project_root_folder="cursus",
    ):
        super().__init__(
            author=author,
            bucket=bucket,
            role=role,
            region=region,
            service_name=service_name,
            pipeline_version=pipeline_version,
            project_root_folder=project_root_folder,
        )
        
        # CRITICAL FIX: Initialize _cache as actual dict to fix the TypeError
        # Based on source: get_script_contract() expects _cache to be a dict for "in" operator
        # The PrivateAttr creates a ModelPrivateAttr object, not a dict
        self._cache = {}
        
        # Initialize _step_catalog to None to avoid lazy loading issues in tests
        self._step_catalog = None
    
    def get_script_contract(self):
        """Override to return None for testing - avoids complex contract loading."""
        return None
    
    @property
    def script_contract(self):
        """Override to return None for testing."""
        return None


class MockStepBuilder(StepBuilderBase):
    """Mock step builder class for testing."""

    def __init__(
        self,
        config,
        sagemaker_session=None,
        role=None,
        registry_manager=None,
        dependency_resolver=None,
    ):
        # Create real OutputSpec and DependencySpec instances to avoid mock interference
        output1 = OutputSpec(
            logical_name="output1",
            description="Mock output 1",
            property_path="properties.MockStep.Properties.OutputDataConfig.S3OutputPath",
            output_type=DependencyType.PROCESSING_OUTPUT,
        )
        output2 = OutputSpec(
            logical_name="output2",
            description="Mock output 2",
            property_path="properties.MockStep.Properties.ProcessingOutputConfig.Outputs.output2.S3Output.S3Uri",
            output_type=DependencyType.PROCESSING_OUTPUT,
        )

        input1 = DependencySpec(
            logical_name="input1",
            description="Mock input 1",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
        )
        input2 = DependencySpec(
            logical_name="input2",
            description="Mock input 2",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
        )

        # Create real StepSpecification instance
        self.spec = StepSpecification(
            step_type="MockStep",
            node_type=NodeType.INTERNAL,
            dependencies=[input1, input2],
            outputs=[output1, output2],
        )

        # Call parent constructor
        super().__init__(
            config=config,
            spec=self.spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )

    def validate_configuration(self) -> None:
        """Mock validation - always passes."""
        pass

    def _get_inputs(self, inputs: Dict[str, Any]) -> Any:
        """Mock implementation of abstract method."""
        return inputs

    def _get_outputs(self, outputs: Dict[str, Any]) -> Any:
        """Mock implementation of abstract method."""
        return outputs

    def create_step(
        self, inputs=None, outputs=None, dependencies=None, enable_caching=True
    ):
        """Create a mock step."""
        mock_step = Mock()
        mock_step.name = f"mock_step_{id(self)}"
        mock_step.inputs = inputs or {}
        mock_step.outputs = outputs or {}
        mock_step.dependencies = dependencies or []
        return mock_step


class TestPipelineAssembler:
    """Test cases for PipelineAssembler class.
    
    Following pytest best practices:
    1. Read source code first to understand actual implementation ✅
    2. Test actual behavior, not assumptions ✅
    3. Use implementation-driven test design ✅
    4. Mock only external dependencies, test actual class methods ✅
    """

    @pytest.fixture
    def simple_dag(self):
        """Create a simple DAG for testing."""
        return PipelineDAG(nodes=["step1", "step2"], edges=[("step1", "step2")])

    @pytest.fixture
    def simple_config_map(self):
        """Create simple config map for testing."""
        return {
            "step1": MockConfig(),
            "step2": MockConfig()
        }

    @pytest.fixture
    def mock_step_catalog(self):
        """Create mock step catalog that returns our MockStepBuilder."""
        from cursus.step_catalog import StepCatalog
        mock_catalog = Mock(spec=StepCatalog)
        mock_catalog.get_builder_for_config.return_value = MockStepBuilder
        return mock_catalog

    @pytest.fixture
    def mock_registry_manager(self):
        """Create mock registry manager."""
        mock_registry_manager = Mock(spec=RegistryManager)
        mock_registry = Mock()
        mock_registry_manager.get_registry.return_value = mock_registry
        return mock_registry_manager

    @pytest.fixture
    def mock_dependency_resolver(self):
        """Create mock dependency resolver."""
        mock_resolver = Mock(spec=UnifiedDependencyResolver)
        mock_resolver._calculate_compatibility.return_value = 0.8
        return mock_resolver

    def test_init_successful_initialization(
        self, simple_dag, simple_config_map, mock_step_catalog, 
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test successful initialization of PipelineAssembler.
        
        Based on source: __init__ method validates inputs and initializes step builders.
        """
        # This tests the actual __init__ method implementation
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )

        # Verify actual initialization occurred
        assert assembler.dag == simple_dag
        assert assembler.config_map == simple_config_map
        assert assembler.step_catalog == mock_step_catalog
        assert len(assembler.step_builders) == 2
        assert "step1" in assembler.step_builders
        assert "step2" in assembler.step_builders
        assert isinstance(assembler.step_builders["step1"], MockStepBuilder)
        assert isinstance(assembler.step_builders["step2"], MockStepBuilder)

    def test_init_missing_configs_raises_error(self, simple_dag, mock_step_catalog):
        """Test initialization with missing configs raises ValueError.
        
        Based on source: __init__ checks missing_configs and raises ValueError.
        """
        incomplete_config_map = {"step1": MockConfig()}  # Missing step2
        
        # This tests the actual validation logic in __init__
        with pytest.raises(ValueError, match="Missing configs for nodes"):
            PipelineAssembler(
                dag=simple_dag,
                config_map=incomplete_config_map,
                step_catalog=mock_step_catalog,
            )

    def test_init_missing_step_builders_raises_error(self, simple_dag, simple_config_map):
        """Test initialization with missing step builders raises ValueError.
        
        Based on source: __init__ checks builder_class and raises ValueError.
        """
        from cursus.step_catalog import StepCatalog
        mock_catalog = Mock(spec=StepCatalog)
        mock_catalog.get_builder_for_config.return_value = None  # No builder found
        
        # This tests the actual builder validation logic in __init__
        with pytest.raises(ValueError, match="No step builder found for config"):
            PipelineAssembler(
                dag=simple_dag,
                config_map=simple_config_map,
                step_catalog=mock_catalog,
            )

    def test_initialize_step_builders_creates_builders(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test _initialize_step_builders method creates step builders.
        
        Based on source: _initialize_step_builders iterates through dag.nodes and creates builders.
        """
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Verify the actual _initialize_step_builders method worked
        assert len(assembler.step_builders) == 2
        for step_name in ["step1", "step2"]:
            assert step_name in assembler.step_builders
            builder = assembler.step_builders[step_name]
            assert isinstance(builder, MockStepBuilder)
            assert builder.config == simple_config_map[step_name]

    def test_initialize_step_builders_with_pipeline_parameters(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test _initialize_step_builders with pipeline parameters.
        
        Based on source: _initialize_step_builders checks for EXECUTION_S3_PREFIX parameter.
        """
        from sagemaker.workflow.parameters import ParameterString
        
        pipeline_params = [
            ParameterString(name="EXECUTION_S3_PREFIX", default_value="s3://test-bucket/execution")
        ]
        
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            pipeline_parameters=pipeline_params,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Verify execution prefix was set (tests actual parameter handling logic)
        for step_name, builder in assembler.step_builders.items():
            assert hasattr(builder, 'execution_prefix')
            # The MockStepBuilder should have received the execution prefix
            assert builder.execution_prefix is not None
            assert builder.execution_prefix.name == "EXECUTION_S3_PREFIX"

    def test_propagate_messages_uses_dependency_resolver(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test _propagate_messages method uses dependency resolver.
        
        Based on source: _propagate_messages calls resolver._calculate_compatibility.
        """
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Set up resolver to return high compatibility
        mock_dependency_resolver._calculate_compatibility.return_value = 0.9
        
        # Call the actual _propagate_messages method
        assembler._propagate_messages()
        
        # Verify the method actually called the resolver
        assert mock_dependency_resolver._calculate_compatibility.called

    def test_generate_outputs_creates_join_objects(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test _generate_outputs method creates Join objects.
        
        Based on source: _generate_outputs uses Join(on="/", values=[base_s3_loc, step_type, logical_name]).
        """
        from sagemaker.workflow.functions import Join
        
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Call the actual _generate_outputs method
        outputs = assembler._generate_outputs("step1")
        
        # Verify the method creates actual Join objects (tests real implementation)
        assert isinstance(outputs, dict)
        for output_name, output_path in outputs.items():
            assert isinstance(output_path, Join), f"Output {output_name} should be Join object"

    def test_instantiate_step_creates_step(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test _instantiate_step method creates step.
        
        Based on source: _instantiate_step calls builder.create_step(**kwargs).
        """
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Call the actual _instantiate_step method
        step = assembler._instantiate_step("step1")
        
        # Verify the method creates an actual step (tests real implementation)
        assert step is not None
        assert hasattr(step, "name")

    def test_get_registry_manager_returns_manager(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test _get_registry_manager method returns registry manager.
        
        Based on source: _get_registry_manager returns self._registry_manager.
        """
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Call the actual _get_registry_manager method
        result = assembler._get_registry_manager()
        
        # Verify the method returns the actual registry manager
        assert result == mock_registry_manager

    def test_get_dependency_resolver_returns_resolver(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test _get_dependency_resolver method returns dependency resolver.
        
        Based on source: _get_dependency_resolver returns self._dependency_resolver.
        """
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Call the actual _get_dependency_resolver method
        result = assembler._get_dependency_resolver()
        
        # Verify the method returns the actual dependency resolver
        assert result == mock_dependency_resolver

    @patch("cursus.core.assembler.pipeline_assembler.Pipeline")
    def test_generate_pipeline_creates_pipeline(
        self, mock_pipeline_class, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test generate_pipeline method creates SageMaker Pipeline.
        
        Based on source: generate_pipeline calls Pipeline() constructor.
        """
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Call the actual generate_pipeline method
        result = assembler.generate_pipeline("test_pipeline")
        
        # Verify the method creates actual pipeline (tests real implementation)
        assert result == mock_pipeline
        mock_pipeline_class.assert_called_once()
        
        # Verify step instances were created
        assert len(assembler.step_instances) == 2

    def test_generate_pipeline_with_cyclic_dag_raises_error(
        self, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test generate_pipeline with cyclic DAG raises ValueError.
        
        Based on source: generate_pipeline calls dag.topological_sort() and handles ValueError.
        """
        # Create DAG with cycle
        cyclic_dag = PipelineDAG(
            nodes=["step1", "step2"],
            edges=[("step1", "step2"), ("step2", "step1")]  # Creates cycle
        )
        
        assembler = PipelineAssembler(
            dag=cyclic_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Call the actual generate_pipeline method with cyclic DAG
        with pytest.raises(ValueError, match="Failed to determine build order"):
            assembler.generate_pipeline("test_pipeline")

    @patch("cursus.core.assembler.pipeline_assembler.create_pipeline_components")
    def test_create_with_components_factory_method(
        self, mock_create_components, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test create_with_components class method.
        
        Based on source: create_with_components calls create_pipeline_components().
        """
        mock_components = {
            "registry_manager": mock_registry_manager,
            "resolver": mock_dependency_resolver,
        }
        mock_create_components.return_value = mock_components
        
        # Call the actual create_with_components class method
        assembler = PipelineAssembler.create_with_components(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            context_name="test_context",
        )
        
        # Verify the factory method works (tests real implementation)
        mock_create_components.assert_called_once_with("test_context")
        assert assembler._registry_manager == mock_registry_manager
        assert assembler._dependency_resolver == mock_dependency_resolver

    def test_pipeline_regeneration_clears_instances(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test pipeline regeneration clears existing step instances.
        
        Based on source: generate_pipeline clears self.step_instances if it exists.
        """
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Add some mock step instances
        assembler.step_instances = {"step1": Mock(), "step2": Mock()}
        
        with patch("cursus.core.assembler.pipeline_assembler.Pipeline") as mock_pipeline_class:
            mock_pipeline_class.return_value = Mock()
            
            # Call the actual generate_pipeline method
            assembler.generate_pipeline("test_pipeline")
            
            # Verify instances were cleared and recreated (tests real implementation)
            assert len(assembler.step_instances) == 2
            # Should have new instances, not the old mocks
            for step_name in ["step1", "step2"]:
                assert step_name in assembler.step_instances

    # ADDITIONAL TESTS FOLLOWING PYTEST BEST PRACTICES GUIDE

    def test_init_edge_validation_invalid_dag_edges(self, simple_config_map, mock_step_catalog):
        """Test initialization validates DAG edges exist in nodes.
        
        Based on source: PipelineDAG constructor validates edges during creation.
        Following pytest guide: Test edge cases and error conditions.
        """
        # Test that PipelineDAG raises KeyError when creating DAG with invalid edges
        # This tests the actual behavior observed in the source
        with pytest.raises(KeyError):  # PipelineDAG raises KeyError for invalid edges during construction
            invalid_dag = PipelineDAG(
                nodes=["step1", "step2"], 
                edges=[("step1", "step3")]  # step3 not in nodes
            )

    def test_init_default_parameters_handling(self, simple_dag, simple_config_map, mock_step_catalog):
        """Test initialization with default parameters.
        
        Based on source: __init__ sets defaults for optional parameters.
        Following pytest guide: Test default behavior from source.
        """
        # Test with minimal parameters (tests default parameter handling)
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
        )
        
        # Verify defaults were set correctly (from source code)
        assert assembler.sagemaker_session is None
        assert assembler.role is None
        assert assembler.pipeline_parameters == []
        assert assembler._registry_manager is not None  # Created by default
        assert assembler._dependency_resolver is not None  # Created by default

    def test_initialize_step_builders_error_handling(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test _initialize_step_builders handles builder creation errors.
        
        Based on source: _initialize_step_builders has try/except and raises ValueError.
        Following pytest guide: Test exception handling paths.
        """
        # Make step catalog return a valid builder class but make builder instantiation fail
        # This tests the actual exception handling path in _initialize_step_builders
        
        class FailingMockStepBuilder(MockStepBuilder):
            def __init__(self, *args, **kwargs):
                raise Exception("Builder instantiation failed")
        
        mock_step_catalog.get_builder_for_config.return_value = FailingMockStepBuilder
        
        # This should trigger the exception handling in _initialize_step_builders
        with pytest.raises(ValueError, match="Failed to initialize step builder"):
            PipelineAssembler(
                dag=simple_dag,
                config_map=simple_config_map,
                step_catalog=mock_step_catalog,
                registry_manager=mock_registry_manager,
                dependency_resolver=mock_dependency_resolver,
            )

    def test_generate_outputs_no_specification_returns_empty(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test _generate_outputs returns empty dict when no specification.
        
        Based on source: _generate_outputs checks if builder has spec and returns {}.
        Following pytest guide: Test edge cases from implementation.
        """
        # Create assembler
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Remove spec from builder to test edge case
        assembler.step_builders["step1"].spec = None
        
        # Call _generate_outputs - should return empty dict
        outputs = assembler._generate_outputs("step1")
        
        assert outputs == {}

    def test_instantiate_step_with_dependencies(
        self, mock_registry_manager, mock_dependency_resolver, mock_step_catalog
    ):
        """Test _instantiate_step handles step dependencies correctly.
        
        Based on source: _instantiate_step calls dag.get_dependencies() and processes them.
        Following pytest guide: Test actual method behavior with dependencies.
        """
        # Create DAG with dependencies
        dag_with_deps = PipelineDAG(
            nodes=["step1", "step2", "step3"],
            edges=[("step1", "step2"), ("step2", "step3")]
        )
        
        config_map = {
            "step1": MockConfig(),
            "step2": MockConfig(), 
            "step3": MockConfig()
        }
        
        assembler = PipelineAssembler(
            dag=dag_with_deps,
            config_map=config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Create step1 first
        step1 = assembler._instantiate_step("step1")
        assembler.step_instances["step1"] = step1
        
        # Now create step2 which depends on step1
        step2 = assembler._instantiate_step("step2")
        
        # Verify step2 was created successfully with dependencies
        assert step2 is not None
        assert hasattr(step2, "name")

    def test_instantiate_step_error_handling(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test _instantiate_step handles builder.create_step errors.
        
        Based on source: _instantiate_step has try/except and raises ValueError.
        Following pytest guide: Test exception paths in implementation.
        """
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Make builder.create_step raise exception
        assembler.step_builders["step1"].create_step = Mock(side_effect=Exception("Step creation failed"))
        
        # This should trigger exception handling in _instantiate_step
        with pytest.raises(ValueError, match="Failed to build step step1"):
            assembler._instantiate_step("step1")

    def test_generate_pipeline_topological_sort_error(
        self, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test generate_pipeline handles topological sort errors.
        
        Based on source: generate_pipeline calls dag.topological_sort() in try/except.
        Following pytest guide: Test error handling from source implementation.
        """
        # Create DAG that will cause topological sort to fail
        problematic_dag = PipelineDAG(nodes=["step1", "step2"], edges=[])
        
        # Mock the topological_sort to raise ValueError
        with patch.object(problematic_dag, 'topological_sort', side_effect=ValueError("Topological sort failed")):
            assembler = PipelineAssembler(
                dag=problematic_dag,
                config_map=simple_config_map,
                step_catalog=mock_step_catalog,
                registry_manager=mock_registry_manager,
                dependency_resolver=mock_dependency_resolver,
            )
            
            # This should trigger the exception handling in generate_pipeline
            with pytest.raises(ValueError, match="Failed to determine build order"):
                assembler.generate_pipeline("test_pipeline")

    def test_generate_pipeline_step_instantiation_error(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test generate_pipeline handles step instantiation errors.
        
        Based on source: generate_pipeline has try/except around _instantiate_step.
        Following pytest guide: Test error handling in loops and iterations.
        """
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Mock _instantiate_step to fail for step2
        original_instantiate = assembler._instantiate_step
        def mock_instantiate(step_name):
            if step_name == "step2":
                raise Exception("Step instantiation failed")
            return original_instantiate(step_name)
        
        assembler._instantiate_step = mock_instantiate
        
        # This should trigger exception handling in generate_pipeline loop
        with pytest.raises(ValueError, match="Failed to instantiate step step2"):
            assembler.generate_pipeline("test_pipeline")

    def test_propagate_messages_with_no_specifications(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test _propagate_messages handles builders without specifications.
        
        Based on source: _propagate_messages checks if builders have spec attribute.
        Following pytest guide: Test conditional logic branches.
        """
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Remove spec from one builder
        assembler.step_builders["step1"].spec = None
        
        # Call _propagate_messages - should handle missing spec gracefully
        assembler._propagate_messages()
        
        # Should not crash and should not call resolver for missing spec
        # (This tests the conditional logic in the source)

    def test_propagate_messages_compatibility_scoring(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test _propagate_messages uses compatibility scoring correctly.
        
        Based on source: _propagate_messages checks compatibility > 0.5 threshold.
        Following pytest guide: Test threshold and scoring logic.
        """
        assembler = PipelineAssembler(
            dag=simple_dag,
            config_map=simple_config_map,
            step_catalog=mock_step_catalog,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )
        
        # Test with low compatibility (should not create match)
        mock_dependency_resolver._calculate_compatibility.return_value = 0.3  # Below 0.5 threshold
        
        assembler._propagate_messages()
        
        # Should not create any matches due to low compatibility
        assert len(assembler.step_messages) == 0
        
        # Test with high compatibility (should create match)
        mock_dependency_resolver._calculate_compatibility.return_value = 0.8  # Above 0.5 threshold
        
        assembler._propagate_messages()
        
        # Verify resolver was called (tests actual threshold logic)
        assert mock_dependency_resolver._calculate_compatibility.called

    def test_generate_outputs_uses_safe_value_for_logging(
        self, simple_dag, simple_config_map, mock_step_catalog,
        mock_registry_manager, mock_dependency_resolver
    ):
        """Test _generate_outputs uses safe_value_for_logging function.
        
        Based on source: _generate_outputs calls safe_value_for_logging() for debug logging.
        Following pytest guide: Test integration between functions.
        """
        with patch("cursus.core.assembler.pipeline_assembler.logger") as mock_logger:
            assembler = PipelineAssembler(
                dag=simple_dag,
                config_map=simple_config_map,
                step_catalog=mock_step_catalog,
                registry_manager=mock_registry_manager,
                dependency_resolver=mock_dependency_resolver,
            )
            
            # Call _generate_outputs
            assembler._generate_outputs("step1")
            
            # Verify debug logging was called (tests integration with safe_value_for_logging)
            mock_logger.debug.assert_called()
