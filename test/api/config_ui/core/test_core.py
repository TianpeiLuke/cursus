"""
Comprehensive tests for UniversalConfigCore following pytest best practices.

This test module follows the pytest best practices guide:
1. Source Code First Rule - Read core.py implementation completely before writing tests
2. Mock Path Precision - Mock at exact import locations from source
3. Implementation-Driven Testing - Match test behavior to actual implementation
4. Fixture Isolation - Design fixtures for complete test independence
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import logging

# Following Source Code First Rule - import the actual implementation
from cursus.api.config_ui.core import UniversalConfigCore, create_config_widget
from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase


class TestUniversalConfigCore:
    """Comprehensive tests for UniversalConfigCore following pytest best practices."""
    
    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset any global state before each test (Category 17: Global State Management)."""
        yield
        # Cleanup after test - no global state in UniversalConfigCore currently
    
    @pytest.fixture
    def mock_step_catalog(self):
        """
        Mock step catalog with realistic behavior.
        
        Following Category 1: Mock Path Precision pattern - mock at exact import location.
        Source shows: from cursus.step_catalog.step_catalog import StepCatalog
        The import happens inside the step_catalog property method, so we need to patch it there.
        """
        with patch('cursus.step_catalog.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            # Configure realistic discovery behavior based on source implementation
            mock_catalog.discover_config_classes.return_value = {
                "BasePipelineConfig": BasePipelineConfig,
                "ProcessingStepConfigBase": ProcessingStepConfigBase,
                "CradleDataLoadConfig": Mock(spec=['from_base_config', 'model_fields']),
                "XGBoostTrainingConfig": Mock(spec=['from_base_config', 'model_fields'])
            }
            
            yield mock_catalog
    
    @pytest.fixture
    def temp_workspace(self):
        """Create realistic temporary workspace structure (Category 9: Workspace and Path Resolution)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            
            # Create realistic directory structure based on source expectations
            dev_workspace = workspace_root / "dev1"
            dev_workspace.mkdir(parents=True)
            
            for component_type in ["scripts", "contracts", "specs", "configs"]:
                component_dir = dev_workspace / component_type
                component_dir.mkdir()
                sample_file = component_dir / f"sample_{component_type[:-1]}.py"
                sample_file.write_text(f"# Sample {component_type[:-1]} file")
            
            yield workspace_root
    
    @pytest.fixture
    def example_base_config(self):
        """Create example base configuration for testing."""
        return BasePipelineConfig(
            author="test-user",
            bucket="test-bucket",
            role="test-role",
            region="NA",  # Fixed: use valid region code
            service_name="test-service",
            pipeline_version="1.0.0",
            project_root_folder="test-project"
        )
    
    def test_init_with_workspace_dirs(self, temp_workspace, mock_step_catalog):
        """Test initialization with workspace directories."""
        # Following Source Code First Rule - read core.py __init__ method first
        workspace_dirs = [temp_workspace]
        
        core = UniversalConfigCore(workspace_dirs=workspace_dirs)
        
        # Test actual implementation behavior
        assert core.workspace_dirs == [temp_workspace]  # Path objects, not strings
        assert core.field_types is not None
        assert len(core.field_types) > 0
        assert core._step_catalog is None  # Lazy loading
        assert core._config_classes_cache is None  # Lazy loading
    
    def test_init_without_workspace_dirs(self, mock_step_catalog):
        """Test initialization without workspace directories."""
        core = UniversalConfigCore()
        
        # Test actual implementation behavior
        assert core.workspace_dirs == []
        assert core.field_types is not None
        assert str in core.field_types  # Check specific field type mapping
        assert core.field_types[str] == "text"
    
    def test_step_catalog_lazy_loading_success(self, mock_step_catalog):
        """Test successful step catalog lazy loading."""
        core = UniversalConfigCore()
        
        # First access should trigger lazy loading
        catalog = core.step_catalog
        
        assert catalog == mock_step_catalog
        # Second access should use cached instance
        catalog2 = core.step_catalog
        assert catalog2 == mock_step_catalog
    
    def test_step_catalog_lazy_loading_import_error(self):
        """Test step catalog lazy loading with ImportError."""
        # Mock ImportError during step catalog import
        with patch('cursus.step_catalog.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog_class.side_effect = ImportError("Step catalog not available")
            
            core = UniversalConfigCore()
            catalog = core.step_catalog
            
            assert catalog is None
    
    def test_discover_config_classes_success(self, mock_step_catalog):
        """Test successful config class discovery."""
        # Following Category 2: Mock Configuration pattern
        core = UniversalConfigCore()
        
        result = core.discover_config_classes()
        
        # Test actual implementation behavior - includes both step catalog and base classes
        assert isinstance(result, dict)
        assert "BasePipelineConfig" in result
        assert "ProcessingStepConfigBase" in result
        assert "CradleDataLoadConfig" in result  # From mock step catalog
        assert "XGBoostTrainingConfig" in result  # From mock step catalog
        
        # Verify step catalog was called
        mock_step_catalog.discover_config_classes.assert_called_once()
    
    def test_discover_config_classes_with_caching(self, mock_step_catalog):
        """Test config class discovery caching behavior."""
        core = UniversalConfigCore()
        
        # First call
        result1 = core.discover_config_classes()
        # Second call should use cache
        result2 = core.discover_config_classes()
        
        assert result1 == result2
        # Should only call step catalog once due to caching
        mock_step_catalog.discover_config_classes.assert_called_once()
    
    def test_discover_config_classes_step_catalog_failure(self, mock_step_catalog):
        """Test config class discovery when step catalog fails."""
        # Configure step catalog to raise exception
        mock_step_catalog.discover_config_classes.side_effect = Exception("Discovery failed")
        
        core = UniversalConfigCore()
        result = core.discover_config_classes()
        
        # Should fall back to base classes only
        assert isinstance(result, dict)
        assert "BasePipelineConfig" in result
        assert "ProcessingStepConfigBase" in result
        # Should not include step catalog classes
        assert len(result) == 2  # Only base classes
    
    def test_create_config_widget_success(self, mock_step_catalog, example_base_config):
        """Test successful config widget creation."""
        # Following Category 4: Test Expectations vs Implementation pattern
        core = UniversalConfigCore()
        
        # Mock the widget creation process - import happens from .widget
        with patch('cursus.api.config_ui.widget.UniversalConfigWidget') as mock_widget_class:
            mock_widget = Mock()
            mock_widget_class.return_value = mock_widget
            
            result = core.create_config_widget("BasePipelineConfig", example_base_config)
            
            assert result == mock_widget
            mock_widget_class.assert_called_once()
            
            # Verify form_data structure matches implementation
            call_args = mock_widget_class.call_args[0][0]  # First positional argument
            assert "config_class" in call_args
            assert "config_class_name" in call_args
            assert "fields" in call_args
            assert "values" in call_args
            assert "inheritance_chain" in call_args
            assert call_args["config_class_name"] == "BasePipelineConfig"
    
    def test_create_config_widget_class_not_found(self, mock_step_catalog):
        """Test config widget creation with non-existent class."""
        # Following Category 6: Exception Handling pattern
        core = UniversalConfigCore()
        
        with pytest.raises(ValueError, match="Configuration class 'NonExistentConfig' not found"):
            core.create_config_widget("NonExistentConfig")
    
    def test_create_config_widget_with_from_base_config(self, mock_step_catalog, example_base_config):
        """Test config widget creation using from_base_config method."""
        # Following Category 2: Mock Behavior Matching pattern
        core = UniversalConfigCore()
        
        # Mock config class with from_base_config method and proper field categorization
        mock_config_class = Mock()
        mock_config_class.__name__ = "TestConfig"
        mock_config_class.model_fields = {"field1": Mock(), "field2": Mock()}
        mock_config_instance = Mock()
        mock_config_class.from_base_config.return_value = mock_config_instance
        mock_config_instance.model_dump.return_value = {"test": "data"}
        
        # Update mock discovery to return our mock class
        mock_step_catalog.discover_config_classes.return_value = {
            "BasePipelineConfig": BasePipelineConfig,
            "ProcessingStepConfigBase": ProcessingStepConfigBase,
            "TestConfig": mock_config_class
        }
        
        # Mock the _categorize_fields method to return proper dictionary
        with patch.object(core, '_categorize_fields') as mock_categorize:
            mock_categorize.return_value = {
                "essential": ["field1"],
                "system": ["field2"],
                "derived": []
            }
            
            with patch('cursus.api.config_ui.widget.UniversalConfigWidget') as mock_widget_class:
                core.create_config_widget("TestConfig", example_base_config)
                
                mock_config_class.from_base_config.assert_called_once_with(example_base_config)
                mock_widget_class.assert_called_once()
    
    def test_get_form_fields_pydantic_v2(self, mock_step_catalog):
        """Test form field extraction from Pydantic v2 models."""
        # Following Category 7: Data Structure Fidelity pattern
        core = UniversalConfigCore()
        
        # Test with actual BasePipelineConfig
        result = core._get_form_fields(BasePipelineConfig)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Check field structure matches implementation
        for field in result:
            assert "name" in field
            assert "type" in field
            assert "required" in field
            assert "tier" in field
            assert field["tier"] in ["essential", "system"]  # No derived fields in UI
    
    def test_get_inheritance_chain(self, mock_step_catalog):
        """Test inheritance chain analysis."""
        core = UniversalConfigCore()
        
        # Test with actual ProcessingStepConfigBase
        result = core._get_inheritance_chain(ProcessingStepConfigBase)
        
        assert isinstance(result, list)
        assert "ProcessingStepConfigBase" in result
        # Should not include BasePipelineConfig itself (per implementation)
        assert "BasePipelineConfig" not in result
    
    def test_categorize_fields_with_categorize_method(self, mock_step_catalog):
        """Test field categorization using config class's categorize_fields method."""
        core = UniversalConfigCore()
        
        # Mock config class with categorize_fields method
        mock_config_class = Mock()
        mock_instance = Mock()
        mock_config_class.return_value = mock_instance
        mock_instance.categorize_fields.return_value = {
            "essential": ["field1", "field2"],
            "system": ["field3"],
            "derived": ["field4"]
        }
        
        result = core._categorize_fields(mock_config_class)
        
        assert result["essential"] == ["field1", "field2"]
        assert result["system"] == ["field3"]
        assert result["derived"] == ["field4"]
    
    def test_categorize_fields_fallback_to_manual(self, mock_step_catalog):
        """Test field categorization fallback to manual categorization."""
        core = UniversalConfigCore()
        
        # Test with actual BasePipelineConfig (no categorize_fields method)
        result = core._categorize_fields(BasePipelineConfig)
        
        assert isinstance(result, dict)
        assert "essential" in result
        assert "system" in result
        assert "derived" in result
        assert isinstance(result["essential"], list)
        assert isinstance(result["system"], list)
        assert isinstance(result["derived"], list)
    
    def test_create_pipeline_config_widget_success(self, mock_step_catalog, example_base_config):
        """Test successful pipeline config widget creation."""
        core = UniversalConfigCore()
        
        # Mock pipeline DAG
        mock_dag = Mock()
        mock_dag.nodes = ["step1", "step2"]
        
        # Mock StepConfigResolverAdapter
        with patch('cursus.step_catalog.adapters.config_resolver.StepConfigResolverAdapter') as mock_resolver_class:
            mock_resolver = Mock()
            mock_resolver_class.return_value = mock_resolver
            
            # Mock MultiStepWizard
            with patch('cursus.api.config_ui.widget.MultiStepWizard') as mock_wizard_class:
                mock_wizard = Mock()
                mock_wizard_class.return_value = mock_wizard
                
                result = core.create_pipeline_config_widget(mock_dag, example_base_config)
                
                assert result == mock_wizard
                mock_wizard_class.assert_called_once()
    
    def test_create_pipeline_config_widget_no_resolver(self, mock_step_catalog, example_base_config):
        """Test pipeline config widget creation when resolver is not available."""
        core = UniversalConfigCore()
        
        # Mock pipeline DAG
        mock_dag = Mock()
        mock_dag.nodes = ["step1", "step2"]
        
        # Mock ImportError for StepConfigResolverAdapter
        with patch('cursus.step_catalog.adapters.config_resolver.StepConfigResolverAdapter') as mock_resolver_class:
            mock_resolver_class.side_effect = ImportError("Resolver not available")
            
            with patch('cursus.api.config_ui.widget.MultiStepWizard') as mock_wizard_class:
                mock_wizard = Mock()
                mock_wizard_class.return_value = mock_wizard
                
                result = core.create_pipeline_config_widget(mock_dag, example_base_config)
                
                assert result == mock_wizard
                mock_wizard_class.assert_called_once()
    
    def test_infer_config_class_from_node_name_cradle(self, mock_step_catalog):
        """Test config class inference for Cradle node names."""
        core = UniversalConfigCore()
        
        # Mock available config classes
        mock_step_catalog.discover_config_classes.return_value = {
            "BasePipelineConfig": BasePipelineConfig,
            "ProcessingStepConfigBase": ProcessingStepConfigBase,
            "CradleDataLoadConfig": Mock()
        }
        
        result = core._infer_config_class_from_node_name("cradle_data_load", None)
        
        assert result is not None
        assert result["config_class_name"] == "CradleDataLoadConfig"
        assert result["node_name"] == "cradle_data_load"
        assert result["inferred"] is True
    
    def test_infer_config_class_from_node_name_xgboost(self, mock_step_catalog):
        """Test config class inference for XGBoost node names."""
        core = UniversalConfigCore()
        
        # Mock available config classes
        mock_step_catalog.discover_config_classes.return_value = {
            "BasePipelineConfig": BasePipelineConfig,
            "ProcessingStepConfigBase": ProcessingStepConfigBase,
            "XGBoostTrainingConfig": Mock()
        }
        
        result = core._infer_config_class_from_node_name("xgboost_training", None)
        
        assert result is not None
        assert result["config_class_name"] == "XGBoostTrainingConfig"
        assert result["node_name"] == "xgboost_training"
        assert result["inferred"] is True
    
    def test_infer_config_class_from_node_name_no_match(self, mock_step_catalog):
        """Test config class inference when no match is found."""
        core = UniversalConfigCore()
        
        result = core._infer_config_class_from_node_name("unknown_step", None)
        
        assert result is None
    
    def test_get_inheritance_pattern_processing_based(self, mock_step_catalog):
        """Test inheritance pattern detection for processing-based configs."""
        core = UniversalConfigCore()
        
        result = core._get_inheritance_pattern(ProcessingStepConfigBase)
        
        assert result == "processing_based"
    
    def test_get_inheritance_pattern_base_only(self, mock_step_catalog):
        """Test inheritance pattern detection for base-only configs."""
        core = UniversalConfigCore()
        
        result = core._get_inheritance_pattern(BasePipelineConfig)
        
        assert result == "base_only"
    
    def test_is_specialized_config_cradle(self, mock_step_catalog):
        """Test specialized config detection for CradleDataLoadConfig."""
        core = UniversalConfigCore()
        
        # Mock config class with correct name
        mock_config_class = Mock()
        mock_config_class.__name__ = "CradleDataLoadConfig"
        
        result = core._is_specialized_config(mock_config_class)
        
        assert result is True
    
    def test_is_specialized_config_regular(self, mock_step_catalog):
        """Test specialized config detection for regular configs."""
        core = UniversalConfigCore()
        
        result = core._is_specialized_config(BasePipelineConfig)
        
        assert result is False
    
    def test_create_workflow_structure(self, mock_step_catalog):
        """Test workflow structure creation."""
        core = UniversalConfigCore()
        
        # Mock required configs
        required_configs = [
            {
                "node_name": "test_step",
                "config_class_name": "TestConfig",
                "config_class": Mock(),
                "inheritance_pattern": "base_only",
                "is_specialized": False
            }
        ]
        
        result = core._create_workflow_structure(required_configs)
        
        assert isinstance(result, list)
        assert len(result) >= 2  # Base + specific step
        
        # Check base configuration step
        base_step = result[0]
        assert base_step["title"] == "Base Configuration"
        assert base_step["config_class"] == BasePipelineConfig
        assert base_step["type"] == "base"
        assert base_step["required"] is True
    
    def test_create_workflow_structure_with_processing(self, mock_step_catalog):
        """Test workflow structure creation with processing-based configs."""
        core = UniversalConfigCore()
        
        # Mock required configs with processing-based inheritance
        required_configs = [
            {
                "node_name": "test_step",
                "config_class_name": "TestConfig",
                "config_class": Mock(),
                "inheritance_pattern": "processing_based",
                "is_specialized": False
            }
        ]
        
        result = core._create_workflow_structure(required_configs)
        
        assert len(result) >= 3  # Base + Processing + specific step
        
        # Check processing configuration step is added
        processing_step = result[1]
        assert processing_step["title"] == "Processing Configuration"
        assert processing_step["config_class"] == ProcessingStepConfigBase
        assert processing_step["type"] == "processing"


class TestCreateConfigWidgetFactory:
    """Test the factory function for creating config widgets."""
    
    def test_create_config_widget_factory_success(self):
        """Test successful widget creation via factory function."""
        with patch('cursus.api.config_ui.core.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_widget = Mock()
            mock_core_class.return_value = mock_core
            mock_core.create_config_widget.return_value = mock_widget
            
            result = create_config_widget("BasePipelineConfig")
            
            assert result == mock_widget
            mock_core.create_config_widget.assert_called_once_with("BasePipelineConfig", None)
    
    def test_create_config_widget_factory_with_base_config(self):
        """Test factory function with base config parameter."""
        base_config = BasePipelineConfig(
            author="test-user",
            bucket="test-bucket",
            role="test-role",
            region="NA",  # Fixed: use valid region code
            service_name="test-service",
            pipeline_version="1.0.0",
            project_root_folder="test-project"
        )
        
        with patch('cursus.api.config_ui.core.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_widget = Mock()
            mock_core_class.return_value = mock_core
            mock_core.create_config_widget.return_value = mock_widget
            
            result = create_config_widget("ProcessingStepConfigBase", base_config)
            
            assert result == mock_widget
            mock_core.create_config_widget.assert_called_once_with("ProcessingStepConfigBase", base_config)
    
    def test_create_config_widget_factory_with_workspace_dirs(self):
        """Test factory function with workspace directories."""
        workspace_dirs = ["/path/to/workspace"]
        
        with patch('cursus.api.config_ui.core.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_widget = Mock()
            mock_core_class.return_value = mock_core
            mock_core.create_config_widget.return_value = mock_widget
            
            result = create_config_widget("BasePipelineConfig", workspace_dirs=workspace_dirs)
            
            assert result == mock_widget
            # Verify UniversalConfigCore was initialized with workspace_dirs
            mock_core_class.assert_called_once_with(workspace_dirs=workspace_dirs)
            mock_core.create_config_widget.assert_called_once_with("BasePipelineConfig", None)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases following pytest best practices."""
    
    def test_step_catalog_initialization_failure(self):
        """Test handling of step catalog initialization failure."""
        # Following Category 16: Exception Handling vs Test Expectations
        with patch('cursus.step_catalog.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog_class.side_effect = ImportError("Step catalog not available")
            
            # Should handle gracefully, not crash
            core = UniversalConfigCore()
            catalog = core.step_catalog
            
            # Should fall back gracefully
            assert catalog is None
    
    def test_field_categorization_exception_handling(self):
        """Test field categorization with exception in categorize_fields method."""
        core = UniversalConfigCore()
        
        # Mock config class that raises exception in categorize_fields
        mock_config_class = Mock()
        mock_config_class.__name__ = "TestConfig"  # Add required __name__ attribute
        mock_instance = Mock()
        mock_config_class.return_value = mock_instance
        mock_instance.categorize_fields.side_effect = Exception("Categorization failed")
        
        # Should fall back to manual categorization
        result = core._categorize_fields(mock_config_class)
        
        assert isinstance(result, dict)
        assert "essential" in result
        assert "system" in result
        assert "derived" in result
    
    def test_form_fields_extraction_with_invalid_config_class(self):
        """Test form field extraction with invalid config class."""
        core = UniversalConfigCore()
        
        # Mock config class without model_fields or __init__ signature
        mock_config_class = Mock()
        del mock_config_class.model_fields  # Remove model_fields attribute
        mock_config_class.__init__ = Mock(side_effect=Exception("No signature"))
        
        # Should handle gracefully and return empty list
        result = core._get_form_fields(mock_config_class)
        
        assert isinstance(result, list)
        # May be empty or have fallback fields
    
    def test_create_config_widget_pre_population_failure(self):
        """Test config widget creation when pre-population fails."""
        with patch('cursus.step_catalog.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            # Mock config class that fails during from_base_config
            mock_config_class = Mock()
            mock_config_class.__name__ = "TestConfig"
            mock_config_class.model_fields = {"field1": Mock(), "field2": Mock()}
            mock_config_class.from_base_config.side_effect = Exception("Pre-population failed")
            
            mock_catalog.discover_config_classes.return_value = {
                "BasePipelineConfig": BasePipelineConfig,
                "ProcessingStepConfigBase": ProcessingStepConfigBase,
                "TestConfig": mock_config_class
            }
            
            core = UniversalConfigCore()
            base_config = BasePipelineConfig(
                author="test-user",
                bucket="test-bucket",
                role="test-role",
                region="NA",  # Fixed: use valid region code
                service_name="test-service",
                pipeline_version="1.0.0",
                project_root_folder="test-project"
            )
            
            # Mock the _categorize_fields method to return proper dictionary
            with patch.object(core, '_categorize_fields') as mock_categorize:
                mock_categorize.return_value = {
                    "essential": ["field1"],
                    "system": ["field2"],
                    "derived": []
                }
                
                with patch('cursus.api.config_ui.widget.UniversalConfigWidget') as mock_widget_class:
                    mock_widget = Mock()
                    mock_widget_class.return_value = mock_widget
                    
                    # Should handle gracefully and still create widget
                    result = core.create_config_widget("TestConfig", base_config)
                    
                    assert result == mock_widget
                    mock_widget_class.assert_called_once()
    
    def test_pipeline_dag_without_nodes_attribute(self):
        """Test pipeline config widget creation with DAG that has no nodes attribute."""
        with patch('cursus.step_catalog.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            core = UniversalConfigCore()
            
            # Mock pipeline DAG without nodes attribute
            mock_dag = Mock()
            del mock_dag.nodes  # Remove nodes attribute
            
            base_config = BasePipelineConfig(
                author="test-user",
                bucket="test-bucket",
                role="test-role",
                region="NA",  # Fixed: use valid region code
                service_name="test-service",
                pipeline_version="1.0.0",
                project_root_folder="test-project"
            )
            
            with patch('cursus.api.config_ui.widget.MultiStepWizard') as mock_wizard_class:
                mock_wizard = Mock()
                mock_wizard_class.return_value = mock_wizard
                
                # Should handle gracefully
                result = core.create_pipeline_config_widget(mock_dag, base_config)
                
                assert result == mock_wizard
                mock_wizard_class.assert_called_once()
    
    def test_logging_behavior(self, caplog):
        """Test that appropriate logging messages are generated."""
        with caplog.at_level(logging.INFO):
            core = UniversalConfigCore(workspace_dirs=[Path("/test")])
            
            # Check initialization logging
            assert "UniversalConfigCore initialized" in caplog.text
    
    def test_field_types_mapping_completeness(self):
        """Test that field types mapping covers expected types."""
        core = UniversalConfigCore()
        
        # Verify expected field type mappings
        expected_mappings = {
            str: "text",
            int: "number",
            float: "number", 
            bool: "checkbox",
            list: "list",
            dict: "keyvalue"
        }
        
        for python_type, expected_ui_type in expected_mappings.items():
            assert core.field_types[python_type] == expected_ui_type
    
    def test_workspace_dirs_string_to_path_conversion(self):
        """Test that string workspace directories are converted to Path objects."""
        workspace_dirs = ["/path/to/workspace", "/another/path"]
        
        core = UniversalConfigCore(workspace_dirs=workspace_dirs)
        
        # Should convert strings to Path objects
        assert all(isinstance(path, Path) for path in core.workspace_dirs)
        assert core.workspace_dirs[0] == Path("/path/to/workspace")
        assert core.workspace_dirs[1] == Path("/another/path")
    
    def test_mixed_workspace_dirs_types(self):
        """Test initialization with mixed string and Path workspace directories."""
        workspace_dirs = ["/string/path", Path("/path/object")]
        
        core = UniversalConfigCore(workspace_dirs=workspace_dirs)
        
        # Should handle mixed types correctly
        assert all(isinstance(path, Path) for path in core.workspace_dirs)
        assert len(core.workspace_dirs) == 2


class TestIntegrationScenarios:
    """Integration tests for complete workflows following pytest best practices."""
    
    def test_end_to_end_widget_creation_workflow(self):
        """Test complete widget creation workflow from start to finish."""
        # Following Category 4: Test Expectations vs Implementation
        with patch('cursus.step_catalog.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            # Configure comprehensive mock discovery
            mock_catalog.discover_config_classes.return_value = {
                "BasePipelineConfig": BasePipelineConfig,
                "ProcessingStepConfigBase": ProcessingStepConfigBase,
                "CradleDataLoadConfig": Mock(spec=['from_base_config', 'model_fields']),
                "XGBoostTrainingConfig": Mock(spec=['from_base_config', 'model_fields'])
            }
            
            base_config = BasePipelineConfig(
                author="test-user",
                bucket="test-bucket",
                role="test-role",
                region="NA",  # Fixed: use valid region code
                service_name="test-service",
                pipeline_version="1.0.0",
                project_root_folder="test-project"
            )
            
            with patch('cursus.api.config_ui.core.UniversalConfigWidget') as mock_widget_class:
                mock_widget = Mock()
                mock_widget_class.return_value = mock_widget
                
                # Test widget creation for multiple config types
                core = UniversalConfigCore()
                
                widget1 = core.create_config_widget("BasePipelineConfig", base_config)
                widget2 = core.create_config_widget("ProcessingStepConfigBase", base_config)
                
                assert widget1 == mock_widget
                assert widget2 == mock_widget
                assert mock_widget_class.call_count == 2
    
    def test_pipeline_workflow_with_multiple_steps(self):
        """Test complete pipeline configuration workflow with multiple steps."""
        with patch('cursus.step_catalog.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            # Mock comprehensive step catalog
            mock_catalog.discover_config_classes.return_value = {
                "BasePipelineConfig": BasePipelineConfig,
                "ProcessingStepConfigBase": ProcessingStepConfigBase,
                "CradleDataLoadConfig": Mock(),
                "XGBoostTrainingConfig": Mock(),
                "TabularPreprocessingConfig": Mock()
            }
            
            core = UniversalConfigCore()
            
            # Mock complex pipeline DAG
            mock_dag = Mock()
            mock_dag.nodes = ["cradle_data_load", "xgboost_training", "tabular_preprocessing"]
            
            base_config = BasePipelineConfig(
                author="test-user",
                bucket="test-bucket",
                role="test-role",
                region="NA",  # Fixed: use valid region code
                service_name="test-service",
                pipeline_version="1.0.0",
                project_root_folder="test-project"
            )
            
            with patch('cursus.api.config_ui.core.StepConfigResolverAdapter') as mock_resolver_class:
                mock_resolver = Mock()
                mock_resolver_class.return_value = mock_resolver
                
                with patch('cursus.api.config_ui.core.MultiStepWizard') as mock_wizard_class:
                    mock_wizard = Mock()
                    mock_wizard_class.return_value = mock_wizard
                    
                    result = core.create_pipeline_config_widget(mock_dag, base_config)
                    
                    assert result == mock_wizard
                    mock_wizard_class.assert_called_once()
                    
                    # Verify workflow steps were created
                    call_args = mock_wizard_class.call_args[0][0]  # First positional argument
                    assert isinstance(call_args, list)
                    assert len(call_args) >= 1  # At least base configuration step
    
    def test_factory_function_integration(self):
        """Test integration of factory function with core functionality."""
        with patch('cursus.api.config_ui.core.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_widget = Mock()
            mock_core_class.return_value = mock_core
            mock_core.create_config_widget.return_value = mock_widget
            
            # Test factory function with various parameters
            result1 = create_config_widget("BasePipelineConfig")
            result2 = create_config_widget("ProcessingStepConfigBase", workspace_dirs=["/test"])
            
            assert result1 == mock_widget
            assert result2 == mock_widget
            assert mock_core.create_config_widget.call_count == 2
            
            # Verify core was initialized with correct parameters
            assert mock_core_class.call_count == 2
            mock_core_class.assert_any_call(workspace_dirs=None)
            mock_core_class.assert_any_call(workspace_dirs=["/test"])
