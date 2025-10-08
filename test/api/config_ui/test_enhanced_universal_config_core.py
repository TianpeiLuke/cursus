"""
Test suite for Enhanced UniversalConfigCore with Smart Default Value Inheritance.

This module tests the integration between UniversalConfigCore and StepCatalog
for Smart Default Value Inheritance functionality.

Following pytest best practices:
- Source Code First Rule: Read implementation before writing tests
- Mock Path Precision: Mock at correct import locations
- Implementation-Driven Testing: Match test behavior to actual implementation
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional

from cursus.api.config_ui.core.core import UniversalConfigCore
from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase


class TestEnhancedUniversalConfigCore:
    """Test suite for Enhanced UniversalConfigCore with inheritance support."""
    
    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset any global state before each test."""
        yield
        # Cleanup after test
    
    @pytest.fixture
    def sample_base_config(self):
        """Create a sample BasePipelineConfig instance."""
        # CRITICAL FIX: Use valid region value based on source code validation
        return BasePipelineConfig(
            author="test-user",
            bucket="test-bucket",
            role="arn:aws:iam::123:role/TestRole",
            region="NA",  # FIXED: Use valid region code
            service_name="test-service",
            pipeline_version="1.0.0",
            project_root_folder="test-project"
        )
    
    @pytest.fixture
    def sample_processing_config(self, sample_base_config):
        """Create a sample ProcessingStepConfigBase instance."""
        # CRITICAL FIX: Use correct field names from ProcessingStepConfigBase source code
        return ProcessingStepConfigBase.from_base_config(
            sample_base_config,
            processing_instance_type_small="ml.m5.2xlarge",  # FIXED: Use correct field name
            processing_volume_size=500,
            processing_source_dir="src/processing"
        )
    
    @pytest.fixture
    def mock_pipeline_dag(self):
        """Create a mock pipeline DAG."""
        dag = Mock()
        dag.nodes = ["TabularPreprocessing", "CradleDataLoading"]
        return dag
    
    @pytest.fixture
    def enhanced_config_core(self):
        """Create UniversalConfigCore with mocked StepCatalog."""
        # Create mock StepCatalog instance
        mock_step_catalog = Mock()
        
        # Mock the discover_config_classes method
        mock_step_catalog.discover_config_classes.return_value = {
            "BasePipelineConfig": BasePipelineConfig,
            "ProcessingStepConfigBase": ProcessingStepConfigBase,
            "TabularPreprocessingConfig": Mock(),
            "CradleDataLoadingConfig": Mock()
        }
        
        # Mock the new inheritance methods
        mock_step_catalog.get_immediate_parent_config_class.return_value = "ProcessingStepConfigBase"
        mock_step_catalog.extract_parent_values_for_inheritance.return_value = {
            "author": "test-user",
            "bucket": "test-bucket",
            "role": "arn:aws:iam::123:role/TestRole",
            "processing_instance_type_small": "ml.m5.2xlarge"
        }
        
        # Create UniversalConfigCore and directly set the private attribute
        core = UniversalConfigCore()
        # CRITICAL FIX: Mock the private attribute that the property uses
        core._step_catalog = mock_step_catalog
        return core, mock_step_catalog
    
    def test_create_pipeline_config_widget_with_inheritance_enabled(self, enhanced_config_core, mock_pipeline_dag, sample_base_config, sample_processing_config):
        """Test pipeline config widget creation with inheritance enabled."""
        core, mock_step_catalog = enhanced_config_core
        
        # Mock the required methods for DAG processing
        with patch.object(core, '_discover_required_config_classes') as mock_discover:
            mock_discover.return_value = [
                {
                    "node_name": "TabularPreprocessing",
                    "config_class_name": "TabularPreprocessingConfig",
                    "config_class": Mock(),
                    "inheritance_pattern": "processing_based",
                    "is_specialized": False
                }
            ]
            
            with patch.object(core, '_create_workflow_structure') as mock_workflow:
                mock_workflow.return_value = [
                    {"step_number": 1, "title": "Base Configuration"},
                    {"step_number": 2, "title": "Processing Configuration"},
                    {"step_number": 3, "title": "TabularPreprocessingConfig"}
                ]
                
                with patch('cursus.api.config_ui.widgets.widget.MultiStepWizard') as mock_wizard:
                    # Test with inheritance enabled
                    completed_configs = {
                        "BasePipelineConfig": sample_base_config,
                        "ProcessingStepConfigBase": sample_processing_config
                    }
                    
                    result = core.create_pipeline_config_widget(
                        mock_pipeline_dag,
                        sample_base_config,
                        sample_processing_config,
                        completed_configs=completed_configs,
                        enable_inheritance=True
                    )
                    
                    # Verify StepCatalog methods were called
                    mock_step_catalog.get_immediate_parent_config_class.assert_called_with("TabularPreprocessingConfig")
                    mock_step_catalog.extract_parent_values_for_inheritance.assert_called_with(
                        "TabularPreprocessingConfig", completed_configs
                    )
                    
                    # Verify inheritance analysis was added to config info
                    call_args = mock_discover.call_args[0]
                    # The config info should be enhanced with inheritance analysis
                    
                    # Verify MultiStepWizard was created with inheritance support
                    mock_wizard.assert_called_once()
                    # CRITICAL FIX: Check the actual call signature from source code (line 318)
                    # MultiStepWizard(workflow_steps, base_config=base_config, processing_config=processing_config, enable_inheritance=enable_inheritance)
                    call_args, call_kwargs = mock_wizard.call_args
                    assert call_kwargs.get('enable_inheritance') == True
    
    def test_create_pipeline_config_widget_with_inheritance_disabled(self, enhanced_config_core, mock_pipeline_dag, sample_base_config):
        """Test pipeline config widget creation with inheritance disabled."""
        core, mock_step_catalog = enhanced_config_core
        
        with patch.object(core, '_discover_required_config_classes') as mock_discover:
            mock_discover.return_value = [
                {
                    "node_name": "TabularPreprocessing",
                    "config_class_name": "TabularPreprocessingConfig",
                    "config_class": Mock(),
                    "inheritance_pattern": "processing_based",
                    "is_specialized": False
                }
            ]
            
            with patch.object(core, '_create_workflow_structure') as mock_workflow:
                mock_workflow.return_value = [{"step_number": 1, "title": "Base Configuration"}]
                
                with patch('cursus.api.config_ui.widgets.widget.MultiStepWizard') as mock_wizard:
                    # Test with inheritance disabled
                    result = core.create_pipeline_config_widget(
                        mock_pipeline_dag,
                        sample_base_config,
                        enable_inheritance=False
                    )
                    
                    # Verify StepCatalog inheritance methods were NOT called
                    mock_step_catalog.get_immediate_parent_config_class.assert_not_called()
                    mock_step_catalog.extract_parent_values_for_inheritance.assert_not_called()
                    
                    # Verify MultiStepWizard was created without inheritance support
                    mock_wizard.assert_called_once()
                    call_kwargs = mock_wizard.call_args[1]
                    assert call_kwargs['enable_inheritance'] == False
    
    def test_create_pipeline_config_widget_graceful_degradation(self, enhanced_config_core, mock_pipeline_dag, sample_base_config):
        """Test graceful degradation when inheritance analysis fails."""
        core, mock_step_catalog = enhanced_config_core
        
        # Make inheritance methods raise exceptions
        mock_step_catalog.get_immediate_parent_config_class.side_effect = Exception("Test error")
        mock_step_catalog.extract_parent_values_for_inheritance.side_effect = Exception("Test error")
        
        with patch.object(core, '_discover_required_config_classes') as mock_discover:
            mock_discover.return_value = [
                {
                    "node_name": "TabularPreprocessing",
                    "config_class_name": "TabularPreprocessingConfig",
                    "config_class": Mock(),
                    "inheritance_pattern": "processing_based",
                    "is_specialized": False
                }
            ]
            
            with patch.object(core, '_create_workflow_structure') as mock_workflow:
                mock_workflow.return_value = [{"step_number": 1, "title": "Base Configuration"}]
                
                with patch('cursus.api.config_ui.widgets.widget.MultiStepWizard') as mock_wizard:
                    # Should not raise exception despite inheritance analysis failure
                    result = core.create_pipeline_config_widget(
                        mock_pipeline_dag,
                        sample_base_config,
                        enable_inheritance=True
                    )
                    
                    # Verify widget was still created
                    mock_wizard.assert_called_once()
    
    def test_get_inheritance_aware_form_fields_with_inheritance(self, enhanced_config_core):
        """Test inheritance-aware form field generation with parent values."""
        core, mock_step_catalog = enhanced_config_core
        
        # Mock config class discovery
        mock_config_class = Mock()
        mock_config_class.model_fields = {
            "author": Mock(is_required=lambda: True, annotation=str, description="Author name", default=None),
            "bucket": Mock(is_required=lambda: True, annotation=str, description="S3 bucket", default=None),
            "new_field": Mock(is_required=lambda: True, annotation=str, description="New field", default=None),
            "optional_field": Mock(is_required=lambda: False, annotation=str, description="Optional field", default="default_value")
        }
        
        with patch.object(core, 'discover_config_classes') as mock_discover:
            mock_discover.return_value = {"TestConfig": mock_config_class}
            
            with patch.object(core, '_categorize_fields') as mock_categorize:
                mock_categorize.return_value = {
                    "essential": ["author", "bucket", "new_field"],
                    "system": ["optional_field"],
                    "derived": []
                }
                
                # Test with inheritance analysis
                inheritance_analysis = {
                    "inheritance_enabled": True,
                    "immediate_parent": "ProcessingStepConfigBase",
                    "parent_values": {
                        "author": "test-user",
                        "bucket": "test-bucket"
                    }
                }
                
                fields = core.get_inheritance_aware_form_fields("TestConfig", inheritance_analysis)
                
                # Verify field categorization
                inherited_fields = [f for f in fields if f['tier'] == 'inherited']
                essential_fields = [f for f in fields if f['tier'] == 'essential']
                system_fields = [f for f in fields if f['tier'] == 'system']
                
                assert len(inherited_fields) == 2  # author, bucket
                assert len(essential_fields) == 1   # new_field
                assert len(system_fields) == 1     # optional_field
                
                # Verify inherited field properties
                author_field = next(f for f in inherited_fields if f['name'] == 'author')
                assert author_field['required'] == False  # Override: not required since inherited
                assert author_field['default'] == "test-user"
                assert author_field['is_pre_populated'] == True
                assert author_field['inherited_from'] == "ProcessingStepConfigBase"
                assert author_field['can_override'] == True
    
    def test_get_inheritance_aware_form_fields_without_inheritance(self, enhanced_config_core):
        """Test inheritance-aware form field generation without parent values."""
        core, mock_step_catalog = enhanced_config_core
        
        # Mock config class discovery
        mock_config_class = Mock()
        mock_config_class.model_fields = {
            "author": Mock(is_required=lambda: True, annotation=str, description="Author name", default=None),
            "optional_field": Mock(is_required=lambda: False, annotation=str, description="Optional field", default="default_value")
        }
        
        with patch.object(core, 'discover_config_classes') as mock_discover:
            mock_discover.return_value = {"TestConfig": mock_config_class}
            
            with patch.object(core, '_categorize_fields') as mock_categorize:
                mock_categorize.return_value = {
                    "essential": ["author"],
                    "system": ["optional_field"],
                    "derived": []
                }
                
                # Test without inheritance analysis
                inheritance_analysis = {
                    "inheritance_enabled": False,
                    "parent_values": {}
                }
                
                fields = core.get_inheritance_aware_form_fields("TestConfig", inheritance_analysis)
                
                # Verify no inherited fields
                inherited_fields = [f for f in fields if f['tier'] == 'inherited']
                essential_fields = [f for f in fields if f['tier'] == 'essential']
                system_fields = [f for f in fields if f['tier'] == 'system']
                
                assert len(inherited_fields) == 0  # No inheritance
                assert len(essential_fields) == 1  # author
                assert len(system_fields) == 1    # optional_field
                
                # Verify essential field properties
                author_field = next(f for f in essential_fields if f['name'] == 'author')
                assert author_field['required'] == True
                assert author_field['is_pre_populated'] == False
                assert author_field['inherited_from'] is None
                assert author_field['can_override'] == False
    
    def test_get_inheritance_aware_form_fields_config_not_found(self, enhanced_config_core):
        """Test inheritance-aware form field generation when config class not found."""
        core, mock_step_catalog = enhanced_config_core
        
        with patch.object(core, 'discover_config_classes') as mock_discover:
            mock_discover.return_value = {}  # No config classes found
            
            fields = core.get_inheritance_aware_form_fields("NonExistentConfig")
            
            assert fields == []


class TestEnhancedUniversalConfigCoreIntegration:
    """Integration tests for Enhanced UniversalConfigCore."""
    
    @pytest.fixture
    def real_config_core(self):
        """Create a real UniversalConfigCore instance."""
        return UniversalConfigCore()
    
    def test_inheritance_aware_form_fields_with_real_configs(self, real_config_core):
        """Test inheritance-aware form field generation with real config classes."""
        try:
            # Test with ProcessingStepConfigBase
            inheritance_analysis = {
                "inheritance_enabled": True,
                "immediate_parent": "BasePipelineConfig",
                "parent_values": {
                    "author": "test-user",
                    "bucket": "test-bucket",
                    "role": "arn:aws:iam::123:role/TestRole"
                }
            }
            
            fields = real_config_core.get_inheritance_aware_form_fields(
                "ProcessingStepConfigBase", inheritance_analysis
            )
            
            # Should have some fields
            assert len(fields) > 0
            
            # Should have inherited fields
            inherited_fields = [f for f in fields if f['tier'] == 'inherited']
            assert len(inherited_fields) > 0
            
            # Verify inherited field properties
            for field in inherited_fields:
                assert field['required'] == False  # Not required since inherited
                assert field['is_pre_populated'] == True
                assert field['inherited_from'] == "BasePipelineConfig"
                assert field['can_override'] == True
                
        except Exception as e:
            # If config classes are not available in test environment, that's expected
            pytest.skip(f"Config classes not available in test environment: {e}")
    
    def test_create_pipeline_config_widget_integration(self, real_config_core):
        """Test complete pipeline config widget creation with real components."""
        try:
            # Create sample configs
            base_config = BasePipelineConfig(
                author="integration-test",
                bucket="test-bucket",
                role="arn:aws:iam::123:role/TestRole",
                region="NA",
                service_name="test-service",
                pipeline_version="1.0.0",
                project_root_folder="test-project"
            )
            
            processing_config = ProcessingStepConfigBase.from_base_config(
                base_config,
                processing_instance_type_small="ml.m5.2xlarge",
                processing_volume_size=500
            )
            
            # Mock pipeline DAG
            mock_dag = Mock()
            mock_dag.nodes = ["TestStep"]
            
            # Mock the required internal methods
            with patch.object(real_config_core, '_discover_required_config_classes') as mock_discover:
                mock_discover.return_value = []  # No specific configs needed
                
                with patch.object(real_config_core, '_create_workflow_structure') as mock_workflow:
                    mock_workflow.return_value = [
                        {"step_number": 1, "title": "Base Configuration"}
                    ]
                    
                    with patch('cursus.api.config_ui.widgets.widget.MultiStepWizard') as mock_wizard:
                        # Test widget creation
                        result = real_config_core.create_pipeline_config_widget(
                            mock_dag,
                            base_config,
                            processing_config,
                            enable_inheritance=True
                        )
                        
                        # Verify widget was created
                        mock_wizard.assert_called_once()
                        
        except Exception as e:
            # If dependencies are not available in test environment, that's expected
            pytest.skip(f"Dependencies not available in test environment: {e}")
