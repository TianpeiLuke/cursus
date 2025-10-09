"""
Test Smart Default Value Inheritance in Multi-Step Wizard

This test verifies that user data from the base config page is properly
populated as default values in the processing config page and subsequent steps.

Based on analysis of the actual source code:
- MultiStepWizard._save_current_step() stores configs with both step title and class name
- MultiStepWizard._get_step_fields() calls UniversalConfigCore.get_inheritance_aware_form_fields()
- UniversalConfigCore uses StepCatalog.get_immediate_parent_config_class() and extract_parent_values_for_inheritance()
- The key fix is storing configs with class name keys for inheritance lookup
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase
from cursus.api.config_ui.widgets.widget import MultiStepWizard, UniversalConfigWidget
from cursus.api.config_ui.core.core import UniversalConfigCore


class TestSmartDefaultValueInheritance:
    """Test suite for Smart Default Value Inheritance in Multi-Step Wizard."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test base configuration with user data
        self.test_base_config = BasePipelineConfig(
            author="lukexie",
            bucket="test-sagemaker-bucket",
            role="arn:aws:iam::123456789012:role/TestSageMakerRole",
            region="NA",
            service_name="test-service",
            pipeline_version="1.0.0",
            project_root_folder="test-project"
        )
        
        # Create mock DAG for testing
        self.mock_dag = Mock()
        self.mock_dag.nodes = ["TestStep1", "TestStep2"]
        
        # Create test steps for multi-step wizard
        self.test_steps = [
            {
                "step_number": 1,
                "title": "Base Configuration",
                "config_class": BasePipelineConfig,
                "config_class_name": "BasePipelineConfig",
                "type": "base",
                "required": True
            },
            {
                "step_number": 2,
                "title": "Processing Configuration", 
                "config_class": ProcessingStepConfigBase,
                "config_class_name": "ProcessingStepConfigBase",
                "type": "processing",
                "required": True
            }
        ]
    
    def test_base_config_storage_with_inheritance_keys(self):
        """Test that base config is stored with proper keys for inheritance."""
        # Create multi-step wizard with inheritance enabled
        wizard = MultiStepWizard(
            steps=self.test_steps,
            base_config=self.test_base_config,
            enable_inheritance=True
        )
        
        # Simulate user filling out base config form
        wizard.current_step = 0
        
        # Mock the step widget with user input
        mock_widget = Mock()
        mock_widget.widgets = {
            "author": Mock(value="lukexie"),
            "bucket": Mock(value="test-sagemaker-bucket"),
            "role": Mock(value="arn:aws:iam::123456789012:role/TestSageMakerRole"),
            "region": Mock(value="NA"),
            "service_name": Mock(value="test-service"),
            "pipeline_version": Mock(value="1.0.0"),
            "project_root_folder": Mock(value="test-project")
        }
        mock_widget.fields = [
            {"name": "author", "type": "text"},
            {"name": "bucket", "type": "text"},
            {"name": "role", "type": "text"},
            {"name": "region", "type": "text"},
            {"name": "service_name", "type": "text"},
            {"name": "pipeline_version", "type": "text"},
            {"name": "project_root_folder", "type": "text"}
        ]
        
        wizard.step_widgets[0] = mock_widget
        
        # Save current step (base config)
        success = wizard._save_current_step()
        
        # Verify save was successful
        assert success, "Base config save should succeed"
        
        # Verify config is stored with both step title and class name
        assert "Base Configuration" in wizard.completed_configs
        assert "BasePipelineConfig" in wizard.completed_configs
        
        # Verify both references point to the same config instance
        base_config_by_title = wizard.completed_configs["Base Configuration"]
        base_config_by_class = wizard.completed_configs["BasePipelineConfig"]
        assert base_config_by_title is base_config_by_class
        
        # Verify base_config reference is updated
        assert wizard.base_config is base_config_by_class
        
        # Verify config contains user data
        assert base_config_by_class.author == "lukexie"
        assert base_config_by_class.bucket == "test-sagemaker-bucket"
        assert base_config_by_class.role == "arn:aws:iam::123456789012:role/TestSageMakerRole"
    
    def test_processing_config_inherits_base_values(self):
        """Test that processing config step shows inherited values from base config."""
        # Create multi-step wizard with inheritance enabled
        wizard = MultiStepWizard(
            steps=self.test_steps,
            base_config=self.test_base_config,
            enable_inheritance=True
        )
        
        # Simulate completed base config
        wizard.completed_configs["BasePipelineConfig"] = self.test_base_config
        wizard.base_config = self.test_base_config
        
        # Move to processing config step
        wizard.current_step = 1
        
        # Mock UniversalConfigCore to test inheritance
        with patch('cursus.api.config_ui.core.core.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            
            # Mock step catalog for inheritance analysis
            mock_step_catalog = Mock()
            mock_core.step_catalog = mock_step_catalog
            mock_step_catalog.get_immediate_parent_config_class.return_value = "BasePipelineConfig"
            mock_step_catalog.extract_parent_values_for_inheritance.return_value = {
                "author": "lukexie",
                "bucket": "test-sagemaker-bucket", 
                "role": "arn:aws:iam::123456789012:role/TestSageMakerRole",
                "region": "NA",
                "service_name": "test-service",
                "pipeline_version": "1.0.0",
                "project_root_folder": "test-project"
            }
            
            # Mock inheritance-aware field generation
            mock_core.get_inheritance_aware_form_fields.return_value = [
                {
                    "name": "author",
                    "type": "text",
                    "required": False,
                    "tier": "inherited",
                    "default": "lukexie",
                    "is_pre_populated": True,
                    "inherited_from": "BasePipelineConfig",
                    "inheritance_note": "Auto-filled from BasePipelineConfig"
                },
                {
                    "name": "bucket", 
                    "type": "text",
                    "required": False,
                    "tier": "inherited",
                    "default": "test-sagemaker-bucket",
                    "is_pre_populated": True,
                    "inherited_from": "BasePipelineConfig",
                    "inheritance_note": "Auto-filled from BasePipelineConfig"
                },
                {
                    "name": "processing_instance_count",
                    "type": "number",
                    "required": False,
                    "tier": "system",
                    "default": 1,
                    "is_pre_populated": False
                }
            ]
            
            # Get step fields for processing config
            step = self.test_steps[1]
            fields = wizard._get_step_fields(step)
            
            # Verify inheritance analysis was called
            mock_core.get_inheritance_aware_form_fields.assert_called_once()
            
            # Verify inherited fields are present
            inherited_fields = [f for f in fields if f.get('tier') == 'inherited']
            assert len(inherited_fields) >= 2, "Should have inherited fields from base config"
            
            # Verify specific inherited values
            author_field = next((f for f in fields if f['name'] == 'author'), None)
            assert author_field is not None, "Author field should be present"
            assert author_field['tier'] == 'inherited', "Author should be inherited"
            assert author_field['default'] == 'lukexie', "Author should have inherited value"
            assert author_field['is_pre_populated'] == True, "Author should be pre-populated"
            
            bucket_field = next((f for f in fields if f['name'] == 'bucket'), None)
            assert bucket_field is not None, "Bucket field should be present"
            assert bucket_field['tier'] == 'inherited', "Bucket should be inherited"
            assert bucket_field['default'] == 'test-sagemaker-bucket', "Bucket should have inherited value"
    
    def test_inheritance_analysis_creation(self):
        """Test that inheritance analysis is created correctly."""
        wizard = MultiStepWizard(
            steps=self.test_steps,
            base_config=self.test_base_config,
            enable_inheritance=True
        )
        
        # Add completed base config
        wizard.completed_configs["BasePipelineConfig"] = self.test_base_config
        
        # Mock UniversalConfigCore and step catalog
        with patch('cursus.api.config_ui.widgets.widget.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            
            mock_step_catalog = Mock()
            mock_core.step_catalog = mock_step_catalog
            mock_step_catalog.get_immediate_parent_config_class.return_value = "BasePipelineConfig"
            mock_step_catalog.extract_parent_values_for_inheritance.return_value = {
                "author": "lukexie",
                "bucket": "test-sagemaker-bucket"
            }
            
            # Create inheritance analysis
            analysis = wizard._create_inheritance_analysis("ProcessingStepConfigBase")
            
            # Verify analysis structure
            assert analysis['inheritance_enabled'] == True
            assert analysis['immediate_parent'] == "BasePipelineConfig"
            assert analysis['parent_values']['author'] == "lukexie"
            assert analysis['parent_values']['bucket'] == "test-sagemaker-bucket"
            assert analysis['total_inherited_fields'] == 2
    
    def test_inheritance_disabled_fallback(self):
        """Test behavior when inheritance is disabled."""
        wizard = MultiStepWizard(
            steps=self.test_steps,
            base_config=self.test_base_config,
            enable_inheritance=False  # Inheritance disabled
        )
        
        # Move to processing config step
        wizard.current_step = 1
        
        # Mock UniversalConfigCore
        with patch('cursus.api.config_ui.widgets.widget.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            
            # Mock standard field generation (no inheritance)
            mock_core._get_form_fields.return_value = [
                {
                    "name": "processing_instance_count",
                    "type": "number", 
                    "required": False,
                    "tier": "system",
                    "default": 1
                }
            ]
            
            # Get step fields
            step = self.test_steps[1]
            fields = wizard._get_step_fields(step)
            
            # Verify standard field generation was used
            mock_core._get_form_fields.assert_called_once()
            mock_core.get_inheritance_aware_form_fields.assert_not_called()
    
    def test_processing_config_storage_updates_reference(self):
        """Test that processing config storage updates the processing_config reference."""
        wizard = MultiStepWizard(
            steps=self.test_steps,
            base_config=self.test_base_config,
            enable_inheritance=True
        )
        
        # Move to processing config step
        wizard.current_step = 1
        
        # Mock the step widget with processing config data (include all required fields)
        mock_widget = Mock()
        mock_widget.widgets = {
            "author": Mock(value="lukexie"),
            "bucket": Mock(value="test-sagemaker-bucket"),
            "role": Mock(value="arn:aws:iam::123456789012:role/TestSageMakerRole"),
            "region": Mock(value="NA"),
            "service_name": Mock(value="test-service"),
            "pipeline_version": Mock(value="1.0.0"),
            "project_root_folder": Mock(value="test-project"),
            "processing_instance_count": Mock(value=2)
        }
        mock_widget.fields = [
            {"name": "author", "type": "text"},
            {"name": "bucket", "type": "text"},
            {"name": "role", "type": "text"},
            {"name": "region", "type": "text"},
            {"name": "service_name", "type": "text"},
            {"name": "pipeline_version", "type": "text"},
            {"name": "project_root_folder", "type": "text"},
            {"name": "processing_instance_count", "type": "number"}
        ]
        
        wizard.step_widgets[1] = mock_widget
        
        # Save processing config step
        success = wizard._save_current_step()
        
        # Verify save was successful
        assert success, "Processing config save should succeed"
        
        # Verify config is stored with both keys
        assert "Processing Configuration" in wizard.completed_configs
        assert "ProcessingStepConfigBase" in wizard.completed_configs
        
        # Verify processing_config reference is updated
        processing_config = wizard.completed_configs["ProcessingStepConfigBase"]
        assert wizard.processing_config is processing_config
        
        # Verify config contains inherited and new data
        assert processing_config.author == "lukexie"  # Inherited
        assert processing_config.bucket == "test-sagemaker-bucket"  # Inherited
        assert processing_config.processing_instance_count == 2  # New
    
    def test_end_to_end_inheritance_workflow(self):
        """Test complete end-to-end inheritance workflow."""
        # Create wizard
        wizard = MultiStepWizard(
            steps=self.test_steps,
            base_config=None,  # Start with no base config
            enable_inheritance=True
        )
        
        # Step 1: Fill and save base config
        wizard.current_step = 0
        mock_base_widget = Mock()
        mock_base_widget.widgets = {
            "author": Mock(value="lukexie"),
            "bucket": Mock(value="test-bucket"),
            "role": Mock(value="test-role"),
            "region": Mock(value="NA"),
            "service_name": Mock(value="test-service"),
            "pipeline_version": Mock(value="1.0.0"),
            "project_root_folder": Mock(value="test-project")
        }
        mock_base_widget.fields = [
            {"name": field, "type": "text"} 
            for field in mock_base_widget.widgets.keys()
        ]
        wizard.step_widgets[0] = mock_base_widget
        
        # Save base config
        assert wizard._save_current_step(), "Base config save should succeed"
        
        # Verify base config is available for inheritance
        assert "BasePipelineConfig" in wizard.completed_configs
        base_config = wizard.completed_configs["BasePipelineConfig"]
        assert base_config.author == "lukexie"
        
        # Step 2: Move to processing config and verify inheritance
        wizard.current_step = 1
        
        with patch('cursus.api.config_ui.widgets.widget.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            
            # Mock successful inheritance
            mock_step_catalog = Mock()
            mock_core.step_catalog = mock_step_catalog
            mock_step_catalog.get_immediate_parent_config_class.return_value = "BasePipelineConfig"
            mock_step_catalog.extract_parent_values_for_inheritance.return_value = {
                "author": "lukexie",
                "bucket": "test-bucket",
                "role": "test-role"
            }
            
            # Create inheritance analysis
            analysis = wizard._create_inheritance_analysis("ProcessingStepConfigBase")
            
            # Verify inheritance works
            assert analysis['inheritance_enabled'] == True
            assert analysis['parent_values']['author'] == "lukexie"
            assert len(analysis['parent_values']) == 3
    
    def test_inheritance_error_handling(self):
        """Test graceful handling of inheritance errors."""
        wizard = MultiStepWizard(
            steps=self.test_steps,
            base_config=self.test_base_config,
            enable_inheritance=True
        )
        
        # Mock UniversalConfigCore to raise exception
        with patch('cursus.api.config_ui.widgets.widget.UniversalConfigCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            mock_core.step_catalog = None  # No step catalog available
            
            # Create inheritance analysis with no step catalog
            analysis = wizard._create_inheritance_analysis("ProcessingStepConfigBase")
            
            # Verify graceful degradation
            assert analysis['inheritance_enabled'] == False
            assert analysis['immediate_parent'] is None
            assert analysis['parent_values'] == {}
            assert analysis['total_inherited_fields'] == 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
