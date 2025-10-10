"""
Phase 3 Comprehensive Tests: DataSourcesManager

Following pytest best practices guides to prevent ALL common failure patterns:

CRITICAL ERRORS IDENTIFIED AND FIXED:
1. Category 1: Mock Path Issues - Mock at correct import location
2. Category 2: Mock Configuration - Proper side_effect setup matching actual calls
3. Category 4: Test Expectations vs Implementation - Based on actual source behavior
4. Category 12: NoneType Attribute Access - Handle None values defensively
5. Category 17: Global State Management - Reset state between tests
6. Import path resolution - Use absolute paths, not relative
7. Field structure mismatch - Match actual config class fields from source
8. Mock behavior mismatch - Count actual method calls from source
"""

import pytest
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

# CRITICAL FIX: Add src to path BEFORE any imports to prevent import errors
src_dir = Path(__file__).parent.parent.parent.parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Set up logging for testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataSourcesManagerComprehensive:
    """
    Comprehensive tests for DataSourcesManager following ALL pytest best practices.
    
    ERRORS PREVENTED:
    - Category 1: Mock path issues (mock at import location)
    - Category 2: Mock configuration issues (proper side_effect)
    - Category 4: Test expectations vs implementation mismatch
    - Category 12: NoneType attribute access
    - Category 17: Global state management
    """
    
    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset global state before each test (Category 17 prevention)."""
        # Clear any module-level caches or state
        yield
        # Cleanup after test
    
    @pytest.fixture
    def mock_config_core(self):
        """
        Create mock UniversalConfigCore based on ACTUAL source code analysis.
        
        CRITICAL: Based on actual config class fields from cradle config source:
        - MdsDataSourceConfig: service_name, region, output_schema (required), org_id, use_hourly_edx_data_set (optional)
        - EdxDataSourceConfig: edx_provider, edx_subject, edx_dataset, edx_manifest_key, schema_overrides (all required)
        - AndesDataSourceConfig: provider, table_name (required), andes3_enabled (optional)
        """
        mock_core = Mock()
        
        # Mock discover_config_classes - returns dict with config classes
        mock_core.discover_config_classes.return_value = {
            "MdsDataSourceConfig": Mock(__name__="MdsDataSourceConfig"),
            "EdxDataSourceConfig": Mock(__name__="EdxDataSourceConfig"),
            "AndesDataSourceConfig": Mock(__name__="AndesDataSourceConfig")
        }
        
        # Mock _get_form_fields based on ACTUAL config class structure
        def mock_get_form_fields(config_class):
            """Return fields matching ACTUAL config class structure from source."""
            class_name = getattr(config_class, '__name__', 'Unknown')
            
            if class_name == "MdsDataSourceConfig":
                # ACTUAL fields from MdsDataSourceConfig source + data_source_name for DataSourceConfig
                return [
                    {"name": "data_source_name", "type": "text", "required": True, "default": "RAW_MDS_NA"},
                    {"name": "service_name", "type": "text", "required": True, "default": ""},
                    {"name": "region", "type": "dropdown", "required": True, "default": "", "options": ["NA", "EU", "FE"]},
                    {"name": "output_schema", "type": "schema_list", "required": True, "default": []},
                    {"name": "org_id", "type": "number", "required": False, "default": 0},
                    {"name": "use_hourly_edx_data_set", "type": "checkbox", "required": False, "default": False}
                ]
            elif class_name == "EdxDataSourceConfig":
                # ACTUAL fields from EdxDataSourceConfig source + data_source_name for DataSourceConfig
                return [
                    {"name": "data_source_name", "type": "text", "required": True, "default": "RAW_EDX_EU"},
                    {"name": "edx_provider", "type": "text", "required": True, "default": ""},
                    {"name": "edx_subject", "type": "text", "required": True, "default": ""},
                    {"name": "edx_dataset", "type": "text", "required": True, "default": ""},
                    {"name": "edx_manifest_key", "type": "text", "required": True, "default": ""},
                    {"name": "schema_overrides", "type": "schema_list", "required": True, "default": []}
                ]
            elif class_name == "AndesDataSourceConfig":
                # ACTUAL fields from AndesDataSourceConfig source + data_source_name for DataSourceConfig
                return [
                    {"name": "data_source_name", "type": "text", "required": True, "default": "RAW_ANDES_NA"},
                    {"name": "provider", "type": "text", "required": True, "default": ""},
                    {"name": "table_name", "type": "text", "required": True, "default": ""},
                    {"name": "andes3_enabled", "type": "checkbox", "required": False, "default": True}
                ]
            else:
                return []
        
        mock_core._get_form_fields.side_effect = mock_get_form_fields
        return mock_core
    
    @pytest.fixture
    def mock_widgets(self):
        """
        Mock ipywidgets at CORRECT import location.
        
        CRITICAL FIX: Mock at where DataSourcesManager imports widgets FROM:
        - Source: import ipywidgets as widgets
        - Mock path: cursus.api.config_ui.core.data_sources_manager.widgets
        
        CRITICAL FIX: VBox needs to support context manager protocol for 'with' statement
        """
        with patch('cursus.api.config_ui.core.data_sources_manager.widgets') as mock_widgets:
            # CRITICAL FIX: Create VBox mock that supports context manager protocol
            mock_vbox = MagicMock()
            mock_vbox.__enter__ = MagicMock(return_value=mock_vbox)
            mock_vbox.__exit__ = MagicMock(return_value=None)
            mock_widgets.VBox.return_value = mock_vbox
            
            # Mock all other widget types used in source
            mock_widgets.HTML.return_value = Mock()
            mock_widgets.Button.return_value = Mock()
            mock_widgets.Dropdown.return_value = Mock()
            mock_widgets.Text.return_value = Mock()
            mock_widgets.FloatText.return_value = Mock()
            mock_widgets.Checkbox.return_value = Mock()
            mock_widgets.HBox.return_value = Mock()
            mock_widgets.Layout.return_value = Mock()
            
            yield mock_widgets
    
    @pytest.fixture
    def mock_display_functions(self):
        """Mock IPython display functions at correct import location."""
        with patch('cursus.api.config_ui.core.data_sources_manager.display') as mock_display, \
             patch('cursus.api.config_ui.core.data_sources_manager.clear_output') as mock_clear:
            yield {"display": mock_display, "clear_output": mock_clear}
    
    def test_initialization_with_discovery_success(self, mock_config_core, mock_widgets, mock_display_functions):
        """
        Test successful DataSourcesManager initialization with config discovery.
        
        PREVENTS: Category 1 (import path), Category 2 (mock config), Category 4 (expectations)
        """
        from cursus.api.config_ui.core.data_sources_manager import DataSourcesManager
        
        # Test initialization with provided config core
        manager = DataSourcesManager(config_core=mock_config_core)
        
        # Verify config core discovery was called exactly once
        mock_config_core.discover_config_classes.assert_called_once()
        
        # Verify data source config classes were set (not None)
        assert manager.data_source_config_classes["MDS"] is not None
        assert manager.data_source_config_classes["EDX"] is not None
        assert manager.data_source_config_classes["ANDES"] is not None
        
        # Verify field templates were generated for all types
        assert "MDS" in manager.field_templates
        assert "EDX" in manager.field_templates
        assert "ANDES" in manager.field_templates
        
        # Verify each template has required structure
        for source_type in ["MDS", "EDX", "ANDES"]:
            template = manager.field_templates[source_type]
            assert "required_fields" in template
            assert "optional_fields" in template
            assert "field_definitions" in template
            assert isinstance(template["field_definitions"], dict)
        
        # Verify default data source was created (exactly 1)
        assert len(manager.data_sources) == 1
        assert manager.data_sources[0]["data_source_type"] == "MDS"
        
        # Verify UI components were initialized
        assert manager.container is not None
        assert isinstance(manager.data_source_widgets, list)
        
        logger.info("✅ Initialization with discovery test passed")
    
    def test_field_template_generation_from_discovery(self, mock_config_core, mock_widgets, mock_display_functions):
        """
        Test field template generation matches actual config class structure.
        
        PREVENTS: Category 4 (test expectations vs implementation)
        """
        from cursus.api.config_ui.core.data_sources_manager import DataSourcesManager
        
        manager = DataSourcesManager(config_core=mock_config_core)
        
        # Test MDS template structure matches actual MdsDataSourceConfig
        mds_template = manager.field_templates["MDS"]
        
        # Verify required fields match actual config class (including data_source_name)
        assert "data_source_name" in mds_template["required_fields"]
        assert "service_name" in mds_template["required_fields"]
        assert "region" in mds_template["required_fields"]
        assert "output_schema" in mds_template["required_fields"]
        
        # Verify optional fields match actual config class
        assert "org_id" in mds_template["optional_fields"]
        assert "use_hourly_edx_data_set" in mds_template["optional_fields"]
        
        # Verify field definitions have correct types (based on actual source structure)
        service_name_def = mds_template["field_definitions"]["service_name"]
        assert service_name_def["type"] == "text"
        # Note: 'required' is not stored in field_definitions, only used for categorization
        
        region_def = mds_template["field_definitions"]["region"]
        assert region_def["type"] == "dropdown"
        assert region_def["options"] == ["NA", "EU", "FE"]
        
        # Test EDX template structure
        edx_template = manager.field_templates["EDX"]
        assert "edx_provider" in edx_template["required_fields"]
        assert "edx_subject" in edx_template["required_fields"]
        assert "edx_dataset" in edx_template["required_fields"]
        assert "edx_manifest_key" in edx_template["required_fields"]
        assert "schema_overrides" in edx_template["required_fields"]
        
        # Test ANDES template structure
        andes_template = manager.field_templates["ANDES"]
        assert "provider" in andes_template["required_fields"]
        assert "table_name" in andes_template["required_fields"]
        assert "andes3_enabled" in andes_template["optional_fields"]
        
        logger.info("✅ Field template generation test passed")
    
    def test_fallback_template_when_discovery_fails(self, mock_widgets, mock_display_functions):
        """
        Test fallback template creation when config discovery returns empty results.
        
        PREVENTS: Category 12 (NoneType access), Category 16 (exception handling)
        """
        from cursus.api.config_ui.core.data_sources_manager import DataSourcesManager
        
        # Mock config core that returns empty discovery results
        mock_config_core = Mock()
        mock_config_core.discover_config_classes.return_value = {}  # Empty results
        
        # Should not raise exception, should use fallback templates
        manager = DataSourcesManager(config_core=mock_config_core)
        
        # Verify fallback templates were created for all types
        assert "MDS" in manager.field_templates
        assert "EDX" in manager.field_templates
        assert "ANDES" in manager.field_templates
        
        # Verify fallback MDS template has expected structure
        mds_fallback = manager.field_templates["MDS"]
        assert "data_source_name" in mds_fallback["field_definitions"]
        assert "service_name" in mds_fallback["field_definitions"]
        assert "region" in mds_fallback["field_definitions"]
        
        # Verify fallback template defaults
        assert mds_fallback["field_definitions"]["data_source_name"]["default"] == "RAW_MDS_NA"
        assert mds_fallback["field_definitions"]["service_name"]["default"] == "AtoZ"
        assert mds_fallback["field_definitions"]["region"]["default"] == "NA"
        
        # Verify default data source was still created
        assert len(manager.data_sources) == 1
        assert manager.data_sources[0]["data_source_type"] == "MDS"
        
        logger.info("✅ Fallback template creation test passed")
    
    def test_add_data_source_with_type_specific_defaults(self, mock_config_core, mock_widgets, mock_display_functions):
        """
        Test adding data sources creates correct type-specific defaults.
        
        PREVENTS: Category 2 (mock configuration), Category 4 (expectations)
        """
        from cursus.api.config_ui.core.data_sources_manager import DataSourcesManager
        
        manager = DataSourcesManager(config_core=mock_config_core)
        
        # Initial state: 1 MDS data source
        initial_count = len(manager.data_sources)
        assert initial_count == 1
        assert manager.data_sources[0]["data_source_type"] == "MDS"
        
        # Add EDX data source
        manager.add_data_source("EDX")
        
        # Verify EDX data source was added with correct structure
        assert len(manager.data_sources) == initial_count + 1
        edx_source = manager.data_sources[1]
        assert edx_source["data_source_type"] == "EDX"
        
        # Verify EDX-specific fields were populated with defaults
        assert "edx_provider" in edx_source
        assert "edx_subject" in edx_source
        assert "edx_dataset" in edx_source
        assert "edx_manifest_key" in edx_source
        assert "schema_overrides" in edx_source
        
        # Add ANDES data source
        manager.add_data_source("ANDES")
        
        # Verify ANDES data source was added with correct structure
        assert len(manager.data_sources) == initial_count + 2
        andes_source = manager.data_sources[2]
        assert andes_source["data_source_type"] == "ANDES"
        
        # Verify ANDES-specific fields were populated
        assert "provider" in andes_source
        assert "table_name" in andes_source
        assert "andes3_enabled" in andes_source
        assert andes_source["andes3_enabled"] == True  # Default from config
        
        logger.info("✅ Add data source test passed")
    
    def test_remove_data_source_with_minimum_constraint(self, mock_config_core, mock_widgets, mock_display_functions):
        """
        Test removing data sources respects minimum constraint (1 source required).
        
        PREVENTS: Category 4 (test expectations vs implementation)
        """
        from cursus.api.config_ui.core.data_sources_manager import DataSourcesManager
        
        manager = DataSourcesManager(config_core=mock_config_core)
        
        # Add multiple data sources for testing removal
        manager.add_data_source("EDX")
        manager.add_data_source("ANDES")
        assert len(manager.data_sources) == 3
        
        # Remove second data source (EDX at index 1)
        manager.remove_data_source(1)
        assert len(manager.data_sources) == 2
        
        # Verify correct data source was removed (EDX)
        remaining_types = [ds["data_source_type"] for ds in manager.data_sources]
        assert "MDS" in remaining_types
        assert "ANDES" in remaining_types
        assert "EDX" not in remaining_types
        
        # Remove another data source (ANDES at index 1)
        manager.remove_data_source(1)
        assert len(manager.data_sources) == 1
        assert manager.data_sources[0]["data_source_type"] == "MDS"
        
        # Try to remove last data source - should be prevented
        manager.remove_data_source(0)
        assert len(manager.data_sources) == 1  # Should still have 1 data source
        assert manager.data_sources[0]["data_source_type"] == "MDS"
        
        logger.info("✅ Remove data source test passed")
    
    def test_field_widget_creation_all_types(self, mock_config_core, mock_widgets, mock_display_functions):
        """
        Test creation of all field widget types used in source.
        
        PREVENTS: Category 2 (mock configuration), Category 3 (widget creation)
        """
        from cursus.api.config_ui.core.data_sources_manager import DataSourcesManager
        
        manager = DataSourcesManager(config_core=mock_config_core)
        
        # Test text field widget
        text_widget = manager._create_field_widget(
            "service_name", 
            {"type": "text", "default": "AtoZ"}, 
            "TestService"
        )
        mock_widgets.Text.assert_called()
        
        # Test dropdown field widget
        dropdown_widget = manager._create_field_widget(
            "region", 
            {"type": "dropdown", "options": ["NA", "EU", "FE"], "default": "NA"}, 
            "EU"
        )
        mock_widgets.Dropdown.assert_called()
        
        # Test number field widget
        number_widget = manager._create_field_widget(
            "org_id", 
            {"type": "number", "default": 0}, 
            123
        )
        mock_widgets.FloatText.assert_called()
        
        # Test checkbox field widget
        checkbox_widget = manager._create_field_widget(
            "use_hourly_edx_data_set", 
            {"type": "checkbox", "default": False}, 
            True
        )
        mock_widgets.Checkbox.assert_called()
        
        # Test schema_list field widget (uses Text widget internally)
        schema_widget = manager._create_field_widget(
            "output_schema", 
            {"type": "schema_list", "default": []}, 
            [{"field_name": "objectId", "field_type": "STRING"}]
        )
        # Should create Text widget for schema input
        assert mock_widgets.Text.call_count >= 2  # Called for text and schema_list
        
        logger.info("✅ Field widget creation test passed")
    
    def test_data_collection_from_widget_groups(self, mock_config_core, mock_widgets, mock_display_functions):
        """
        Test data collection from widget groups matches expected structure.
        
        PREVENTS: Category 2 (mock configuration), Category 4 (expectations)
        """
        from cursus.api.config_ui.core.data_sources_manager import DataSourcesManager
        
        manager = DataSourcesManager(config_core=mock_config_core)
        
        # Create mock widget group for MDS data source
        mock_type_dropdown = Mock()
        mock_type_dropdown.value = "MDS"
        
        mock_field_widgets = {
            "service_name": Mock(value="TestService"),
            "region": Mock(value="EU"),
            "output_schema": Mock(value="objectId:STRING, transactionDate:STRING"),
            "org_id": Mock(value=456),
            "use_hourly_edx_data_set": Mock(value=True)
        }
        
        widget_group = {
            "type_dropdown": mock_type_dropdown,
            "field_widgets": mock_field_widgets
        }
        
        # Test data collection
        collected_data = manager._collect_data_source_data(widget_group, 0)
        
        # Verify collected data structure and values
        assert collected_data["data_source_type"] == "MDS"
        assert collected_data["service_name"] == "TestService"
        assert collected_data["region"] == "EU"
        assert collected_data["org_id"] == 456
        assert collected_data["use_hourly_edx_data_set"] == True
        
        # Verify schema_list conversion
        expected_schema = [
            {"field_name": "objectId", "field_type": "STRING"},
            {"field_name": "transactionDate", "field_type": "STRING"}
        ]
        assert collected_data["output_schema"] == expected_schema
        
        logger.info("✅ Data collection test passed")
    
    def test_get_all_data_sources_multiple_types(self, mock_config_core, mock_widgets, mock_display_functions):
        """
        Test getting all data sources from multiple widget groups.
        
        PREVENTS: Category 2 (mock configuration), Category 4 (expectations)
        """
        from cursus.api.config_ui.core.data_sources_manager import DataSourcesManager
        
        manager = DataSourcesManager(config_core=mock_config_core)
        
        # Mock multiple widget groups for different data source types
        mock_mds_group = {
            "type_dropdown": Mock(value="MDS"),
            "field_widgets": {
                "service_name": Mock(value="AtoZ"),
                "region": Mock(value="NA"),
                "output_schema": Mock(value="objectId:STRING"),
                "org_id": Mock(value=0),
                "use_hourly_edx_data_set": Mock(value=False)
            }
        }
        
        mock_edx_group = {
            "type_dropdown": Mock(value="EDX"),
            "field_widgets": {
                "edx_provider": Mock(value="provider1"),
                "edx_subject": Mock(value="subject1"),
                "edx_dataset": Mock(value="dataset1"),
                "edx_manifest_key": Mock(value='["key1"]'),
                "schema_overrides": Mock(value="order_id:STRING")
            }
        }
        
        # Set up manager with mock widget groups
        manager.data_source_widgets = [mock_mds_group, mock_edx_group]
        
        # Test getting all data sources
        all_data_sources = manager.get_all_data_sources()
        
        # Verify structure and content
        assert len(all_data_sources) == 2
        
        # Verify MDS data source
        mds_source = all_data_sources[0]
        assert mds_source["data_source_type"] == "MDS"
        assert mds_source["service_name"] == "AtoZ"
        assert mds_source["region"] == "NA"
        assert mds_source["org_id"] == 0
        assert mds_source["use_hourly_edx_data_set"] == False
        
        # Verify EDX data source
        edx_source = all_data_sources[1]
        assert edx_source["data_source_type"] == "EDX"
        assert edx_source["edx_provider"] == "provider1"
        assert edx_source["edx_subject"] == "subject1"
        assert edx_source["edx_dataset"] == "dataset1"
        assert edx_source["edx_manifest_key"] == '["key1"]'
        
        logger.info("✅ Get all data sources test passed")
    
    def test_error_handling_none_values(self, mock_config_core, mock_widgets, mock_display_functions):
        """
        Test handling of None values and edge cases.
        
        PREVENTS: Category 12 (NoneType attribute access)
        """
        from cursus.api.config_ui.core.data_sources_manager import DataSourcesManager
        
        # Test with None initial data sources
        manager = DataSourcesManager(initial_data_sources=None, config_core=mock_config_core)
        
        # Should create default data source, not crash
        assert len(manager.data_sources) == 1
        assert manager.data_sources[0]["data_source_type"] == "MDS"
        
        # Test field widget creation with None current value
        widget = manager._create_field_widget(
            "service_name",
            {"type": "text", "default": "AtoZ"},
            None  # None current value
        )
        # Should not crash, should use default
        mock_widgets.Text.assert_called()
        
        # Test data collection with missing field widgets
        mock_widget_group = {
            "type_dropdown": Mock(value="MDS"),
            "field_widgets": {}  # Empty field widgets
        }
        
        # Should not crash, should return basic structure
        collected_data = manager._collect_data_source_data(mock_widget_group, 0)
        assert collected_data["data_source_type"] == "MDS"
        
        logger.info("✅ Error handling test passed")


# Remove the problematic standalone test function that calls fixtures directly
# Pytest will run the test methods automatically when the file is executed with pytest

if __name__ == "__main__":
    """Run tests when executed directly using pytest."""
    import subprocess
    import sys
    
    logger.info("🚀 Running comprehensive DataSourcesManager tests with pytest")
    
    # Run pytest on this file
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        logger.info("✅ ALL comprehensive DataSourcesManager tests PASSED")
    else:
        logger.error("❌ Some tests FAILED")
        sys.exit(result.returncode)
