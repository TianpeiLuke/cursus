"""
Comprehensive tests for field definitions following pytest best practices.

This test module follows the pytest best practices guide:
1. Source Code First Rule - Read field_definitions.py implementation completely before writing tests
2. Mock Path Precision - Mock at exact import locations from source
3. Implementation-Driven Testing - Match test behavior to actual implementation
4. Fixture Isolation - Design fixtures for complete test independence
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Following Source Code First Rule - import the actual implementation
from cursus.api.config_ui.core.field_definitions import get_cradle_data_loading_fields


class TestCradleDataLoadingFieldDefinitions:
    """Comprehensive tests for cradle data loading field definitions following pytest best practices."""
    
    def test_get_cradle_data_loading_fields_returns_list(self):
        """Test that get_cradle_data_loading_fields returns a list."""
        # Following Category 1: Basic Function Behavior
        result = get_cradle_data_loading_fields()
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_field_structure_completeness(self):
        """Test that all fields have required structure."""
        # Following Category 7: Data Structure Fidelity pattern
        fields = get_cradle_data_loading_fields()
        
        required_keys = ["name", "type", "tier"]
        optional_keys = ["required", "default", "description", "options", "placeholder", "conditional", "height", "language"]
        
        for field in fields:
            # Check required keys
            for key in required_keys:
                assert key in field, f"Field {field.get('name', 'unknown')} missing required key: {key}"
            
            # Check that all keys are either required or optional
            for key in field.keys():
                assert key in required_keys + optional_keys, f"Field {field['name']} has unexpected key: {key}"
    
    def test_field_names_uniqueness(self):
        """Test that all field names are unique."""
        # Following Category 8: Uniqueness and Duplication Prevention
        fields = get_cradle_data_loading_fields()
        field_names = [field["name"] for field in fields]
        
        assert len(field_names) == len(set(field_names)), "Duplicate field names found"
    
    def test_tier_categorization_completeness(self):
        """Test that all fields are properly categorized into tiers."""
        # Following Category 7: Data Structure Fidelity pattern
        fields = get_cradle_data_loading_fields()
        valid_tiers = ["inherited", "essential", "system"]
        
        tier_counts = {"inherited": 0, "essential": 0, "system": 0}
        
        for field in fields:
            tier = field["tier"]
            assert tier in valid_tiers, f"Field {field['name']} has invalid tier: {tier}"
            tier_counts[tier] += 1
        
        # Verify we have fields in each tier
        assert tier_counts["inherited"] > 0, "No inherited fields found"
        assert tier_counts["essential"] > 0, "No essential fields found"
        assert tier_counts["system"] > 0, "No system fields found"
    
    def test_inherited_fields_structure(self):
        """Test inherited fields have correct structure and values."""
        # Following Category 4: Test Expectations vs Implementation pattern
        fields = get_cradle_data_loading_fields()
        inherited_fields = [f for f in fields if f["tier"] == "inherited"]
        
        expected_inherited = ["author", "bucket", "role", "region", "service_name", "pipeline_version", "project_root_folder"]
        inherited_names = [f["name"] for f in inherited_fields]
        
        for expected_name in expected_inherited:
            assert expected_name in inherited_names, f"Missing expected inherited field: {expected_name}"
        
        # Check specific inherited field properties
        for field in inherited_fields:
            if field["name"] == "region":
                assert field["type"] == "dropdown"
                assert "options" in field
                assert "NA" in field["options"]
    
    def test_essential_fields_structure(self):
        """Test essential fields have correct structure and requirements."""
        # Following Category 4: Test Expectations vs Implementation pattern
        fields = get_cradle_data_loading_fields()
        essential_fields = [f for f in fields if f["tier"] == "essential"]
        
        # Check that essential fields include key data loading fields
        essential_names = [f["name"] for f in essential_fields]
        expected_essential = ["start_date", "end_date", "data_source_name", "transform_sql", "job_type", "cradle_account"]
        
        for expected_name in expected_essential:
            assert expected_name in essential_names, f"Missing expected essential field: {expected_name}"
        
        # Check specific essential field properties
        for field in essential_fields:
            if field["name"] == "start_date":
                assert field["type"] == "datetime"
                assert field["required"] is True
                assert "placeholder" in field
            elif field["name"] == "transform_sql":
                assert field["type"] == "code_editor"
                assert field["required"] is True
                assert field.get("language") == "sql"
            elif field["name"] == "job_type":
                assert field["type"] == "radio"
                assert field["required"] is True
                assert "options" in field
                assert "training" in field["options"]
    
    def test_system_fields_structure(self):
        """Test system fields have correct structure and defaults."""
        # Following Category 4: Test Expectations vs Implementation pattern
        fields = get_cradle_data_loading_fields()
        system_fields = [f for f in fields if f["tier"] == "system"]
        
        # Check that system fields include configuration options
        system_names = [f["name"] for f in system_fields]
        expected_system = ["output_format", "cluster_type", "job_retry_count", "split_job"]
        
        for expected_name in expected_system:
            assert expected_name in system_names, f"Missing expected system field: {expected_name}"
        
        # Check specific system field properties
        for field in system_fields:
            if field["name"] == "output_format":
                assert field["type"] == "dropdown"
                assert "default" in field
                assert field["default"] == "PARQUET"
                assert "options" in field
            elif field["name"] == "split_job":
                assert field["type"] == "checkbox"
                assert "default" in field
                assert field["default"] is False
    
    def test_field_types_validity(self):
        """Test that all field types are valid and supported."""
        # Following Category 7: Data Structure Fidelity pattern
        fields = get_cradle_data_loading_fields()
        
        valid_field_types = [
            "text", "datetime", "code_editor", "tag_list", "radio", "dropdown", 
            "textarea", "number", "checkbox", "list", "keyvalue"
        ]
        
        for field in fields:
            field_type = field["type"]
            assert field_type in valid_field_types, f"Field {field['name']} has invalid type: {field_type}"
    
    def test_conditional_fields_structure(self):
        """Test conditional fields have proper conditional logic."""
        # Following Category 7: Data Structure Fidelity pattern
        fields = get_cradle_data_loading_fields()
        conditional_fields = [f for f in fields if "conditional" in f]
        
        # Should have conditional fields for different data source types
        assert len(conditional_fields) > 0, "No conditional fields found"
        
        # Check MDS-specific fields
        mds_fields = [f for f in conditional_fields if f.get("conditional") == "data_source_type==MDS"]
        assert len(mds_fields) > 0, "No MDS-specific conditional fields found"
        
        mds_field_names = [f["name"] for f in mds_fields]
        expected_mds_fields = ["mds_service", "mds_region", "mds_org_id", "mds_use_hourly"]
        
        for expected_name in expected_mds_fields:
            assert expected_name in mds_field_names, f"Missing expected MDS field: {expected_name}"
        
        # Check EDX-specific fields
        edx_fields = [f for f in conditional_fields if f.get("conditional") == "data_source_type==EDX"]
        assert len(edx_fields) > 0, "No EDX-specific conditional fields found"
        
        edx_field_names = [f["name"] for f in edx_fields]
        expected_edx_fields = ["edx_provider", "edx_subject", "edx_dataset", "edx_manifest_key"]
        
        for expected_name in expected_edx_fields:
            assert expected_name in edx_field_names, f"Missing expected EDX field: {expected_name}"
        
        # Check ANDES-specific fields
        andes_fields = [f for f in conditional_fields if f.get("conditional") == "data_source_type==ANDES"]
        assert len(andes_fields) > 0, "No ANDES-specific conditional fields found"
        
        andes_field_names = [f["name"] for f in andes_fields]
        expected_andes_fields = ["andes_provider", "andes_table_name", "andes3_enabled"]
        
        for expected_name in expected_andes_fields:
            assert expected_name in andes_field_names, f"Missing expected ANDES field: {expected_name}"
    
    def test_required_fields_specification(self):
        """Test that required fields are properly specified."""
        # Following Category 4: Test Expectations vs Implementation pattern
        fields = get_cradle_data_loading_fields()
        required_fields = [f for f in fields if f.get("required", False)]
        
        # Should have required fields
        assert len(required_fields) > 0, "No required fields found"
        
        # Check specific required fields
        required_names = [f["name"] for f in required_fields]
        expected_required = ["author", "bucket", "role", "service_name", "pipeline_version", 
                           "project_root_folder", "start_date", "end_date", "data_source_name", 
                           "transform_sql", "job_type", "cradle_account"]
        
        for expected_name in expected_required:
            assert expected_name in required_names, f"Expected required field not marked as required: {expected_name}"
    
    def test_default_values_consistency(self):
        """Test that default values are consistent and appropriate."""
        # Following Category 7: Data Structure Fidelity pattern
        fields = get_cradle_data_loading_fields()
        fields_with_defaults = [f for f in fields if "default" in f]
        
        # Should have fields with defaults
        assert len(fields_with_defaults) > 0, "No fields with default values found"
        
        # Check specific default values
        for field in fields_with_defaults:
            default_value = field["default"]
            field_type = field["type"]
            
            # Type consistency checks
            if field_type == "checkbox":
                assert isinstance(default_value, bool), f"Checkbox field {field['name']} has non-boolean default"
            elif field_type == "number":
                assert isinstance(default_value, (int, float)), f"Number field {field['name']} has non-numeric default"
            elif field_type == "tag_list":
                assert isinstance(default_value, list), f"Tag list field {field['name']} has non-list default"
            elif field_type == "dropdown":
                if "options" in field:
                    assert default_value in field["options"], f"Dropdown field {field['name']} default not in options"
    
    def test_description_completeness(self):
        """Test that important fields have descriptions."""
        # Following Category 7: Data Structure Fidelity pattern
        fields = get_cradle_data_loading_fields()
        fields_with_descriptions = [f for f in fields if "description" in f and f["description"]]
        
        # Should have descriptions for most fields
        assert len(fields_with_descriptions) > len(fields) * 0.7, "Too few fields have descriptions"
        
        # Check that complex fields have descriptions
        complex_field_types = ["code_editor", "tag_list", "datetime", "radio"]
        complex_fields = [f for f in fields if f["type"] in complex_field_types]
        
        for field in complex_fields:
            assert "description" in field and field["description"], f"Complex field {field['name']} missing description"
    
    def test_placeholder_values_appropriateness(self):
        """Test that placeholder values are appropriate for field types."""
        # Following Category 7: Data Structure Fidelity pattern
        fields = get_cradle_data_loading_fields()
        fields_with_placeholders = [f for f in fields if "placeholder" in f]
        
        # Should have placeholders for input fields
        assert len(fields_with_placeholders) > 0, "No fields with placeholders found"
        
        # Check specific placeholder appropriateness
        for field in fields_with_placeholders:
            placeholder = field["placeholder"]
            field_type = field["type"]
            
            if field_type == "datetime":
                assert "YYYY-MM-DD" in placeholder, f"Datetime field {field['name']} has inappropriate placeholder"
            elif field_type == "tag_list":
                assert "comma" in placeholder.lower(), f"Tag list field {field['name']} should mention comma separation"
    
    def test_options_completeness_for_dropdown_fields(self):
        """Test that dropdown fields have appropriate options."""
        # Following Category 7: Data Structure Fidelity pattern
        fields = get_cradle_data_loading_fields()
        dropdown_fields = [f for f in fields if f["type"] == "dropdown"]
        
        # Should have dropdown fields
        assert len(dropdown_fields) > 0, "No dropdown fields found"
        
        # Check that all dropdown fields have options
        for field in dropdown_fields:
            assert "options" in field, f"Dropdown field {field['name']} missing options"
            assert isinstance(field["options"], list), f"Dropdown field {field['name']} options not a list"
            assert len(field["options"]) > 0, f"Dropdown field {field['name']} has empty options"
            
            # Check specific dropdown options
            if field["name"] == "data_source_type":
                expected_options = ["MDS", "EDX", "ANDES"]
                for option in expected_options:
                    assert option in field["options"], f"Missing data source type option: {option}"
            elif field["name"] == "output_format":
                expected_options = ["PARQUET", "CSV", "JSON"]
                for option in expected_options:
                    assert option in field["options"], f"Missing output format option: {option}"
    
    def test_radio_button_fields_structure(self):
        """Test radio button fields have proper structure."""
        # Following Category 7: Data Structure Fidelity pattern
        fields = get_cradle_data_loading_fields()
        radio_fields = [f for f in fields if f["type"] == "radio"]
        
        # Should have radio fields
        assert len(radio_fields) > 0, "No radio fields found"
        
        # Check job_type radio field specifically
        job_type_field = next((f for f in radio_fields if f["name"] == "job_type"), None)
        assert job_type_field is not None, "job_type radio field not found"
        assert "options" in job_type_field, "job_type field missing options"
        
        expected_job_types = ["training", "validation", "testing", "calibration"]
        for job_type in expected_job_types:
            assert job_type in job_type_field["options"], f"Missing job type option: {job_type}"
    
    def test_code_editor_fields_configuration(self):
        """Test code editor fields have proper configuration."""
        # Following Category 7: Data Structure Fidelity pattern
        fields = get_cradle_data_loading_fields()
        code_editor_fields = [f for f in fields if f["type"] == "code_editor"]
        
        # Should have code editor fields
        assert len(code_editor_fields) > 0, "No code editor fields found"
        
        # Check transform_sql field specifically
        transform_sql_field = next((f for f in code_editor_fields if f["name"] == "transform_sql"), None)
        assert transform_sql_field is not None, "transform_sql code editor field not found"
        assert transform_sql_field.get("language") == "sql", "transform_sql field should specify SQL language"
        assert "height" in transform_sql_field, "transform_sql field should specify height"
    
    def test_field_count_expectations(self):
        """Test that we have the expected number of fields."""
        # Following Category 4: Test Expectations vs Implementation pattern
        fields = get_cradle_data_loading_fields()
        
        # Should have comprehensive field coverage (47 fields as per design)
        assert len(fields) >= 40, f"Expected at least 40 fields, got {len(fields)}"
        assert len(fields) <= 50, f"Expected at most 50 fields, got {len(fields)}"
    
    def test_data_source_type_field_structure(self):
        """Test the critical data_source_type field structure."""
        # Following Category 4: Test Expectations vs Implementation pattern
        fields = get_cradle_data_loading_fields()
        
        data_source_type_field = next((f for f in fields if f["name"] == "data_source_type"), None)
        assert data_source_type_field is not None, "data_source_type field not found"
        
        # Critical field properties
        assert data_source_type_field["type"] == "dropdown"
        assert data_source_type_field["tier"] == "essential"
        assert "default" in data_source_type_field
        assert data_source_type_field["default"] == "MDS"
        assert "options" in data_source_type_field
        assert set(data_source_type_field["options"]) == {"MDS", "EDX", "ANDES"}


class TestFieldDefinitionsIntegration:
    """Integration tests for field definitions with other components."""
    
    def test_field_definitions_integration_with_core(self):
        """Test that field definitions integrate properly with UniversalConfigCore."""
        # Following Category 4: Test Expectations vs Implementation pattern
        from cursus.api.config_ui.core.core import UniversalConfigCore
        
        with patch('cursus.step_catalog.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            mock_catalog.discover_config_classes.return_value = {}
            
            core = UniversalConfigCore()
            
            # Mock CradleDataLoadingConfig to test field discovery
            mock_config_class = Mock()
            mock_config_class.__name__ = "CradleDataLoadingConfig"
            
            # Should use field definitions for CradleDataLoadingConfig
            with patch('cursus.api.config_ui.core.field_definitions.get_cradle_data_loading_fields') as mock_get_fields:
                mock_fields = [{"name": "test_field", "type": "text", "tier": "system"}]
                mock_get_fields.return_value = mock_fields
                
                result = core._get_form_fields(mock_config_class)
                
                mock_get_fields.assert_called_once()
                assert result == mock_fields
    
    def test_field_definitions_completeness_for_transformation(self):
        """Test that field definitions include all fields needed for data transformation."""
        # Following Category 7: Data Structure Fidelity pattern
        fields = get_cradle_data_loading_fields()
        field_names = [f["name"] for f in fields]
        
        # Fields needed for ui_data transformation
        required_for_transformation = [
            # Root level fields
            "job_type", "author", "bucket", "role", "region", "service_name", 
            "pipeline_version", "project_root_folder",
            
            # Data sources spec
            "start_date", "end_date", "data_source_name", "data_source_type",
            
            # Transform spec
            "transform_sql", "split_job", "days_per_split", "merge_sql",
            
            # Output spec
            "output_schema", "output_format", "output_save_mode", "output_file_count",
            "keep_dot_in_output_schema", "include_header_in_s3_output",
            
            # Cradle job spec
            "cradle_account", "cluster_type", "extra_spark_job_arguments", "job_retry_count"
        ]
        
        for required_field in required_for_transformation:
            assert required_field in field_names, f"Missing field required for transformation: {required_field}"
    
    def test_field_definitions_data_source_specific_fields(self):
        """Test that all data source specific fields are present."""
        # Following Category 7: Data Structure Fidelity pattern
        fields = get_cradle_data_loading_fields()
        field_names = [f["name"] for f in fields]
        
        # MDS-specific fields
        mds_fields = ["mds_service", "mds_region", "mds_org_id", "mds_use_hourly"]
        for field_name in mds_fields:
            assert field_name in field_names, f"Missing MDS-specific field: {field_name}"
        
        # EDX-specific fields
        edx_fields = ["edx_provider", "edx_subject", "edx_dataset", "edx_manifest_key", "edx_schema_overrides"]
        for field_name in edx_fields:
            assert field_name in field_names, f"Missing EDX-specific field: {field_name}"
        
        # ANDES-specific fields
        andes_fields = ["andes_provider", "andes_table_name", "andes3_enabled"]
        for field_name in andes_fields:
            assert field_name in field_names, f"Missing ANDES-specific field: {field_name}"


class TestCradleDataLoadingConfigIntegration:
    """Test that CradleDataLoadingConfig uses comprehensive field definitions instead of specialized widgets."""
    
    def test_cradle_config_not_in_specialized_registry(self):
        """Test that CradleDataLoadingConfig is NOT in the specialized widget registry."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # This verifies our fix - CradleDataLoadingConfig should be removed from specialized registry
        
        from cursus.api.config_ui.widgets.specialized_widgets import SpecializedComponentRegistry
        
        registry = SpecializedComponentRegistry()
        
        # CRITICAL: CradleDataLoadingConfig should NOT be in specialized registry
        has_specialized = registry.has_specialized_component("CradleDataLoadingConfig")
        assert not has_specialized, "CradleDataLoadingConfig should NOT be in specialized registry after fix"
        
        # Verify it's not in the SPECIALIZED_COMPONENTS dict
        assert "CradleDataLoadingConfig" not in registry.SPECIALIZED_COMPONENTS, \
            "CradleDataLoadingConfig should be removed from SPECIALIZED_COMPONENTS"
    
    def test_cradle_config_uses_comprehensive_field_definitions(self):
        """Test that CradleDataLoadingConfig uses comprehensive field definitions from field_definitions.py."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # This verifies the core integration works correctly
        
        from cursus.api.config_ui.core.core import UniversalConfigCore
        
        # Mock the config class
        mock_config_class = Mock()
        mock_config_class.__name__ = "CradleDataLoadingConfig"
        
        core = UniversalConfigCore()
        
        # Should use comprehensive field definitions
        fields = core._get_form_fields(mock_config_class)
        
        # CRITICAL: Should have comprehensive fields (40+ fields), not just 4 basic fields
        assert len(fields) >= 40, f"Expected 40+ comprehensive fields, got {len(fields)}"
        assert len(fields) <= 50, f"Expected at most 50 fields, got {len(fields)}"
        
        # Verify it has the expected field structure from field_definitions.py
        field_names = [f["name"] for f in fields]
        
        # Should have inherited fields
        inherited_fields = ["author", "bucket", "role", "region", "service_name", "pipeline_version", "project_root_folder"]
        for field_name in inherited_fields:
            assert field_name in field_names, f"Missing inherited field: {field_name}"
        
        # Should have essential cradle fields
        essential_fields = ["start_date", "end_date", "data_source_name", "data_source_type", "transform_sql", "job_type", "cradle_account"]
        for field_name in essential_fields:
            assert field_name in field_names, f"Missing essential field: {field_name}"
        
        # Should have system fields
        system_fields = ["output_format", "cluster_type", "job_retry_count", "split_job"]
        for field_name in system_fields:
            assert field_name in field_names, f"Missing system field: {field_name}"
        
        # Should have conditional fields for different data source types
        mds_fields = ["mds_service", "mds_region", "mds_org_id"]
        edx_fields = ["edx_provider", "edx_subject", "edx_dataset"]
        andes_fields = ["andes_provider", "andes_table_name"]
        
        for field_name in mds_fields + edx_fields + andes_fields:
            assert field_name in field_names, f"Missing data source specific field: {field_name}"
    
    def test_cradle_config_field_tiers_are_comprehensive(self):
        """Test that CradleDataLoadingConfig fields have proper 3-tier categorization."""
        # Following Category 7: Data Structure Fidelity pattern
        
        from cursus.api.config_ui.core.core import UniversalConfigCore
        
        mock_config_class = Mock()
        mock_config_class.__name__ = "CradleDataLoadingConfig"
        
        core = UniversalConfigCore()
        fields = core._get_form_fields(mock_config_class)
        
        # Count fields by tier
        tier_counts = {"inherited": 0, "essential": 0, "system": 0}
        for field in fields:
            tier = field.get("tier", "unknown")
            if tier in tier_counts:
                tier_counts[tier] += 1
        
        # Should have fields in all three tiers
        assert tier_counts["inherited"] > 0, "Should have inherited fields (Tier 3)"
        assert tier_counts["essential"] > 0, "Should have essential fields (Tier 1)"
        assert tier_counts["system"] > 0, "Should have system fields (Tier 2)"
        
        # Verify reasonable distribution
        total_fields = sum(tier_counts.values())
        assert tier_counts["inherited"] >= 5, f"Should have at least 5 inherited fields, got {tier_counts['inherited']}"
        assert tier_counts["essential"] >= 10, f"Should have at least 10 essential fields, got {tier_counts['essential']}"
        assert tier_counts["system"] >= 10, f"Should have at least 10 system fields, got {tier_counts['system']}"
    
    def test_cradle_config_widget_creation_uses_field_definitions(self):
        """Test that creating a widget for CradleDataLoadingConfig uses comprehensive field definitions."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # This is the end-to-end test that verifies the user's issue is fixed
        
        from cursus.api.config_ui.core.core import UniversalConfigCore
        from cursus.core.base.config_base import BasePipelineConfig
        
        # Create base config for testing
        base_config = BasePipelineConfig(
            author="test-user",
            bucket="test-bucket", 
            role="arn:aws:iam::123456789012:role/TestRole",
            region="NA",
            service_name="test-service",
            pipeline_version="1.0.0",
            project_root_folder="test-project"
        )
        
        # Mock step catalog to avoid import errors
        with patch('cursus.step_catalog.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            mock_catalog.discover_config_classes.return_value = {}
            
            core = UniversalConfigCore()
            
            # Create a proper mock class that behaves like a real class
            class MockCradleDataLoadingConfig:
                pass
                
            # Set the __name__ attribute correctly
            MockCradleDataLoadingConfig.__name__ = "CradleDataLoadingConfig"
            mock_config_class = MockCradleDataLoadingConfig
            
            with patch.object(core, 'discover_config_classes') as mock_discover:
                mock_discover.return_value = {"CradleDataLoadingConfig": mock_config_class}
                
                # Mock UniversalConfigWidget to avoid widget creation complexity
                with patch('cursus.api.config_ui.widgets.widget.UniversalConfigWidget') as mock_widget_class:
                    mock_widget = Mock()
                    mock_widget_class.return_value = mock_widget
                    
                    # Create widget - this should use comprehensive field definitions
                    widget = core.create_config_widget("CradleDataLoadingConfig", base_config=base_config)
                    
                    assert widget is not None, "Widget creation should succeed"
                    
                    # Verify that UniversalConfigWidget was called with comprehensive field data
                    mock_widget_class.assert_called_once()
                    call_args = mock_widget_class.call_args[0][0]  # First positional argument (form_data)
                    
                    # Check that form_data contains comprehensive fields
                    assert "fields" in call_args, "form_data should contain fields"
                    fields = call_args["fields"]
                    
                    # Should have comprehensive fields (40+ fields), not just 4 basic fields
                    assert len(fields) >= 40, f"Expected 40+ comprehensive fields, got {len(fields)}"
                    assert len(fields) <= 50, f"Expected at most 50 fields, got {len(fields)}"
                    
                    # Verify field structure matches field_definitions.py
                    field_names = [f["name"] for f in fields]
                    
                    # Should have key cradle fields
                    key_fields = ["start_date", "end_date", "data_source_type", "transform_sql", "job_type"]
                    for field_name in key_fields:
                        assert field_name in field_names, f"Missing key field: {field_name}"


class TestFieldDefinitionsErrorHandling:
    """Test error handling and edge cases for field definitions."""
    
    def test_field_definitions_function_stability(self):
        """Test that field definitions function is stable and consistent."""
        # Following Category 16: Exception Handling vs Test Expectations
        
        # Should return consistent results across multiple calls
        fields1 = get_cradle_data_loading_fields()
        fields2 = get_cradle_data_loading_fields()
        
        assert len(fields1) == len(fields2), "Field definitions function returns inconsistent results"
        
        # Compare field names (order might differ but content should be same)
        names1 = {f["name"] for f in fields1}
        names2 = {f["name"] for f in fields2}
        assert names1 == names2, "Field definitions function returns different field names"
    
    def test_field_definitions_no_none_values(self):
        """Test that field definitions don't contain None values in critical fields."""
        # Following Category 16: Exception Handling vs Test Expectations
        fields = get_cradle_data_loading_fields()
        
        for field in fields:
            # Critical fields should never be None
            assert field["name"] is not None, f"Field has None name: {field}"
            assert field["type"] is not None, f"Field {field['name']} has None type"
            assert field["tier"] is not None, f"Field {field['name']} has None tier"
            
            # Optional fields can be None, but if present should not be empty strings
            if "description" in field and field["description"] is not None:
                assert field["description"].strip() != "", f"Field {field['name']} has empty description"
    
    def test_field_definitions_type_safety(self):
        """Test that field definitions have proper type safety."""
        # Following Category 16: Exception Handling vs Test Expectations
        fields = get_cradle_data_loading_fields()
        
        for field in fields:
            # Type checks for field structure
            assert isinstance(field, dict), f"Field is not a dictionary: {field}"
            assert isinstance(field["name"], str), f"Field name is not string: {field['name']}"
            assert isinstance(field["type"], str), f"Field type is not string: {field['type']}"
            assert isinstance(field["tier"], str), f"Field tier is not string: {field['tier']}"
            
            # Optional field type checks
            if "required" in field:
                assert isinstance(field["required"], bool), f"Field {field['name']} required is not boolean"
            if "options" in field:
                assert isinstance(field["options"], list), f"Field {field['name']} options is not list"
            if "default" in field and field["default"] is not None:
                # Default can be various types, just check it's not a complex object
                assert not hasattr(field["default"], "__dict__"), f"Field {field['name']} default is complex object"
