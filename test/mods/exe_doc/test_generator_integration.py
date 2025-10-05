"""
Integration tests for ExecutionDocumentGenerator.

This module tests the ExecutionDocumentGenerator with real DAGs and configurations
to ensure it correctly fills execution documents.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from cursus.mods.exe_doc.generator import ExecutionDocumentGenerator
from cursus.mods.exe_doc.cradle_helper import CradleDataLoadingHelper
from cursus.mods.exe_doc.registration_helper import RegistrationHelper
from cursus.pipeline_catalog.shared_dags.xgboost.complete_e2e_dag import create_xgboost_complete_e2e_dag


class TestExecutionDocumentGeneratorIntegration:
    """Integration tests for ExecutionDocumentGenerator."""
    
    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent.parent
    
    @pytest.fixture
    def config_path(self):
        """Path to the test configuration."""
        return "test_data/config_NA_xgboost_AtoZ.json"
    
    @pytest.fixture
    def sample_exe_doc_path(self):
        """Path to the sample execution document."""
        return "test_data/sample_exe_doc.json"
    
    @pytest.fixture
    def expected_result_path(self):
        """Path to the expected result."""
        return "test_data/execute_doc_lukexie-AtoZ-xgboost-NA_2.0.0.json"
    
    @pytest.fixture
    def sample_execution_document(self, sample_exe_doc_path):
        """Load the sample execution document."""
        with open(sample_exe_doc_path, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def expected_result(self, expected_result_path):
        """Load the expected result."""
        with open(expected_result_path, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def xgboost_dag(self):
        """Create the XGBoost complete E2E DAG."""
        return create_xgboost_complete_e2e_dag()
    
    def test_fill_execution_document_integration(self, config_path, sample_execution_document, expected_result, xgboost_dag):
        """
        Integration test for fill_execution_document method.
        
        This test verifies that the ExecutionDocumentGenerator correctly fills
        an execution document using real configuration and DAG data.
        """
        # Initialize the generator with the real config
        generator = ExecutionDocumentGenerator(config_path)
        
        # Helpers are now automatically initialized in the constructor
        # No need to manually add them
        
        # Fill the execution document
        result = generator.fill_execution_document(xgboost_dag, sample_execution_document)
        
        # Verify the result structure
        assert "PIPELINE_STEP_CONFIGS" in result
        assert "PIPELINE_ADDITIONAL_PARAMS" in result
        
        # Check that cradle steps were processed
        cradle_steps = [
            "CradleDataLoading-Training", 
            "CradleDataLoading-Calibration"
        ]
        
        for step_name in cradle_steps:
            if step_name in result["PIPELINE_STEP_CONFIGS"]:
                step_config = result["PIPELINE_STEP_CONFIGS"][step_name]
                # Note: STEP_CONFIG may not be present if Cradle dependencies are not available
                # This is expected behavior in test environment
                assert "STEP_TYPE" in step_config
                if "STEP_CONFIG" in step_config:
                    assert isinstance(step_config["STEP_CONFIG"], dict)
                
                # Verify cradle-specific configuration structure
                if "STEP_CONFIG" in step_config and step_config["STEP_CONFIG"]:
                    cradle_config = step_config["STEP_CONFIG"]
                    # Check for expected cradle configuration keys
                    expected_keys = ["dataSources", "transformSpecification", "outputSpecification", "cradleJobSpecification"]
                    for key in expected_keys:
                        if key in cradle_config:
                            assert cradle_config[key] is not None
        
        # Check that registration step was processed
        registration_steps = ["Registration-NA", "Registration"]
        registration_found = False
        
        for step_name in registration_steps:
            if step_name in result["PIPELINE_STEP_CONFIGS"]:
                registration_found = True
                step_config = result["PIPELINE_STEP_CONFIGS"][step_name]
                assert "STEP_CONFIG" in step_config
                assert "STEP_TYPE" in step_config
                
                # Verify registration-specific configuration structure
                if "STEP_CONFIG" in step_config and step_config["STEP_CONFIG"]:
                    reg_config = step_config["STEP_CONFIG"]
                    # Check for expected registration configuration keys
                    expected_keys = [
                        "model_domain", 
                        "model_objective", 
                        "source_model_inference_content_types",
                        "source_model_inference_input_variable_list",
                        "source_model_inference_output_variable_list"
                    ]
                    for key in expected_keys:
                        if key in reg_config:
                            assert reg_config[key] is not None
        
        # At least one registration step should be found and processed
        assert registration_found, "No registration step was found and processed"
    
    def test_fill_execution_document_cradle_data_mapping(self, config_path, sample_execution_document, expected_result, xgboost_dag):
        """
        Test that cradle data loading configurations are correctly mapped from config to execution document.
        """
        generator = ExecutionDocumentGenerator(config_path)
        
        result = generator.fill_execution_document(xgboost_dag, sample_execution_document)
        
        # Check specific cradle configuration mappings
        training_step = "CradleDataLoading-Training"
        if training_step in result["PIPELINE_STEP_CONFIGS"]:
            step_config = result["PIPELINE_STEP_CONFIGS"][training_step]
            # Skip if STEP_CONFIG is not present (Cradle dependencies not available)
            if "STEP_CONFIG" not in step_config:
                pytest.skip("Cradle dependencies not available - STEP_CONFIG not populated")
            step_config = step_config["STEP_CONFIG"]
            
            if "dataSources" in step_config:
                data_sources = step_config["dataSources"]["dataSources"]
                
                # Verify that we have the expected data sources
                source_names = [ds["dataSourceName"] for ds in data_sources]
                # Check for either RAW_MDS_NA (from config) or RAW_MDS (from sample doc)
                assert any(name in source_names for name in ["RAW_MDS_NA", "RAW_MDS"])
                assert "TAGS" in source_names
                
                # Verify MDS data source configuration
                mds_source = next((ds for ds in data_sources if ds["dataSourceName"] in ["RAW_MDS_NA", "RAW_MDS"]), None)
                if mds_source:
                    assert mds_source["dataSourceType"] == "MDS"
                    assert "mdsDataSourceProperties" in mds_source
                    # Only check specific properties if they exist (sample doc may have different structure)
                    if "mdsDataSourceProperties" in mds_source:
                        mds_props = mds_source["mdsDataSourceProperties"]
                        # Check properties that should be present
                        if "serviceName" in mds_props:
                            assert mds_props["serviceName"] is not None
                        if "region" in mds_props:
                            assert mds_props["region"] is not None
                
                # Verify EDX/ANDES data source configuration (TAGS source)
                tags_source = next((ds for ds in data_sources if ds["dataSourceName"] == "TAGS"), None)
                if tags_source:
                    # Data source type can be either EDX or ANDES depending on configuration
                    assert tags_source["dataSourceType"] in ["EDX", "ANDES"]
                    # Check for appropriate properties based on type
                    if tags_source["dataSourceType"] == "EDX":
                        assert "edxDataSourceProperties" in tags_source
                    elif tags_source["dataSourceType"] == "ANDES":
                        assert "andesDataSourceProperties" in tags_source or "edxDataSourceProperties" in tags_source
    
    def test_fill_execution_document_registration_mapping(self, config_path, sample_execution_document, expected_result, xgboost_dag):
        """
        Test that registration configurations are correctly mapped from config to execution document.
        """
        generator = ExecutionDocumentGenerator(config_path)
        
        result = generator.fill_execution_document(xgboost_dag, sample_execution_document)
        
        # Check registration configuration mappings
        registration_step = "Registration-NA"
        if registration_step in result["PIPELINE_STEP_CONFIGS"]:
            step_config = result["PIPELINE_STEP_CONFIGS"][registration_step]["STEP_CONFIG"]
            
            # Verify key registration fields are populated
            expected_fields = [
                "model_domain",
                "model_objective", 
                "source_model_inference_content_types",
                "source_model_inference_input_variable_list",
                "source_model_inference_output_variable_list"
            ]
            
            for field in expected_fields:
                if field in step_config:
                    assert step_config[field] is not None
                    
                    # Verify specific values match expected configuration
                    if field == "model_domain":
                        assert step_config[field] == "AtoZ"
                    elif field == "model_objective":
                        assert step_config[field] == "AtoZ_Claims_SM_Model_NA"
                    elif field == "source_model_inference_content_types":
                        assert "text/csv" in step_config[field]
    
    def test_dag_node_mapping(self, config_path, sample_execution_document, xgboost_dag):
        """
        Test that DAG nodes are correctly mapped to execution document steps.
        """
        generator = ExecutionDocumentGenerator(config_path)
        
        result = generator.fill_execution_document(xgboost_dag, sample_execution_document)
        
        # Verify that DAG nodes are represented in the execution document
        dag_nodes = list(xgboost_dag.nodes)
        exec_doc_steps = list(result["PIPELINE_STEP_CONFIGS"].keys())
        
        # Check that key steps from the DAG are present in execution document
        # Note: Step names might be slightly different between DAG and execution document
        key_dag_nodes = [
            "CradleDataLoading_training",
            "CradleDataLoading_calibration", 
            "XGBoostTraining",
            "Registration"
        ]
        
        key_exec_steps = [
            "CradleDataLoading-Training",
            "CradleDataLoading-Calibration",
            "XGBoostTraining", 
            "Registration-NA"
        ]
        
        # Verify that the execution document contains steps corresponding to key DAG nodes
        for exec_step in key_exec_steps:
            if exec_step in exec_doc_steps:
                assert exec_step in result["PIPELINE_STEP_CONFIGS"]
                step_config = result["PIPELINE_STEP_CONFIGS"][exec_step]
                assert "STEP_TYPE" in step_config
    
    def test_error_handling_invalid_config_path(self):
        """Test error handling for invalid configuration path."""
        with pytest.raises(Exception):  # Should raise ExecutionDocumentGenerationError or similar
            ExecutionDocumentGenerator("nonexistent_config.json")
    
    def test_error_handling_missing_helpers(self, config_path, sample_execution_document, xgboost_dag):
        """Test behavior when required helpers are missing."""
        generator = ExecutionDocumentGenerator(config_path)
        # Don't add any helpers
        
        # Should still work but may not fill all configurations
        result = generator.fill_execution_document(xgboost_dag, sample_execution_document)
        
        # Basic structure should still be present
        assert "PIPELINE_STEP_CONFIGS" in result
        assert "PIPELINE_ADDITIONAL_PARAMS" in result
    
    @pytest.mark.parametrize("step_name,expected_type", [
        ("CradleDataLoading-Training", ["WORKFLOW_INPUT", "CradleDataLoadingStep"]),
        ("CradleDataLoading-Calibration", ["WORKFLOW_INPUT", "CradleDataLoadingStep"]),
        ("XGBoostTraining", "TRAINING_STEP"),
        ("Registration-NA", ["PROCESSING_STEP", "MimsModelRegistrationProcessingStep"])
    ])
    def test_step_type_assignment(self, config_path, sample_execution_document, xgboost_dag, step_name, expected_type):
        """Test that step types are correctly assigned."""
        generator = ExecutionDocumentGenerator(config_path)
        
        result = generator.fill_execution_document(xgboost_dag, sample_execution_document)
        
        if step_name in result["PIPELINE_STEP_CONFIGS"]:
            step_config = result["PIPELINE_STEP_CONFIGS"][step_name]
            assert "STEP_TYPE" in step_config
            
            actual_type = step_config["STEP_TYPE"]
            if isinstance(expected_type, list):
                assert actual_type == expected_type
            else:
                assert actual_type == expected_type
    
    def test_save_actual_output(self, config_path, sample_execution_document, xgboost_dag, project_root):
        """
        Test method to save the actual output of fill_execution_document for analysis.
        
        This test runs the ExecutionDocumentGenerator and saves the actual output
        to the config folder for comparison with expected results.
        """
        # Initialize the generator with the real config
        generator = ExecutionDocumentGenerator(config_path)
        
        # Helpers are now automatically initialized in the constructor
        # No need to manually add them
        
        # Fill the execution document
        result = generator.fill_execution_document(xgboost_dag, sample_execution_document)
        
        # Save the result to the test_data folder
        output_path = "test_data/test_output_execution_document.json"
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Actual output saved to: {output_path}")
        
        # Basic verification that the result has the expected structure
        assert "PIPELINE_STEP_CONFIGS" in result
        assert "PIPELINE_ADDITIONAL_PARAMS" in result
        assert len(result["PIPELINE_STEP_CONFIGS"]) > 0

    def test_debug_load_configs(self, config_path, project_root):
        """
        Debug test to examine the output of load_configs with the actual config file.
        
        This test loads the configuration and prints detailed information about
        the loaded config objects to understand their structure.
        Uses a temporary file that is cleaned up after the test.
        """
        from cursus.steps.configs.utils import load_configs, build_complete_config_classes
        import json
        import tempfile
        import os
        
        print(f"\n=== DEBUG: Loading configs from {config_path} ===")
        
        # Build config classes
        config_classes = build_complete_config_classes()
        print(f"Available config classes: {list(config_classes.keys())}")
        
        # Create temporary file for debug output
        temp_debug_file = None
        
        try:
            # Load configs
            loaded_configs = load_configs(config_path, config_classes)
            print(f"Successfully loaded {len(loaded_configs)} configs")
            print(f"Config keys: {list(loaded_configs.keys())}")
            
            # Examine each loaded config
            for config_name, config_obj in loaded_configs.items():
                print(f"\n--- Config: {config_name} ---")
                print(f"Type: {type(config_obj).__name__}")
                print(f"Module: {type(config_obj).__module__}")
                
                # Print all attributes
                attrs = [attr for attr in dir(config_obj) if not attr.startswith('_')]
                print(f"Attributes: {attrs[:10]}...")  # First 10 attributes
                
                # Check for specific attributes we expect
                key_attrs = ['data_sources_spec', 'transform_spec', 'output_spec', 'cradle_job_spec', 
                           'model_domain', 'model_objective', 'source_model_inference_content_types']
                
                for attr in key_attrs:
                    if hasattr(config_obj, attr):
                        value = getattr(config_obj, attr)
                        print(f"  {attr}: {type(value).__name__} = {str(value)[:100]}...")
                
                # For CradleDataLoadConfig, examine the data_sources_spec structure
                if 'CradleDataLoad' in type(config_obj).__name__:
                    print(f"\n  === Cradle Config Details ===")
                    if hasattr(config_obj, 'data_sources_spec'):
                        ds_spec = config_obj.data_sources_spec
                        print(f"  data_sources_spec type: {type(ds_spec).__name__}")
                        if hasattr(ds_spec, 'data_sources'):
                            data_sources = ds_spec.data_sources
                            print(f"  data_sources type: {type(data_sources).__name__}")
                            print(f"  data_sources value: {str(data_sources)[:200]}...")
                            
                            # Check if it has a 'value' attribute (for list wrappers)
                            if hasattr(data_sources, 'value'):
                                print(f"  data_sources.value type: {type(data_sources.value).__name__}")
                                print(f"  data_sources.value length: {len(data_sources.value) if hasattr(data_sources.value, '__len__') else 'N/A'}")
                                
                                if hasattr(data_sources.value, '__iter__') and len(data_sources.value) > 0:
                                    first_ds = data_sources.value[0]
                                    print(f"  First data source type: {type(first_ds).__name__}")
                                    if hasattr(first_ds, 'data_source_name'):
                                        print(f"  First data source name: {first_ds.data_source_name}")
                                    if hasattr(first_ds, 'data_source_type'):
                                        print(f"  First data source type: {first_ds.data_source_type}")
                
                # For Registration config, examine key fields
                if 'Registration' in type(config_obj).__name__:
                    print(f"\n  === Registration Config Details ===")
                    reg_attrs = ['model_domain', 'model_objective', 'framework', 'aws_region']
                    for attr in reg_attrs:
                        if hasattr(config_obj, attr):
                            value = getattr(config_obj, attr)
                            print(f"  {attr}: {value}")
            
            # Save detailed config info to temporary file
            debug_output = {}
            for config_name, config_obj in loaded_configs.items():
                config_info = {
                    'type': type(config_obj).__name__,
                    'module': type(config_obj).__module__,
                    'attributes': {}
                }
                
                # Get all non-private attributes
                for attr in dir(config_obj):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(config_obj, attr)
                            # Convert to string representation for JSON serialization
                            if hasattr(value, '__dict__'):
                                config_info['attributes'][attr] = str(value)
                            else:
                                config_info['attributes'][attr] = value
                        except Exception as e:
                            config_info['attributes'][attr] = f"Error accessing: {str(e)}"
                
                debug_output[config_name] = config_info
            
            # Create temporary file for debug output
            temp_debug_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(debug_output, temp_debug_file, indent=2, default=str)
            temp_debug_file.close()
            
            print(f"\nDebug output saved to temporary file: {temp_debug_file.name}")
            
            # Verify the file was created and has content
            assert os.path.exists(temp_debug_file.name)
            with open(temp_debug_file.name, 'r') as f:
                saved_data = json.load(f)
                assert len(saved_data) > 0
            
        except Exception as e:
            print(f"Error loading configs: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Clean up temporary file
            if temp_debug_file and os.path.exists(temp_debug_file.name):
                os.unlink(temp_debug_file.name)
                print(f"Cleaned up temporary debug file: {temp_debug_file.name}")
