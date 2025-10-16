#!/usr/bin/env python3
"""
Test config instance-based script discovery enhancement in ScriptAutoDiscovery.

This test demonstrates the new functionality that allows ScriptAutoDiscovery to work
with loaded config instances, eliminating phantom script discovery.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cursus.step_catalog.script_discovery import ScriptAutoDiscovery, ScriptInfo


class MockConfigInstance:
    """Mock config instance for testing."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __class__(self):
        return type('MockConfig', (), {})


def test_discover_scripts_from_config_instances():
    """Test the new discover_scripts_from_config_instances method."""
    
    # Create mock package root
    package_root = Path("/mock/cursus")
    
    # Create ScriptAutoDiscovery instance
    discovery = ScriptAutoDiscovery(package_root)
    
    # Create mock loaded configs with different entry points
    loaded_configs = {
        'TabularPreprocessing': MockConfigInstance(
            processing_entry_point='tabular_preprocessing.py',
            label_name='target',
            train_ratio=0.7,
            test_val_ratio=0.5
        ),
        'XGBoostTraining': MockConfigInstance(
            training_entry_point='xgboost_training.py',
            framework_version='1.7-1',
            py_version='3.8',
            job_type='training'
        ),
        'CradleDataLoading': MockConfigInstance(
            # No entry point - this is a data transformation step
            data_source='s3://bucket/data',
            output_format='parquet'
        ),
        'Package': MockConfigInstance(
            processing_entry_point='package.py'
        )
    }
    
    # Mock the script file finding to always return a valid path
    def mock_find_script_file(config_instance, script_name, entry_point_value):
        if hasattr(config_instance, 'processing_entry_point') or hasattr(config_instance, 'training_entry_point'):
            return Path(f"/mock/scripts/{script_name}.py")
        return None
    
    discovery._find_script_file_from_config_instance = mock_find_script_file
    
    # Test the new method
    discovered_scripts = discovery.discover_scripts_from_config_instances(loaded_configs)
    
    # Verify results
    assert len(discovered_scripts) == 3  # Only configs with entry points
    assert 'tabular_preprocessing' in discovered_scripts
    assert 'xgboost_training' in discovered_scripts
    assert 'package' in discovered_scripts
    assert 'cradle_data_loading' not in discovered_scripts  # No entry point - phantom eliminated!
    
    # Check script info details
    tabular_script = discovered_scripts['tabular_preprocessing']
    assert tabular_script.script_name == 'tabular_preprocessing'
    assert tabular_script.step_name == 'TabularPreprocessing'
    assert 'environment_variables' in tabular_script.metadata
    assert 'job_arguments' in tabular_script.metadata
    
    # Check environment variables extraction
    env_vars = tabular_script.metadata['environment_variables']
    assert env_vars['LABEL_FIELD'] == 'target'  # label_name -> LABEL_FIELD
    assert env_vars['TRAIN_RATIO'] == '0.7'
    assert env_vars['TEST_VAL_RATIO'] == '0.5'
    assert env_vars['PYTHONPATH'] == '/opt/ml/code'
    assert env_vars['CURSUS_ENV'] == 'testing'
    
    xgboost_script = discovered_scripts['xgboost_training']
    xgb_env_vars = xgboost_script.metadata['environment_variables']
    assert xgb_env_vars['FRAMEWORK_VERSION'] == '1.7-1'
    assert xgb_env_vars['PYTHON_VERSION'] == '3.8'  # py_version -> PYTHON_VERSION
    
    # Check job arguments extraction
    xgb_job_args = xgboost_script.metadata['job_arguments']
    assert xgb_job_args['job_type'] == 'training'


def test_discover_scripts_from_dag_and_configs():
    """Test the new discover_scripts_from_dag_and_configs method."""
    
    # Create mock DAG
    mock_dag = Mock()
    mock_dag.nodes = ['TabularPreprocessing', 'XGBoostTraining', 'CradleDataLoading', 'Package']
    
    # Create mock package root
    package_root = Path("/mock/cursus")
    
    # Create ScriptAutoDiscovery instance
    discovery = ScriptAutoDiscovery(package_root)
    
    # Create mock loaded configs
    loaded_configs = {
        'TabularPreprocessing': MockConfigInstance(
            processing_entry_point='tabular_preprocessing.py',
            label_name='target'
        ),
        'XGBoostTraining': MockConfigInstance(
            training_entry_point='xgboost_training.py',
            framework_version='1.7-1'
        ),
        'CradleDataLoading': MockConfigInstance(
            # No entry point - data transformation only
            data_source='s3://bucket/data'
        ),
        'Package': MockConfigInstance(
            processing_entry_point='package.py'
        )
    }
    
    # Mock the script file finding
    def mock_find_script_file(config_instance, script_name, entry_point_value):
        if hasattr(config_instance, 'processing_entry_point') or hasattr(config_instance, 'training_entry_point'):
            return Path(f"/mock/scripts/{script_name}.py")
        return None
    
    discovery._find_script_file_from_config_instance = mock_find_script_file
    
    # Test the new method
    discovered_scripts = discovery.discover_scripts_from_dag_and_configs(mock_dag, loaded_configs)
    
    # Verify results - only DAG nodes with actual scripts
    assert len(discovered_scripts) == 3
    assert 'tabular_preprocessing' in discovered_scripts
    assert 'xgboost_training' in discovered_scripts
    assert 'package' in discovered_scripts
    assert 'cradle_data_loading' not in discovered_scripts  # Phantom eliminated!
    
    # Verify step names are set correctly for DAG context
    for script_info in discovered_scripts.values():
        assert script_info.step_name in mock_dag.nodes


def test_environment_variable_extraction_rules():
    """Test the simplified environment variable extraction rules."""
    
    package_root = Path("/mock/cursus")
    discovery = ScriptAutoDiscovery(package_root)
    
    # Test config with various field names
    config_instance = MockConfigInstance(
        label_name='target_variable',  # Should map to LABEL_FIELD
        train_ratio=0.8,               # Should map to TRAIN_RATIO
        test_val_ratio=0.3,            # Should map to TEST_VAL_RATIO
        framework_version='2.0.1',     # Should map to FRAMEWORK_VERSION
        py_version='3.9',              # Should map to PYTHON_VERSION
        processing_framework_version='1.5.0'  # Should map to PROCESSING_FRAMEWORK_VERSION
    )
    
    env_vars = discovery._extract_environment_variables_from_config_instance(config_instance)
    
    # Check rule-based mapping: CAPITAL_CASE env var <- lowercase config field
    assert env_vars['LABEL_FIELD'] == 'target_variable'
    assert env_vars['TRAIN_RATIO'] == '0.8'
    assert env_vars['TEST_VAL_RATIO'] == '0.3'
    assert env_vars['FRAMEWORK_VERSION'] == '2.0.1'
    assert env_vars['PYTHON_VERSION'] == '3.9'
    assert env_vars['PROCESSING_FRAMEWORK_VERSION'] == '1.5.0'
    
    # Check default environment variables
    assert env_vars['PYTHONPATH'] == '/opt/ml/code'
    assert env_vars['CURSUS_ENV'] == 'testing'


def test_config_field_variations():
    """Test handling of config field name variations."""
    
    package_root = Path("/mock/cursus")
    discovery = ScriptAutoDiscovery(package_root)
    
    # Test with field variations
    config_instance = MockConfigInstance(
        label_field='target',  # Alternative to label_name
        python_version='3.10'  # Alternative to py_version
    )
    
    env_vars = discovery._extract_environment_variables_from_config_instance(config_instance)
    
    # Should find variations
    assert env_vars['LABEL_FIELD'] == 'target'
    assert env_vars['PYTHON_VERSION'] == '3.10'


if __name__ == "__main__":
    # Run the tests
    print("Testing config instance-based script discovery...")
    
    test_discover_scripts_from_config_instances()
    print("✅ test_discover_scripts_from_config_instances passed")
    
    test_discover_scripts_from_dag_and_configs()
    print("✅ test_discover_scripts_from_dag_and_configs passed")
    
    test_environment_variable_extraction_rules()
    print("✅ test_environment_variable_extraction_rules passed")
    
    test_config_field_variations()
    print("✅ test_config_field_variations passed")
    
    print("\n🎉 All tests passed! Config instance-based script discovery is working correctly.")
    print("\nKey Benefits Demonstrated:")
    print("  - ✅ Eliminates phantom scripts (CradleDataLoading has no entry point)")
    print("  - ✅ Extracts environment variables using simple CAPITAL_CASE rules")
    print("  - ✅ Handles config field variations (label_name vs label_field)")
    print("  - ✅ Works with loaded config instances (no file parsing needed)")
    print("  - ✅ Integrates with DAG-based filtering for definitive validation")
