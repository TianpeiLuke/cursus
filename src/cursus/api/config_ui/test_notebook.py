#!/usr/bin/env python3
"""
Test script for the Universal Configuration Widget notebook.

This script tests the notebook functionality by running the same code
that would be executed in the Jupyter notebook cells.
"""

import sys
import os
from pathlib import Path

def test_project_root_detection():
    """Test the project root detection logic."""
    print("🧪 Testing project root detection...")
    
    # Find the actual project root (cursus directory)
    current_path = Path().cwd()
    project_root = current_path
    
    print(f"Starting from: {current_path}")
    
    # Navigate up to find the cursus project root
    # Look for the directory that contains 'src' and has 'cursus' in its path
    while project_root.parent != project_root:
        print(f"Checking: {project_root} (name: {project_root.name})")
        # Check if this directory contains 'src' and is the main cursus directory
        if (project_root / 'src').exists() and 'cursus' in str(project_root) and project_root.name == 'cursus':
            print(f"✅ Found main cursus directory: {project_root}")
            break
        # Also check if we're in a subdirectory and need to go up to find the main cursus directory
        if (project_root.parent / 'src').exists() and project_root.parent.name == 'cursus':
            project_root = project_root.parent
            print(f"✅ Found cursus directory by going up: {project_root}")
            break
        project_root = project_root.parent
    
    # Final fallback: if we still haven't found it, use current directory
    if not (project_root / 'src').exists():
        print("❌ Could not find proper cursus directory, using current directory")
        project_root = current_path
    
    project_root_str = str(project_root)
    src_path = str(project_root / 'src')
    
    print(f"📁 Project root: {project_root_str}")
    print(f"📁 Source path: {src_path}")
    
    # Verify the paths exist
    if project_root.exists():
        print("✅ Project root exists")
    else:
        print("❌ Project root does not exist")
    
    if Path(src_path).exists():
        print("✅ Source path exists")
    else:
        print("❌ Source path does not exist")
    
    return project_root_str, src_path

def test_imports(src_path):
    """Test importing the required modules."""
    print("\n🧪 Testing imports...")
    
    # Add src to path if not already there
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"✅ Added {src_path} to Python path")
    
    try:
        # Test core imports
        from cursus.core.base.config_base import BasePipelineConfig
        print("✅ BasePipelineConfig imported successfully")
        
        from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase
        print("✅ ProcessingStepConfigBase imported successfully")
        
        from cursus.core.base.hyperparameters_base import ModelHyperparameters
        print("✅ ModelHyperparameters imported successfully")
        
        from cursus.steps.hyperparams.hyperparameters_xgboost import XGBoostModelHyperparameters
        print("✅ XGBoostModelHyperparameters imported successfully")
        
        # Test widget imports
        from cursus.api.config_ui.jupyter_widget import (
            create_config_widget,
            create_complete_config_ui_widget,
            create_enhanced_save_all_merged_widget
        )
        print("✅ Jupyter widget functions imported successfully")
        
        # Test DAG manager imports
        from cursus.api.config_ui.dag_manager import (
            create_pipeline_config_widget,
            analyze_pipeline_dag
        )
        print("✅ DAG manager functions imported successfully")
        
        return True, {
            'BasePipelineConfig': BasePipelineConfig,
            'ProcessingStepConfigBase': ProcessingStepConfigBase,
            'ModelHyperparameters': ModelHyperparameters,
            'XGBoostModelHyperparameters': XGBoostModelHyperparameters
        }
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False, None

def test_configurations(classes):
    """Test creating configuration objects."""
    print("\n🧪 Testing configuration creation...")
    
    try:
        # Test base configuration
        base_config = classes['BasePipelineConfig'](
            author="lukexie",
            bucket="example-bucket", 
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            region="NA",
            service_name="AtoZ",
            pipeline_version="1.0.0",
            project_root_folder="example-project"
        )
        print("✅ BasePipelineConfig created successfully")
        
        # Test processing configuration
        processing_step_config = classes['ProcessingStepConfigBase'].from_base_config(
            base_config,
            processing_step_name="data_processing",
            instance_type="ml.m5.2xlarge",
            volume_size=500,
            processing_source_dir="src/processing",
            entry_point="main.py"
        )
        print("✅ ProcessingStepConfigBase created successfully")
        
        # Test hyperparameters
        base_hyperparameter = classes['ModelHyperparameters'](
            full_field_list=["PAYMETH", "claim_reason", "claimAmount_value", "COMP_DAYOB"],
            tab_field_list=["PAYMETH", "claim_reason"],
            cat_field_list=["PAYMETH", "claim_reason"],
            label_name="is_abuse",
            id_name="objectId",
            multiclass_categories=[0, 1]  # Binary classification: non-abuse (0) and abuse (1)
        )
        print("✅ ModelHyperparameters created successfully")
        
        # Test XGBoost hyperparameters
        xgb_hyperparams = classes['XGBoostModelHyperparameters'].from_base_hyperparam(
            base_hyperparameter,
            num_round=100,
            max_depth=6,
            min_child_weight=1
        )
        print("✅ XGBoostModelHyperparameters created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_server_status():
    """Test server status checking."""
    print("\n🧪 Testing server status...")
    
    try:
        import requests
        
        SERVER_URL = "http://127.0.0.1:8003"
        
        def check_server_status():
            """Check if the config UI server is running."""
            try:
                # Use the root endpoint instead of /health since /health doesn't exist
                response = requests.get(SERVER_URL, timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    return data.get('message') == 'Cursus Config UI API'
                return False
            except requests.exceptions.RequestException:
                return False
        
        if check_server_status():
            print("✅ Config UI server is running!")
            print(f"🌐 Access at: {SERVER_URL}/config-ui")
            return True
        else:
            print("⚠️ Config UI server is not running")
            print("💡 Start it with: python src/cursus/api/config_ui/start_server.py --port 8003")
            return False
            
    except ImportError:
        print("❌ requests module not available")
        return False
    except Exception as e:
        print(f"❌ Error checking server status: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Universal Configuration Widget Notebook Test")
    print("=" * 50)
    
    # Test 1: Project root detection
    project_root, src_path = test_project_root_detection()
    
    # Test 2: Imports
    imports_success, classes = test_imports(src_path)
    if not imports_success:
        print("\n❌ Import tests failed. Cannot continue with configuration tests.")
        return False
    
    # Test 3: Configuration creation
    config_success = test_configurations(classes)
    if not config_success:
        print("\n❌ Configuration tests failed.")
        return False
    
    # Test 4: Server status
    server_running = test_server_status()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"✅ Project root detection: {'PASS' if project_root else 'FAIL'}")
    print(f"✅ Imports: {'PASS' if imports_success else 'FAIL'}")
    print(f"✅ Configuration creation: {'PASS' if config_success else 'FAIL'}")
    print(f"{'✅' if server_running else '⚠️'} Server status: {'RUNNING' if server_running else 'NOT RUNNING'}")
    
    if imports_success and config_success:
        print("\n🎉 Notebook functionality tests PASSED!")
        print("The notebook should work correctly with all imports and configurations.")
        if not server_running:
            print("💡 Note: Start the server to use the interactive widgets.")
        return True
    else:
        print("\n❌ Some tests FAILED. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
