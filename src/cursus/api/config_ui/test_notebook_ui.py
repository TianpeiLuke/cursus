#!/usr/bin/env python3
"""
Test script for the Universal Configuration Widget notebook UI experience.

This script tests the key functionality demonstrated in the example notebook
without requiring the full server setup or complex imports.
"""

import sys
import json
from pathlib import Path

def test_basic_imports():
    """Test that basic imports work correctly."""
    print("🧪 Testing Basic Imports...")
    
    try:
        # Test core imports
        from .core import UniversalConfigCore
        from .utils import discover_available_configs
        from .dag_manager import DAGConfigurationManager, analyze_pipeline_dag
        print("✅ Core imports successful")
        
        # Test widget imports
        from .jupyter_widget import (
            create_config_widget,
            create_complete_config_ui_widget,
            create_enhanced_save_all_merged_widget
        )
        print("✅ Widget imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config_discovery():
    """Test configuration discovery functionality."""
    print("\n🧪 Testing Configuration Discovery...")
    
    try:
        from .utils import discover_available_configs
        
        # Discover available configurations
        configs = discover_available_configs()
        
        print(f"✅ Discovered {len(configs)} configuration classes:")
        for name in sorted(configs.keys())[:10]:  # Show first 10
            print(f"  • {name}")
        
        if len(configs) > 10:
            print(f"  ... and {len(configs) - 10} more")
        
        return len(configs) > 0
        
    except Exception as e:
        print(f"❌ Configuration discovery failed: {e}")
        return False


def test_dag_analysis():
    """Test DAG analysis functionality."""
    print("\n🧪 Testing DAG Analysis...")
    
    try:
        from .dag_manager import analyze_pipeline_dag
        
        # Create mock DAG for testing
        mock_dag = {
            "nodes": [
                "cradle_data_loading",
                "tabular_preprocessing_training", 
                "xgboost_training",
                "xgboost_model_creation",
                "model_registration"
            ]
        }
        
        class MockDAG:
            def __init__(self, nodes):
                self.nodes = nodes
        
        pipeline_dag = MockDAG(mock_dag["nodes"])
        
        # Analyze the DAG
        analysis = analyze_pipeline_dag(pipeline_dag)
        
        print("✅ DAG Analysis Results:")
        print(f"  📊 Discovered Steps: {len(analysis['discovered_steps'])}")
        print(f"  ⚙️ Required Configs: {len(analysis['required_configs'])}")
        print(f"  📋 Workflow Steps: {len(analysis['workflow_steps'])}")
        print(f"  ❌ Hidden Configs: {analysis['hidden_configs_count']}")
        
        print("\n🎯 Required Configuration Classes:")
        for config in analysis['required_configs'][:5]:  # Show first 5
            specialized = "(Specialized)" if config.get('is_specialized', False) else ""
            print(f"  ✅ {config['config_class_name']} {specialized}")
        
        return len(analysis['required_configs']) > 0
        
    except Exception as e:
        print(f"❌ DAG analysis failed: {e}")
        return False


def test_widget_creation():
    """Test widget creation functionality."""
    print("\n🧪 Testing Widget Creation...")
    
    try:
        from .jupyter_widget import create_config_widget
        from cursus.core.base.config_base import BasePipelineConfig
        
        # Create base config for testing
        base_config = BasePipelineConfig(
            author="test-user",
            bucket="test-bucket",
            role="test-role",
            region="NA",
            service_name="test-service",
            pipeline_version="1.0.0",
            project_root_folder="test-project"
        )
        
        # Create a widget
        widget = create_config_widget(
            config_class_name="ProcessingStepConfigBase",
            base_config=base_config
        )
        
        print("✅ Widget created successfully")
        print(f"  📋 Widget ID: {widget.widget_id[:8]}...")
        print(f"  🎯 Config Class: {widget.config_class_name}")
        print(f"  🔗 Server URL: {widget.server_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ Widget creation failed: {e}")
        return False


def test_enhanced_save_widget():
    """Test enhanced save all merged widget."""
    print("\n🧪 Testing Enhanced Save All Merged Widget...")
    
    try:
        from .jupyter_widget import create_enhanced_save_all_merged_widget
        
        # Create sample session configs
        session_configs = {
            "BasePipelineConfig": {
                "author": "test-user",
                "bucket": "test-bucket",
                "region": "NA",
                "service_name": "AtoZ"
            },
            "ProcessingStepConfigBase": {
                "processing_step_name": "test_processing",
                "instance_type": "ml.m5.2xlarge"
            }
        }
        
        # Create enhanced save widget
        save_widget = create_enhanced_save_all_merged_widget(session_configs)
        
        print("✅ Enhanced Save All Merged widget created successfully")
        print(f"  📋 Widget ID: {save_widget.widget_id[:8]}...")
        print(f"  📊 Session Configs: {len(save_widget.session_configs)}")
        print(f"  🔗 Server URL: {save_widget.server_url}")
        
        # Test smart filename generation
        smart_filename = save_widget._generate_smart_filename()
        print(f"  📄 Smart Filename: {smart_filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced save widget creation failed: {e}")
        return False


def test_api_functionality():
    """Test API functionality without starting server."""
    print("\n🧪 Testing API Functionality...")
    
    try:
        from .api import create_config_ui_app
        
        # Create FastAPI app
        app = create_config_ui_app()
        
        print("✅ FastAPI app created successfully")
        print(f"  📋 App Title: {app.title}")
        print(f"  🔢 Version: {app.version}")
        print(f"  📚 Routes: {len(app.routes)} endpoints")
        
        # List some key routes
        config_ui_routes = [route for route in app.routes if hasattr(route, 'path') and 'config-ui' in route.path]
        print(f"  🎯 Config UI Routes: {len(config_ui_routes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ API functionality test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide summary."""
    print("🎯 Universal Configuration Widget - Notebook UI Experience Test")
    print("=" * 70)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration Discovery", test_config_discovery),
        ("DAG Analysis", test_dag_analysis),
        ("Widget Creation", test_widget_creation),
        ("Enhanced Save Widget", test_enhanced_save_widget),
        ("API Functionality", test_api_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 Test Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The notebook UI experience is ready.")
        print("\n💡 Next Steps:")
        print("  1. Open example_universal_config_widget.ipynb in Jupyter")
        print("  2. Run the cells to test the interactive widgets")
        print("  3. Try the DAG-driven pipeline configuration")
        print("  4. Test the Save All Merged functionality")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
