"""
Test script for the Unified Alignment Tester.

This demonstrates how to use the comprehensive alignment validation system
to validate scripts, contracts, specifications, and builders.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from cursus.validation.alignment import UnifiedAlignmentTester


def create_test_files():
    """Create test files for demonstration."""
    # Create temporary directories
    temp_dir = Path(tempfile.mkdtemp())
    scripts_dir = temp_dir / "scripts"
    contracts_dir = temp_dir / "contracts"
    specs_dir = temp_dir / "specs"
    builders_dir = temp_dir / "builders"
    
    for dir_path in [scripts_dir, contracts_dir, specs_dir, builders_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Create a sample script
    script_content = '''#!/usr/bin/env python3
"""
Sample processing script for testing alignment validation.
"""

import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Sample processing script")
    parser.add_argument("--input-data", required=True, help="Input data path")
    parser.add_argument("--output-path", required=True, help="Output path")
    parser.add_argument("--model-name", default="default", help="Model name")
    
    args = parser.parse_args()
    
    # Access environment variables
    job_name = os.getenv("SM_TRAINING_JOB_NAME", "unknown")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    
    # Read input data
    input_path = "/opt/ml/processing/input/data/train.csv"
    with open(input_path, 'r') as f:
        data = f.read()
    
    # Write output
    output_path = "/opt/ml/processing/output/processed_data.csv"
    with open(output_path, 'w') as f:
        f.write("processed," + data)
    
    print(f"Processing complete for job: {job_name}")

if __name__ == "__main__":
    main()
'''
    
    # Create sample contract
    contract_content = {
        "script_name": "sample_processor",
        "description": "Sample processing script contract",
        "inputs": {
            "train_data": {
                "path": "/opt/ml/processing/input/data/train.csv",
                "data_type": "csv",
                "description": "Training data"
            }
        },
        "outputs": {
            "processed_data": {
                "path": "/opt/ml/processing/output/processed_data.csv",
                "data_type": "csv",
                "description": "Processed training data"
            }
        },
        "arguments": {
            "input_data": {
                "required": True,
                "type": "str",
                "description": "Input data path"
            },
            "output_path": {
                "required": True,
                "type": "str",
                "description": "Output path"
            },
            "model_name": {
                "required": False,
                "type": "str",
                "default": "default",
                "description": "Model name"
            }
        },
        "environment_variables": {
            "required": ["SM_TRAINING_JOB_NAME"],
            "optional": ["AWS_DEFAULT_REGION"]
        }
    }
    
    # Create sample specification
    spec_content = {
        "step_name": "sample_processor",
        "step_type": "processing",
        "description": "Sample processing step specification",
        "dependencies": [
            {
                "logical_name": "train_data",
                "data_type": "csv",
                "source": "data_ingestion"
            }
        ],
        "outputs": [
            {
                "logical_name": "processed_data",
                "data_type": "csv",
                "description": "Processed training data"
            }
        ],
        "configuration": {
            "required": ["instance_type", "instance_count"],
            "optional": ["max_runtime_seconds"],
            "fields": {
                "instance_type": {
                    "type": "str",
                    "default": "ml.m5.large"
                },
                "instance_count": {
                    "type": "int",
                    "default": 1
                },
                "max_runtime_seconds": {
                    "type": "int",
                    "default": 3600
                }
            }
        }
    }
    
    # Create sample builder (simplified)
    builder_content = '''"""
Sample step builder for testing alignment validation.
"""

class SampleProcessorBuilder:
    def __init__(self, config):
        self.config = config
        
    def build_step(self):
        # Access configuration fields
        instance_type = self.config.instance_type
        instance_count = self.config.instance_count
        max_runtime = getattr(self.config, 'max_runtime_seconds', 3600)
        
        # Validate required fields
        if not instance_type:
            raise ValueError("instance_type is required")
            
        return {
            "instance_type": instance_type,
            "instance_count": instance_count,
            "max_runtime_seconds": max_runtime
        }
'''
    
    # Write files
    (scripts_dir / "sample_processor.py").write_text(script_content)
    (contracts_dir / "sample_processor_contract.json").write_text(json.dumps(contract_content, indent=2))
    (specs_dir / "sample_processor_spec.json").write_text(json.dumps(spec_content, indent=2))
    (builders_dir / "sample_processor_builder.py").write_text(builder_content)
    
    return temp_dir, scripts_dir, contracts_dir, specs_dir, builders_dir


def test_unified_alignment_tester():
    """Test the unified alignment tester with sample files."""
    print("🔍 Testing Unified Alignment Tester")
    print("=" * 50)
    
    # Create test files
    temp_dir, scripts_dir, contracts_dir, specs_dir, builders_dir = create_test_files()
    
    try:
        # Initialize the unified tester
        tester = UnifiedAlignmentTester(
            scripts_dir=str(scripts_dir),
            contracts_dir=str(contracts_dir),
            specs_dir=str(specs_dir),
            builders_dir=str(builders_dir)
        )
        
        print(f"📁 Test files created in: {temp_dir}")
        print(f"📝 Scripts: {scripts_dir}")
        print(f"📋 Contracts: {contracts_dir}")
        print(f"📊 Specifications: {specs_dir}")
        print(f"⚙️  Builders: {builders_dir}")
        print()
        
        # Discover available scripts
        scripts = tester.discover_scripts()
        print(f"🔍 Discovered scripts: {scripts}")
        print()
        
        # Run validation for a specific script
        if scripts:
            script_name = scripts[0]
            print(f"🧪 Testing specific script: {script_name}")
            result = tester.validate_specific_script(script_name)
            
            print(f"📊 Validation Result:")
            print(f"   Overall Status: {result['overall_status']}")
            print(f"   Level 1 (Script ↔ Contract): {result['level1'].get('passed', 'N/A')}")
            print(f"   Level 2 (Contract ↔ Spec): {result['level2'].get('passed', 'N/A')}")
            print(f"   Level 3 (Spec ↔ Dependencies): {result['level3'].get('passed', 'N/A')}")
            print(f"   Level 4 (Builder ↔ Config): {result['level4'].get('passed', 'N/A')}")
            print()
        
        # Run full validation
        print("🚀 Running full alignment validation...")
        report = tester.run_full_validation(target_scripts=["sample_processor"])
        
        # Print summary
        print("\n📊 Validation Summary:")
        tester.print_summary()
        
        # Get validation summary
        summary = tester.get_validation_summary()
        print(f"\n📈 Summary Statistics:")
        print(f"   Overall Status: {summary['overall_status']}")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Pass Rate: {summary['pass_rate']:.1f}%")
        print(f"   Critical Issues: {summary['critical_issues']}")
        print(f"   Error Issues: {summary['error_issues']}")
        print(f"   Warning Issues: {summary['warning_issues']}")
        
        # Get alignment status matrix
        matrix = tester.get_alignment_status_matrix()
        print(f"\n🔍 Alignment Status Matrix:")
        for script, statuses in matrix.items():
            print(f"   {script}:")
            for level, status in statuses.items():
                print(f"     {level}: {status}")
        
        # Get critical issues
        critical_issues = tester.get_critical_issues()
        if critical_issues:
            print(f"\n🚨 Critical Issues ({len(critical_issues)}):")
            for issue in critical_issues[:3]:  # Show first 3
                print(f"   - {issue['message']}")
                if issue.get('recommendation'):
                    print(f"     💡 {issue['recommendation']}")
        
        print(f"\n✅ Test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up test files")
        except:
            pass


def test_individual_levels():
    """Test individual alignment levels."""
    print("\n🔍 Testing Individual Alignment Levels")
    print("=" * 50)
    
    # Create test files
    temp_dir, scripts_dir, contracts_dir, specs_dir, builders_dir = create_test_files()
    
    try:
        tester = UnifiedAlignmentTester(
            scripts_dir=str(scripts_dir),
            contracts_dir=str(contracts_dir),
            specs_dir=str(specs_dir),
            builders_dir=str(builders_dir)
        )
        
        # Test each level individually
        for level in range(1, 5):
            print(f"\n📊 Testing Level {level}...")
            try:
                report = tester.run_level_validation(level, target_scripts=["sample_processor"])
                summary = tester.get_validation_summary()
                print(f"   Level {level} Status: {summary['overall_status']}")
                print(f"   Tests Run: {summary['total_tests']}")
                print(f"   Issues Found: {summary['critical_issues'] + summary['error_issues'] + summary['warning_issues']}")
            except Exception as e:
                print(f"   Level {level} Error: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual level test failed: {str(e)}")
        return False
    
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


if __name__ == "__main__":
    print("🧪 Unified Alignment Tester - Test Suite")
    print("=" * 60)
    
    # Run tests
    success1 = test_unified_alignment_tester()
    success2 = test_individual_levels()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
