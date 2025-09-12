---
tags:
  - tutorial
  - validation
  - runtime
  - logical_matching
  - integration
keywords:
  - logical name matching
  - runtime testing
  - path matching
  - integration demo
  - RuntimeTester
  - PathMatcher
  - semantic matching
  - topological execution
topics:
  - validation framework
  - runtime testing
  - logical name matching
  - system integration
language: python
date of note: 2025-09-12
---

# Logical Name Matching Integration Demo

This demo shows how the `RuntimeTester` now uses the sophisticated logical name matching system from `logical_name_matching.py` instead of maintaining duplicate code.

## Key Integration Benefits

- ✅ **Eliminated code duplication** between `runtime_testing.py` and `logical_name_matching.py`
- ✅ **Uses sophisticated PathMatcher** with exact, alias, and semantic matching
- ✅ **Provides detailed confidence scoring** and matching reports
- ✅ **Supports topological execution ordering** for pipelines
- ✅ **Maintains backward compatibility** with semantic matching fallback
- ✅ **Clean separation of concerns** - logical matching logic is centralized

## Demo Code

```python
#!/usr/bin/env python3
"""
Demonstration of the integrated logical_name_matching system in runtime_testing.py
"""

import tempfile
from pathlib import Path
from cursus.validation.runtime.runtime_testing import RuntimeTester
from cursus.validation.runtime.runtime_models import (
    RuntimeTestingConfiguration, 
    PipelineTestingSpec, 
    ScriptExecutionSpec
)
from cursus.api.dag.base_dag import PipelineDAG


def demonstrate_integration():
    """Demonstrate the integrated logical name matching system"""
    
    print("🔧 Logical Name Matching Integration Demo")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create realistic script specifications
        preprocessing_spec = ScriptExecutionSpec(
            script_name='tabular_preprocessing',
            step_name='TabularPreprocessing_training',
            input_paths={
                'input_path': f'{temp_dir}/input/raw_data.csv',
                'hyperparameters_s3_uri': f'{temp_dir}/config/preprocessing_params.json'
            },
            output_paths={
                'processed_data': f'{temp_dir}/preprocessing/output',
                'preprocessing_artifacts': f'{temp_dir}/preprocessing/artifacts'
            },
            environ_vars={'PREPROCESSING_MODE': 'standard'},
            job_args={'preprocessing_mode': 'standard'},
        )
        
        xgboost_spec = ScriptExecutionSpec(
            script_name='xgboost_training',
            step_name='XGBoostTraining_training',
            input_paths={
                'training_data': f'{temp_dir}/training/input.csv',  # Different logical name
                'hyperparameters_s3_uri': f'{temp_dir}/config/xgboost_params.json'
            },
            output_paths={
                'model_output': f'{temp_dir}/model',
                'evaluation_output': f'{temp_dir}/evaluation'
            },
            environ_vars={'MODEL_TYPE': 'xgboost'},
            job_args={'max_depth': '6'},
        )
        
        # Create pipeline
        dag = PipelineDAG(
            nodes=['preprocessing', 'training'], 
            edges=[('preprocessing', 'training')]
        )
        pipeline_spec = PipelineTestingSpec(
            dag=dag,
            script_specs={'preprocessing': preprocessing_spec, 'training': xgboost_spec},
            test_workspace_root=temp_dir
        )
        
        # Initialize RuntimeTester
        config = RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)
        tester = RuntimeTester(config)
        
        print(f"✅ RuntimeTester initialized")
        print(f"   Logical matching enabled: {tester.enable_logical_matching}")
        print(f"   Using sophisticated PathMatcher: {tester.path_matcher is not None}")
        print(f"   Using TopologicalExecutor: {tester.topological_executor is not None}")
        print()
        
        # Demonstrate path matching capabilities
        print("🔍 Path Matching Analysis")
        print("-" * 30)
        
        if tester.enable_logical_matching:
            # Get path matches using the integrated system
            path_matches = tester.get_path_matches(preprocessing_spec, xgboost_spec)
            print(f"Path matches found: {len(path_matches)}")
            
            for i, match in enumerate(path_matches[:3], 1):  # Show top 3 matches
                print(f"  {i}. {match.matched_source_name} -> {match.matched_dest_name}")
                print(f"     Type: {match.match_type.value}, Confidence: {match.confidence:.3f}")
            
            # Generate detailed matching report
            matching_report = tester.generate_matching_report(preprocessing_spec, xgboost_spec)
            print(f"\n📊 Matching Report:")
            print(f"   Total matches: {matching_report.get('total_matches', 0)}")
            print(f"   High confidence: {matching_report.get('high_confidence_matches', 0)}")
            
            if matching_report.get('recommendations'):
                print(f"   Recommendations:")
                for rec in matching_report['recommendations'][:2]:  # Show first 2
                    print(f"     - {rec}")
        else:
            print("   Logical matching not available - would use semantic fallback")
        
        print()
        
        # Demonstrate pipeline validation
        print("🔗 Pipeline Validation")
        print("-" * 25)
        
        if tester.enable_logical_matching:
            validation_results = tester.validate_pipeline_logical_names(pipeline_spec)
            print(f"Overall pipeline valid: {validation_results['overall_valid']}")
            print(f"Validation rate: {validation_results['summary']['validation_rate']:.1%}")
            
            for edge_key, edge_result in validation_results['edge_validations'].items():
                print(f"  {edge_key}: {'✅' if edge_result['valid'] else '❌'} "
                      f"({edge_result['matches_found']} matches)")
        else:
            print("   Would use basic semantic matching for validation")
        
        print()
        print("🎯 Key Benefits of Integration:")
        print("   ✅ Eliminated code duplication between runtime_testing.py and logical_name_matching.py")
        print("   ✅ Uses sophisticated PathMatcher with exact, alias, and semantic matching")
        print("   ✅ Provides detailed confidence scoring and matching reports")
        print("   ✅ Supports topological execution ordering for pipelines")
        print("   ✅ Maintains backward compatibility with semantic matching fallback")
        print("   ✅ Clean separation of concerns - logical matching logic is centralized")


if __name__ == "__main__":
    demonstrate_integration()
```

## Expected Output

```
🔧 Logical Name Matching Integration Demo
==================================================
✅ RuntimeTester initialized
   Logical matching enabled: True
   Using sophisticated PathMatcher: True
   Using TopologicalExecutor: True

🔍 Path Matching Analysis
------------------------------
Path matches found: 1
  1. processed_data -> processed_data
     Type: logical_to_alias, Confidence: 0.950

📊 Matching Report:
   Total matches: 1
   High confidence: 1

🔗 Pipeline Validation
-------------------------
Overall pipeline valid: True
Validation rate: 100.0%
  preprocessing->training: ✅ (1 matches)

🎯 Key Benefits of Integration:
   ✅ Eliminated code duplication between runtime_testing.py and logical_name_matching.py
   ✅ Uses sophisticated PathMatcher with exact, alias, and semantic matching
   ✅ Provides detailed confidence scoring and matching reports
   ✅ Supports topological execution ordering for pipelines
   ✅ Maintains backward compatibility with semantic matching fallback
   ✅ Clean separation of concerns - logical matching logic is centralized
```

## Integration Architecture

The integration follows these principles:

1. **Primary System**: `RuntimeTester` uses `logical_name_matching` when available
2. **Fallback System**: Falls back to semantic matching for backward compatibility
3. **API Consistency**: Maintains the same public API while using better internals
4. **Clean Delegation**: Converts between `ScriptExecutionSpec` and `EnhancedScriptExecutionSpec` as needed

## Usage Examples

### Basic Path Matching
```python
# Get path matches between two scripts
path_matches = tester.get_path_matches(spec_a, spec_b)
for match in path_matches:
    print(f"{match.matched_source_name} -> {match.matched_dest_name} ({match.confidence:.3f})")
```

### Pipeline Validation
```python
# Validate entire pipeline logical names
validation_results = tester.validate_pipeline_logical_names(pipeline_spec)
print(f"Pipeline valid: {validation_results['overall_valid']}")
```

### Data Compatibility Testing
```python
# Test data compatibility (now uses logical matching internally)
result = tester.test_data_compatibility_with_specs(spec_a, spec_b)
print(f"Compatible: {result.compatible}")
```

This integration provides the best of both worlds: sophisticated logical name matching when available, with graceful degradation to semantic matching for backward compatibility.
