---
tags:
  - design
  - level2_validation
  - property_path_validation
  - sagemaker_integration
  - implementation
keywords:
  - property path validation
  - SageMaker step types
  - documentation compliance
  - Level 2 enhancement
  - unified alignment tester
  - AWS SageMaker SDK
topics:
  - Level 2 validation
  - property path validation
  - SageMaker integration
  - validation framework
language: python
date of note: 2025-08-12
---

# Level 2 Property Path Validation Implementation

## Related Documents
- **[Level 2 Contract Specification Alignment Design](level2_contract_specification_alignment_design.md)** - Core Level 2 validation architecture
- **[Unified Alignment Tester Master Design](unified_alignment_tester_master_design.md)** - Complete system overview
- **[Unified Alignment Tester Architecture](unified_alignment_tester_architecture.md)** - Core architectural patterns
- **[Alignment Rules](../0_developer_guide/alignment_rules.md)** - Centralized alignment guidance and principles
- **[SageMaker Property Path Reference Database](../0_developer_guide/sagemaker_property_path_reference_database.md)** - Comprehensive database of valid SageMaker property paths by step type

## August 2025 Refactoring Update

**ARCHITECTURAL ENHANCEMENT**: The Level 2 Property Path Validation has been enhanced with modular architecture, step type awareness support, and integration with the enhanced ScriptContractValidator from Level 1, creating a unified validation approach across levels.

### Enhanced Module Integration
Level 2 property path validation now leverages the refactored modular architecture:
- **core_models.py**: StepTypeAwareAlignmentIssue for enhanced property path issue context
- **step_type_detection.py**: Step type detection for training script property path validation
- **utils.py**: Common utilities shared across validation levels
- **framework_patterns.py**: Framework-specific property path patterns
- **🆕 ScriptContractValidator Integration**: Level 1 enhanced path validation logic now informs Level 2 property path validation

### Key Enhancements
- **Cross-Level Integration**: Level 2 now leverages Level 1's enhanced path validation insights
- **Training Script Support**: Extended property path validation for training scripts with step type awareness
- **Enhanced Issue Context**: Step type-aware property path validation issues with framework information
- **Framework-Specific Patterns**: Property path patterns specific to XGBoost, PyTorch, and other ML frameworks
- **🆕 Three-Scenario Path Logic**: Integration with Level 1's sophisticated path validation scenarios
- **Improved Maintainability**: Modular components with clear boundaries for property path validation

## Overview

Successfully implemented **Level 2 Property Path Validation** as an enhancement to the unified alignment tester. This validation ensures that SageMaker Step Property Path References used in step specifications are valid according to official SageMaker documentation.

## Implementation Details

### Core Components

#### 1. SageMakerPropertyPathValidator (`property_path_validator.py`)

A dedicated validator module that:

- **Validates property paths** against official SageMaker documentation (v2.92.2)
- **Supports 10 SageMaker step types** with comprehensive property path definitions
- **Provides intelligent suggestions** for invalid property paths using similarity scoring
- **Handles pattern matching** for array indexing (e.g., `[*]`, `['metric_name']`)
- **Caches property path definitions** for performance optimization

**Supported Step Types:**
- TrainingStep (30 valid paths)
- ProcessingStep (26 valid paths) 
- TransformStep (22 valid paths)
- TuningStep (28 valid paths)
- CreateModelStep (14 valid paths)
- LambdaStep, CallbackStep, QualityCheckStep, ClarifyCheckStep, EMRStep

#### 2. Integration with ContractSpecificationAlignmentTester

Enhanced the existing Level 2 tester to include property path validation with Level 1 integration:

```python
# Enhanced: Validate property path references with Level 1 integration
property_path_issues = self._validate_property_paths(unified_spec['primary_spec'], contract_name)

# 🆕 NEW: Integration with Level 1 ScriptContractValidator insights
if hasattr(self, 'level1_validator') and self.level1_validator:
    # Leverage Level 1 path validation insights for enhanced Level 2 validation
    level1_path_insights = self._get_level1_path_insights(contract_name)
    property_path_issues = self._enhance_property_path_validation_with_level1(
        property_path_issues, level1_path_insights
    )

all_issues.extend(property_path_issues)
```

### Key Features

#### 1. Documentation-Based Validation

Property paths are validated against official SageMaker API documentation and our comprehensive reference database:
- **Primary Reference**: https://sagemaker.readthedocs.io/en/v2.92.2/amazon_sagemaker_model_building_pipeline.html#data-dependency-property-reference
- **Internal Database**: [SageMaker Property Path Reference Database](../0_developer_guide/sagemaker_property_path_reference_database.md)
- **Version**: SageMaker SDK v2.92.2
- **Coverage**: All major SageMaker step types with comprehensive property path definitions

#### 2. Intelligent Error Reporting

When invalid property paths are detected:
- **Error severity**: Clearly marked as ERROR level
- **Suggestions provided**: Top-ranked similar valid paths
- **Context included**: Step type, node type, logical name
- **Documentation links**: Direct reference to official docs

#### 3. Pattern Matching Support

Handles complex property path patterns:
- **Array indexing**: `FinalMetricDataList[*].Value`
- **Named indexing**: `ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri`
- **Nested properties**: `properties.ModelArtifacts.S3ModelArtifacts`

#### 4. Performance Optimization

- **Caching**: Property path definitions cached by step type
- **Lazy loading**: Definitions loaded only when needed
- **Efficient matching**: Regex-based pattern matching with fallbacks

## Test Results

The implementation was validated with comprehensive tests:

### Test Scenarios

1. **Valid TrainingStep paths**: ✅ Correctly validated
2. **Invalid property paths**: ✅ Detected with suggestions
3. **ProcessingStep paths**: ✅ Pattern matching works
4. **Unknown step types**: ✅ Graceful handling
5. **Integration testing**: ✅ Works with unified tester

### Sample Output

```
📝 Test 1: Valid TrainingStep Property Paths
Found 3 validation issues:
  INFO: Valid property path in output model_artifacts: properties.ModelArtifacts.S3ModelArtifacts
  INFO: Valid property path in output training_metrics: properties.FinalMetricDataList['accuracy'].Value
  INFO: Property path validation completed for dummy_training

📝 Test 2: Invalid Property Paths
Found 2 validation issues:
  ERROR: Invalid property path in output invalid_output: properties.InvalidPath.DoesNotExist
    Suggestions: properties.TrainingJobName, properties.TrainingJobArn, properties.TrainingJobStatus
  INFO: Property path validation completed for test_invalid
```

## Integration with Unified Alignment Tester

### Level 2 Enhancement with Level 1 Integration

The property path validation is seamlessly integrated into the existing Level 2 validation workflow with enhanced Level 1 integration:

1. **Contract-Specification alignment** (existing)
2. **Logical name validation** (existing)
3. **Data type consistency** (existing)
4. **Property path validation** (enhanced)
5. **🆕 Level 1 Path Validation Integration** (NEW)

### Enhanced Validation Flow

```python
# Level 2 validation now includes enhanced property path validation with Level 1 integration
def validate_contract(self, contract_name: str) -> Dict[str, Any]:
    # ... existing validation logic ...
    
    # Enhanced: Validate property path references with Level 1 insights
    property_path_issues = self._validate_property_paths(unified_spec['primary_spec'], contract_name)
    
    # 🆕 NEW: Integrate Level 1 ScriptContractValidator insights
    if self._has_level1_validation_results(contract_name):
        level1_insights = self._get_level1_validation_insights(contract_name)
        property_path_issues = self._enhance_with_level1_path_validation(
            property_path_issues, level1_insights
        )
    
    all_issues.extend(property_path_issues)
    
    return validation_result

def _enhance_with_level1_path_validation(self, property_path_issues: List[Dict], 
                                       level1_insights: Dict) -> List[Dict]:
    """Enhance Level 2 property path validation with Level 1 path validation insights."""
    enhanced_issues = property_path_issues.copy()
    
    # Extract Level 1 path validation scenarios
    level1_scenarios = level1_insights.get('path_validation_scenarios', {})
    
    for issue in enhanced_issues:
        if issue.get('category') == 'property_path_validation':
            property_path = issue.get('details', {}).get('property_path', '')
            
            # Check if this property path relates to Level 1 validated paths
            related_level1_scenario = self._find_related_level1_scenario(
                property_path, level1_scenarios
            )
            
            if related_level1_scenario:
                # Enhance issue with Level 1 context
                issue['details']['level1_validation_context'] = {
                    'scenario': related_level1_scenario['validation_scenario'],
                    'script_path_usage': related_level1_scenario.get('script_path'),
                    'contract_path_mapping': related_level1_scenario.get('contract_path'),
                    'validation_result': related_level1_scenario.get('severity')
                }
                
                # Adjust severity based on Level 1 results
                if related_level1_scenario.get('severity') == 'ERROR':
                    issue['severity'] = 'CRITICAL'  # Escalate if Level 1 found path issues
                    issue['message'] += f" (Related Level 1 path validation failed: {related_level1_scenario.get('message', '')})"
                elif related_level1_scenario.get('severity') == 'INFO':
                    issue['message'] += f" (Level 1 validation: {related_level1_scenario.get('validation_scenario', 'validated')})"
    
    return enhanced_issues

def _find_related_level1_scenario(self, property_path: str, 
                                level1_scenarios: Dict) -> Optional[Dict]:
    """Find Level 1 path validation scenario related to a Level 2 property path."""
    # Extract the actual file path from property path (e.g., properties.ModelArtifacts.S3ModelArtifacts -> /opt/ml/model)
    extracted_path = self._extract_file_path_from_property_path(property_path)
    
    if extracted_path:
        for scenario in level1_scenarios.get('scenarios', []):
            script_path = scenario.get('script_path', '')
            contract_path = scenario.get('contract_path', '')
            
            # Check if the extracted path matches Level 1 validated paths
            if (extracted_path == script_path or 
                extracted_path == contract_path or
                extracted_path in script_path or
                script_path in extracted_path):
                return scenario
    
    return None

def _extract_file_path_from_property_path(self, property_path: str) -> Optional[str]:
    """Extract actual file system path from SageMaker property path."""
    # Map common SageMaker property paths to their corresponding file system paths
    property_to_path_mapping = {
        'properties.ModelArtifacts.S3ModelArtifacts': '/opt/ml/model',
        'properties.ProcessingOutputConfig.Outputs': '/opt/ml/processing/output',
        'properties.ProcessingInputs': '/opt/ml/processing/input',
        'properties.TrainingJobName': '/opt/ml/model',
        'properties.TransformOutput.S3OutputPath': '/opt/ml/processing/output',
        'properties.HyperParameters': '/opt/ml/input/config/hyperparameters.json'
    }
    
    # Check for direct mapping
    for prop_pattern, file_path in property_to_path_mapping.items():
        if property_path.startswith(prop_pattern):
            return file_path
    
    # Extract from array indexing patterns
    if '[' in property_path and ']' in property_path:
        base_property = property_path.split('[')[0]
        if base_property in property_to_path_mapping:
            return property_to_path_mapping[base_property]
    
    return None
```

### Cross-Level Validation Benefits

The integration between Level 1 and Level 2 validation provides several key benefits:

#### 1. **Consistency Validation**
- **Path Consistency**: Ensures property paths in specifications align with actual script path usage
- **Contract Alignment**: Validates that Level 2 property paths reference paths that Level 1 has validated in contracts
- **Scenario Awareness**: Level 2 understands Level 1's three-scenario path validation results

#### 2. **Enhanced Error Context**
- **Root Cause Analysis**: Level 2 errors can be traced back to Level 1 path validation issues
- **Severity Escalation**: Level 2 issues are escalated to CRITICAL when related Level 1 validation fails
- **Comprehensive Feedback**: Developers get both script-level and specification-level path validation feedback

#### 3. **Intelligent Recommendations**
- **Context-Aware Suggestions**: Level 2 suggestions consider Level 1 validation scenarios
- **Path Construction Awareness**: Understanding of parent-child directory relationships from Level 1
- **Framework-Specific Guidance**: Recommendations tailored to detected frameworks (XGBoost, PyTorch, etc.)

### Enhanced Integration Examples

#### Example 1: XGBoost Training Path Validation
```python
# Level 1 validates script path usage
level1_result = {
    'script_path': '/opt/ml/input/data/config',
    'contract_path': '/opt/ml/input/data/config/hyperparameters.json',
    'validation_scenario': 'parent_child_relationship',
    'severity': 'INFO'
}

# Level 2 validates property path in specification
property_path = 'properties.HyperParameters'
# Enhanced with Level 1 context:
enhanced_issue = {
    'severity': 'INFO',
    'message': 'Valid property path: properties.HyperParameters (Level 1 validation: parent_child_relationship)',
    'details': {
        'level1_validation_context': {
            'scenario': 'parent_child_relationship',
            'script_path_usage': '/opt/ml/input/data/config',
            'contract_path_mapping': '/opt/ml/input/data/config/hyperparameters.json'
        }
    }
}
```

#### Example 2: Path Validation Failure Escalation
```python
# Level 1 detects undeclared path usage
level1_result = {
    'script_path': '/opt/ml/undeclared/path',
    'severity': 'ERROR',
    'message': 'Script uses undeclared SageMaker path'
}

# Level 2 property path validation escalated
enhanced_issue = {
    'severity': 'CRITICAL',  # Escalated from ERROR
    'message': 'Invalid property path: properties.UndeclaredPath (Related Level 1 path validation failed: Script uses undeclared SageMaker path)',
    'recommendation': 'Fix Level 1 path validation issues before addressing Level 2 property paths'
}
```

## Benefits

### 1. Runtime Error Prevention

- **Early detection** of invalid property paths before pipeline execution
- **Prevents SageMaker runtime failures** due to incorrect property references
- **Saves development time** by catching errors during validation phase

### 2. Documentation Compliance

- **Ensures compliance** with official SageMaker documentation
- **Stays current** with SageMaker SDK version v2.92.2
- **Provides authoritative validation** based on AWS documentation

### 3. Developer Experience

- **Clear error messages** with actionable suggestions
- **Intelligent suggestions** based on similarity scoring
- **Comprehensive coverage** of all major SageMaker step types

### 4. Maintainability

- **Modular design** with separate validator class
- **Easy to extend** for new SageMaker step types
- **Cached definitions** for performance
- **Well-documented** with clear API

## Usage Examples

### Standalone Usage

```python
from src.cursus.validation.alignment.property_path_validator import SageMakerPropertyPathValidator

validator = SageMakerPropertyPathValidator()

# Validate a specification
issues = validator.validate_specification_property_paths(specification, contract_name)

# Get documentation for a step type
doc_info = validator.get_step_type_documentation('training', 'training')

# List all supported step types
supported_types = validator.list_supported_step_types()

# Access the comprehensive property path database
# See: slipbox/0_developer_guide/sagemaker_property_path_reference_database.md
property_paths = validator.get_valid_property_paths('TrainingStep')
```

### Integrated Usage

```python
from src.cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester

tester = UnifiedAlignmentTester()

# Run Level 2 validation (includes property path validation)
report = tester.run_level_validation(level=2, target_scripts=['dummy_training'])
```

## Future Enhancements

### Potential Improvements

1. **Dynamic documentation updates**: Automatically sync with latest SageMaker documentation
2. **Custom property paths**: Support for user-defined property paths
3. **Step type auto-detection**: Automatically detect step type from specification
4. **Performance metrics**: Track validation performance and optimization opportunities

### Extension Points

1. **New step types**: Easy to add support for new SageMaker step types
2. **Custom validators**: Plugin architecture for custom property path validators
3. **Integration hooks**: Additional integration points for other validation systems

## Conclusion

The Level 2 Property Path Validation implementation successfully enhances the unified alignment tester with:

- ✅ **Comprehensive validation** of SageMaker property paths
- ✅ **Documentation-based accuracy** using official SageMaker references
- ✅ **Intelligent error reporting** with actionable suggestions
- ✅ **Seamless integration** with existing validation workflow
- ✅ **High performance** with caching and optimization
- ✅ **Extensible design** for future enhancements

This enhancement significantly improves the reliability and developer experience of the SageMaker pipeline validation system.
