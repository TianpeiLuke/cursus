---
tags:
  - project
  - planning
  - pipeline_api
  - portability
  - implementation
keywords:
  - PIPELINE_EXECUTION_TEMP_DIR
  - pipeline portability
  - runtime configuration
  - output destinations
  - parameter substitution
  - Join operations
  - external system integration
  - configuration management
topics:
  - pipeline portability enhancement
  - runtime parameter integration
  - output destination management
  - external system compatibility
language: python
date of note: 2025-09-18
---

# PIPELINE_EXECUTION_TEMP_DIR Implementation Plan

## Executive Summary

This project plan outlines the systematic implementation of PIPELINE_EXECUTION_TEMP_DIR support in the Cursus framework to enable portable pipeline execution across different environments and external systems.

## Purpose and Business Value

### Problem Statement

The current Cursus framework relies on hard-coded S3 directories (`pipeline_s3_loc`) that are embedded in saved configuration files. When external systems load these configurations to trigger pipeline execution, they cannot modify the pre-computed output paths without editing configuration files, creating a significant portability barrier.

### Solution Overview

Implement support for `PIPELINE_EXECUTION_TEMP_DIR` as a runtime parameter that allows external systems to dynamically specify output destinations, making pipelines truly portable across different AWS accounts, environments, and execution contexts.

### Business Benefits

1. **Environment Portability**: Seamless pipeline execution across development, testing, and production environments
2. **External System Integration**: Enable third-party systems to execute pipelines with custom output locations
3. **Operational Efficiency**: Reduce configuration management overhead and deployment complexity
4. **Cost Optimization**: Allow dynamic resource allocation and output location management
5. **Compliance**: Support environment-specific data governance and retention policies

## Key Design References

This implementation is based on comprehensive design analysis documented in:

- **Primary Design**: `slipbox/1_design/pipeline_execution_temp_dir_integration.md`
- **Supporting Architecture**: `slipbox/1_design/cursus_framework_output_management.md`

### Core Technical Approach

The solution implements a three-tier approach:

1. **Parameter Access**: Enable step builders to receive pipeline parameters from PipelineAssembler
2. **Path Resolution**: Intelligent selection between runtime parameters and static configuration
3. **Path Construction**: Replace string interpolation with `sagemaker.workflow.functions.Join()` for proper parameter substitution

### Critical Technical Discovery

The fundamental issue is that f-string path construction breaks with `ParameterString` objects:

```python
# BROKEN - Fails with ParameterString objects
destination = f"{base_path}/step_type/{logical_name}"

# CORRECT - Works with both str and ParameterString
from sagemaker.workflow.functions import Join
destination = Join(on="/", values=[base_path, "step_type", logical_name])
```

## Implementation Phases

### Phase 1: Core Infrastructure (Days 1-3)
**Status: ✅ COMPLETED**

#### Core Components Enhanced
- [x] **StepBuilderBase Enhancement**: Added `execution_prefix` attribute, `set_execution_prefix()` method, and `_get_base_output_path()` for intelligent path resolution
- [x] **PipelineAssembler Integration**: Updated `_initialize_step_builders()` to extract and pass PIPELINE_EXECUTION_TEMP_DIR, modified `_generate_outputs()` to use Join() pattern
- [x] **Backward Compatibility**: Ensured seamless fallback to existing configurations when no parameters provided

**Deliverables:**
- Enhanced `src/cursus/core/base/builder_base.py`
- Updated `src/cursus/core/assembler/pipeline_assembler.py`
- Backward compatibility maintained

### Phase 2: Complete Step Builder Migration and Optimization (Days 4-9)
**Status: ✅ COMPLETED (All 8 Step Builders)**

#### Step Builders Updated with Join() Pattern (9/9 Required)
- [x] **PackageStepBuilder** - Model packaging output paths
- [x] **XGBoostModelEvalStepBuilder** - Model evaluation outputs  
- [x] **ModelCalibrationStepBuilder** - Calibration outputs with job_type support
- [x] **TabularPreprocessingStepBuilder** - Data preprocessing outputs
- [x] **RiskTableMappingStepBuilder** - Risk mapping outputs
- [x] **PayloadStepBuilder** - Payload generation outputs
- [x] **CurrencyConversionStepBuilder** - Currency conversion outputs
- [x] **PyTorchTrainingStepBuilder** - PyTorch training outputs
- [x] **BatchTransformStepBuilder** - Batch transform outputs with consistent folder structure

#### Training Step Builders with Special Lambda Optimizations (2/2 Required)
- [x] **XGBoostTrainingStepBuilder** - Enhanced with Lambda-optimized hyperparameters handling
- [x] **DummyTrainingStepBuilder** - Enhanced with Lambda-optimized hyperparameters and model upload

#### Step Builders Not Requiring Join() Pattern Migration (4/4 Verified)
- [x] **CradleDataLoadingStepBuilder** - Returns empty dictionary, uses different output mechanism
- [x] **PyTorchModelStepBuilder** - CreateModelStep handles outputs automatically
- [x] **RegistrationStepBuilder** - Registration steps produce no outputs
- [x] **XGBoostModelStepBuilder** - CreateModelStep handles outputs automatically

#### Comprehensive Code Cleanup and Optimization
- [x] **Lambda-Optimized File Operations**: UUID-based temporary directories, comprehensive error handling, retry logic with exponential backoff, robust cleanup, file size monitoring
- [x] **Complete Removal of Obsolete Methods**: Removed `_validate_s3_uri`, `_get_s3_directory_path`, `_normalize_s3_uri` (~130 lines of code eliminated)
- [x] **Unified System Path Enforcement**: Removed user override branches, achieved pure Join() architecture with zero exceptions
- [x] **Consistent Path Construction**: All components use identical patterns

**Standardized Migration Pattern:**
```python
# OLD PATTERN (f-string - breaks with ParameterString):
destination = f"{self.config.pipeline_s3_loc}/step_type/{logical_name}"

# NEW PATTERN (Join() - works with both str and ParameterString):
from sagemaker.workflow.functions import Join
base_output_path = self._get_base_output_path()
destination = Join(on="/", values=[base_output_path, "step_type", logical_name])
```

### Phase 3: End-to-End Parameter Flow (Days 10-13)
**Status: ✅ COMPLETED**

#### 3.1 DAGCompiler Parameter Management Implementation
- [x] **Parameter Import Centralization**: Moved all parameter imports from DynamicPipelineTemplate to DAGCompiler with robust fallback
- [x] **Default Parameter Handling**: Implemented intelligent default parameter set when none provided by external systems
- [x] **Parameter Storage and Forwarding**: Store parameters internally and pass to DynamicPipelineTemplate
- [x] **Import Fallback Support**: Handle cases where `mods_workflow_core.utils.constants` is not available with local definitions
- [x] **Complete Parameter Set Support**: Handle all parameters from `mods_workflow_core.utils.constants`:
  - `PIPELINE_EXECUTION_TEMP_DIR` (output destinations)
  - `KMS_ENCRYPTION_KEY_PARAM` (security)
  - `PROCESSING_JOB_SHARED_NETWORK_CONFIG` (network config)
  - `SECURITY_GROUP_ID` (network security)
  - `VPC_SUBNET` (network configuration)

#### 3.2 PipelineTemplateBase Parameter Management Implementation
- [x] **Constructor Parameter Addition**: Added `pipeline_parameters` parameter to constructor with proper type hints
- [x] **Parameter Storage**: Store parameters directly in constructor initialization with type safety
- [x] **Parameter Retrieval Method**: Implemented `_get_pipeline_parameters()` with stored parameter return and fallback logic
- [x] **Setter Method**: Maintained `set_pipeline_parameters()` for additional flexibility
- [x] **Type Safety**: Added consistent type hints `Optional[List[Union[str, ParameterString]]]`

#### 3.3 DynamicPipelineTemplate Simplification
- [x] **Remove Parameter Imports**: Removed all parameter imports (now handled in DAGCompiler)
- [x] **Remove Parameter Logic**: Removed `_get_pipeline_parameters()` method (inherited from parent)
- [x] **Simplify Constructor**: Pass parameters directly to parent constructor
- [x] **Clean Up Code**: Removed all redundant parameter handling code
- [x] **Type Safety**: Added consistent type hints and proper imports

#### 3.4 External System Integration Architecture
- [x] **Top-Level Parameter Injection**: Enabled external systems to provide complete parameter set to DAGCompiler
- [x] **Parameter Flow Validation**: Ensured parameters propagate through simplified layers: External System → DAGCompiler → DynamicTemplate → PipelineTemplateBase → PipelineAssembler → StepBuilders
- [x] **Backward Compatibility**: Maintained existing behavior when no parameters provided (DAGCompiler provides intelligent defaults)
- [x] **Type Safety Implementation**: Consistent type hints across all components: `Optional[List[Union[str, ParameterString]]]`
- [x] **Core Functionality Testing**: Verified parameter flow through unit testing of core components

### Phase 4: Testing and Validation (Days 13-16)
**Status: ⏳ PLANNED**

#### 4.1 Unit Testing
- [ ] Test `_get_base_output_path()` with and without execution_prefix
- [ ] Test parameter extraction in `_initialize_step_builders()`
- [ ] Test Join-based path construction in all updated builders
- [ ] Test backward compatibility with existing configurations

#### 4.2 Integration Testing
- [ ] Test complete pipeline with PIPELINE_EXECUTION_TEMP_DIR parameter
- [ ] Verify parameter substitution at SageMaker runtime
- [ ] Test external system integration scenarios
- [ ] Validate cross-environment pipeline execution

#### 4.3 Performance Testing
- [ ] Measure impact of Join() operations vs f-strings
- [ ] Test parameter resolution performance
- [ ] Validate memory usage with ParameterString objects

### Phase 5: Documentation and Rollout (Days 17-20)
**Status: ⏳ PLANNED**

#### 5.1 Documentation Updates
- [ ] Update developer guides with new patterns
- [ ] Create migration guide for existing step builders
- [ ] Document external system integration examples
- [ ] Update API documentation with parameter usage

#### 5.2 Migration Support
- [ ] Create automated migration tools for remaining step builders
- [ ] Develop validation scripts to detect f-string usage
- [ ] Provide backward compatibility testing framework

#### 5.3 Rollout Strategy
- [ ] Gradual rollout to development environments
- [ ] Validation in staging environments
- [ ] Production deployment with monitoring
- [ ] Post-deployment validation and optimization

## Risk Assessment and Mitigation

### Technical Risks

1. **Parameter Substitution Failures**
   - **Risk**: SageMaker parameter substitution may fail in edge cases
   - **Mitigation**: Comprehensive testing with various parameter types and values
   - **Status**: Mitigated through Join() pattern adoption

2. **Backward Compatibility Issues**
   - **Risk**: Existing pipelines may break with new parameter handling
   - **Mitigation**: Fallback to `pipeline_s3_loc` when no parameter provided
   - **Status**: Addressed in Phase 1 implementation

3. **Performance Impact**
   - **Risk**: Join() operations may be slower than f-strings
   - **Mitigation**: Performance testing and optimization in Phase 4
   - **Status**: To be validated

### Operational Risks

1. **External System Integration Complexity**
   - **Risk**: External systems may struggle with parameter configuration
   - **Mitigation**: Comprehensive documentation and examples
   - **Status**: Planned for Phase 5

2. **Migration Effort**
   - **Risk**: Large number of step builders require manual updates
   - **Mitigation**: Standardized migration pattern and automated tools
   - **Status**: Pattern established, automation planned

## Success Metrics

### Technical Metrics
- [ ] 100% of step builders migrated to Join() pattern
- [ ] Zero backward compatibility breaks
- [ ] Parameter substitution success rate > 99.9%
- [ ] Performance impact < 5% compared to f-strings

### Business Metrics
- [ ] External system integration time reduced by 80%
- [ ] Environment deployment time reduced by 60%
- [ ] Configuration management overhead reduced by 70%
- [ ] Cross-environment pipeline success rate > 95%

## Current Status Summary

### ✅ Completed Work (Phases 1, 2, 2.5, and 3)
- **Core Infrastructure**: Complete implementation in StepBuilderBase and PipelineAssembler
- **Step Builder Migration**: All 8 step builders migrated to Join() pattern with comprehensive optimizations
- **Lambda Optimizations**: Enhanced file operations for serverless environments
- **Code Cleanup**: Removed ~130 lines of obsolete S3 path manipulation code
- **Unified Architecture**: Pure Join() pattern with zero exceptions across all components
- **End-to-End Parameter Flow**: Complete parameter propagation from external systems through DAGCompiler
- **Type Safety Implementation**: Consistent type hints across all components
- **Centralized Parameter Management**: Smart defaults with robust fallback support
- **Backward Compatibility**: Ensured seamless fallback to existing configurations
- **Comprehensive Documentation**: Updated design documents with complete implementation details

### ✅ Major Technical Achievements
- **Pure Join() Architecture**: Zero legacy path manipulation code remaining
- **Lambda-Ready Operations**: Optimized for AWS Lambda with robust error handling and resource management
- **Unified System Path Enforcement**: All step builders use consistent path construction patterns
- **Enhanced Reliability**: Comprehensive error handling, retry logic, and cleanup mechanisms
- **True Pipeline Portability**: All artifacts consistently organized under unified base paths
- **Complete Parameter Flow**: External systems can now provide custom PIPELINE_EXECUTION_TEMP_DIR parameters
- **Smart Default Management**: Intelligent parameter provisioning when none provided by external systems
- **Type-Safe Integration**: Consistent `Optional[List[Union[str, ParameterString]]]` across all components
- **Centralized Constants**: Single source of truth for all pipeline parameters with fallback support

### ⏳ Upcoming Priorities (Phase 4+)
1. **Comprehensive Testing**: Unit, integration, and performance validation
2. **Documentation and Rollout**: Developer guides, migration tools, and production deployment
3. **Performance Optimization**: Validate and optimize Join() operations vs f-strings
4. **External System Integration Examples**: Create comprehensive integration patterns and documentation

## Resource Requirements

### Development Team
- **Lead Developer**: 1 FTE for architecture and complex integrations
- **Implementation Developer**: 1 FTE for step builder migrations
- **QA Engineer**: 0.5 FTE for testing and validation
- **Technical Writer**: 0.25 FTE for documentation

### Timeline
- **Total Duration**: 20 days
- **Critical Path**: Core infrastructure → Step builder migration → End-to-end testing
- **Parallel Work**: Documentation can proceed alongside implementation

### Infrastructure
- Development and staging environments for testing
- External system simulation for integration testing
- Performance monitoring tools for validation

## Conclusion

The PIPELINE_EXECUTION_TEMP_DIR implementation represents a significant enhancement to the Cursus framework's portability and external system integration capabilities. With the core infrastructure already completed and a clear migration pattern established, the remaining work focuses on systematic application of the proven approach across all step builders.

The project is well-positioned for successful completion within the planned timeline, with strong technical foundations and comprehensive risk mitigation strategies in place.

## References

### Design Documents
- [Integration of PIPELINE_EXECUTION_TEMP_DIR in Cursus Framework](../1_design/pipeline_execution_temp_dir_integration.md) - Primary technical design and implementation approach
- [Cursus Framework Output Management Architecture](../1_design/cursus_framework_output_management.md) - Supporting architecture and current limitations analysis

### Implementation Files
- `src/cursus/core/base/builder_base.py` - Core infrastructure implementation
- `src/cursus/core/assembler/pipeline_assembler.py` - Parameter passing and Join() integration
- `src/cursus/steps/builders/builder_package_step.py` - Reference migration example
- `src/cursus/steps/builders/builder_xgboost_model_eval_step.py` - Model evaluation migration
- `src/cursus/steps/builders/builder_model_calibration_step.py` - Calibration with job_type support

### Related Standards
- [Documentation YAML Frontmatter Standard](../1_design/documentation_yaml_frontmatter_standard.md) - Documentation formatting guidelines
