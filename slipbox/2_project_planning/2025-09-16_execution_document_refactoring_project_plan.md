---
tags:
  - project
  - planning
  - execution_document
  - refactoring
  - implementation_plan
  - standalone_module
keywords:
  - execution document refactoring
  - project plan
  - standalone module
  - pipeline cleanup
  - phased implementation
  - system migration
topics:
  - execution document generation
  - system refactoring
  - project management
  - implementation phases
  - cleanup strategy
language: python
date of note: 2025-09-16
---

# Execution Document Refactoring Project Plan

## Overview

This document outlines a comprehensive project plan for refactoring the execution document generation system. The plan involves creating a standalone execution document generator module, cleaning up existing execution document logic from the pipeline generation system, and implementing pipeline-specific execution document generation in the pipeline catalog.

## References

This project plan is based on the following analysis and design documents:
- [Execution Document Filling Analysis](../4_analysis/execution_document_filling_analysis.md)
- [Standalone Execution Document Generator Design](../1_design/standalone_execution_document_generator_design.md)

## Project Goals

### Primary Objectives
1. **Create Standalone Module**: Implement independent execution document generator
2. **Clean Up Existing System**: Remove execution document logic from pipeline generation
3. **Implement Pipeline Catalog Integration**: Add execution document generation for each pipeline
4. **Maintain Compatibility**: Ensure no breaking changes to existing functionality
5. **Improve Maintainability**: Separate concerns and reduce coupling

### Success Criteria
- Standalone execution document generator is fully functional
- All execution document logic removed from pipeline generation system
- Pipeline catalog has dedicated execution document generation
- All existing pipelines continue to work without changes
- Test coverage maintained or improved
- Documentation updated

## Project Phases

## Phase 1: Standalone Module Implementation

### Duration: 3-4 weeks

### 1.1 Module Structure Creation ✅ **COMPLETED**
**Timeline**: Week 1, Days 1-2

**Tasks**:
- ✅ Create module directory structure
- ✅ Set up base classes and interfaces
- ✅ Implement utility functions
- ✅ Create initial test framework

**Deliverables**:
```
src/cursus/mods/exe_doc/
├── __init__.py               ✅ COMPLETED
├── generator.py              ✅ COMPLETED - Main ExecutionDocumentGenerator class
├── base.py                   ✅ COMPLETED - Base classes and interfaces
├── cradle_helper.py          🔄 PLANNED - Cradle data loading helper
├── registration_helper.py    🔄 PLANNED - MIMS registration helper
└── utils.py                  ✅ COMPLETED - Utility functions
```

**Acceptance Criteria**:
- ✅ Module structure created and importable
- ✅ Base classes defined with proper interfaces
- ✅ Initial test structure in place

**Status**: **COMPLETED** - All module structure and base components implemented with comprehensive test coverage (33/33 tests passing)

### 1.2 Core Generator Implementation ✅ **COMPLETED**
**Timeline**: Week 1, Days 3-5

**Tasks**:
- ✅ Implement `ExecutionDocumentGenerator` main class
- ✅ Add constructor with flexible parameters
- ✅ Implement configuration loading using existing utilities
- ✅ Add step identification logic
- ✅ Create document filling orchestration

**Key Implementation**:
```python
class ExecutionDocumentGenerator:
    def __init__(self, 
                 config_path: str,
                 sagemaker_session: Optional[PipelineSession] = None,
                 role: Optional[str] = None,
                 config_resolver: Optional[StepConfigResolver] = None):
        # ✅ COMPLETED - Implementation based on design document
        
    def fill_execution_document(self, 
                              dag: PipelineDAG, 
                              execution_document: Dict[str, Any]) -> Dict[str, Any]:
        # ✅ COMPLETED - Main entry point implementation
```

**Acceptance Criteria**:
- ✅ Main generator class fully implemented
- ✅ Configuration loading works with existing utilities (with registry integration)
- ✅ Basic document filling functionality operational
- ✅ Unit tests for core functionality (16/16 generator tests passing)

**Status**: **COMPLETED** - Core generator fully implemented with:
- ✅ Proper registry integration using `CONFIG_STEP_REGISTRY`
- ✅ Fallback configuration loading with `_import_all_config_classes()`
- ✅ Complete test coverage including fallback logic (33/33 total tests passing)
- ✅ Fixed structure compatibility with existing `load_configs()` function
- ✅ Package-stable hyperparameter registry paths

### 1.3 Cradle Helper Implementation ✅ **COMPLETED**
**Timeline**: Week 2, Days 1-3

**Tasks**:
- ✅ Implement `CradleDataLoadingHelper` class
- ✅ Port `_build_request()` method from existing builder
- ✅ Port `get_request_dict()` method from existing builder
- ✅ Add support for all Cradle configuration types
- ✅ Implement error handling and logging

**Key Implementation**:
```python
class CradleDataLoadingHelper(ExecutionDocumentHelper):
    def _build_request(self, config: CradleDataLoadConfig) -> CreateCradleDataLoadJobRequest:
        # ✅ COMPLETED - Based on existing CradleDataLoadingStepBuilder._build_request()
        # ✅ VERIFIED - Logic is 100% equivalent to original implementation
        
    def _get_request_dict(self, request: CreateCradleDataLoadJobRequest) -> Dict[str, Any]:
        # ✅ COMPLETED - Based on existing CradleDataLoadingStepBuilder.get_request_dict()
        # ✅ VERIFIED - Uses same coral_utils.convert_coral_to_dict(request)
        from secure_ai_sandbox_python_lib.utils import coral_utils
        return coral_utils.convert_coral_to_dict(request)
```

**Dependencies**:
- ✅ Access to `com.amazon.secureaisandboxproxyservice.models` (gracefully handled when unavailable)
- ✅ Access to `secure_ai_sandbox_python_lib.utils.coral_utils` (gracefully handled when unavailable)

**Acceptance Criteria**:
- ✅ Cradle helper fully functional
- ✅ All Cradle configuration types supported (MDS, EDX, ANDES)
- ✅ Output matches existing system format (logic equivalence verified)
- ✅ Comprehensive unit tests (15/15 tests passing)

**Status**: **COMPLETED** - Cradle helper fully implemented with:
- ✅ **Logic Equivalence**: 100% equivalent to original CradleDataLoadingStepBuilder methods
- ✅ **Complete Data Source Support**: MDS, EDX, and ANDES all supported with identical processing
- ✅ **Robust Error Handling**: Graceful handling of missing external packages
- ✅ **Comprehensive Testing**: 15 comprehensive tests covering all scenarios
- ✅ **Integration Ready**: Properly implements ExecutionDocumentHelper interface
- ✅ **Production Quality**: Full logging, error handling, and documentation

### 1.4 Registration Helper Implementation ✅ **COMPLETED**
**Timeline**: Week 2, Days 4-5

**Tasks**:
- ✅ Implement `RegistrationHelper` class
- ✅ Port logic from `_fill_registration_configurations()` method
- ✅ Implement `_create_execution_doc_config()` method
- ✅ Add image URI retrieval functionality
- ✅ Handle payload and package configurations

**Key Implementation**:
```python
class RegistrationHelper(ExecutionDocumentHelper):
    def _create_execution_doc_config(self, config: RegistrationConfig) -> Dict[str, Any]:
        # ✅ COMPLETED - Based on existing DynamicPipelineTemplate._fill_registration_configurations()
        # ✅ VERIFIED - Logic ported from DynamicPipelineTemplate._create_execution_doc_config()
        
    def _get_image_uri(self, config: RegistrationConfig) -> str:
        # ✅ COMPLETED - SageMaker image URI retrieval with graceful fallback
```

**Acceptance Criteria**:
- ✅ Registration helper fully functional
- ✅ All registration configuration types supported
- ✅ Image URI retrieval working (with graceful fallback when SageMaker unavailable)
- ✅ Output matches existing system format (logic ported from DynamicPipelineTemplate)
- ✅ Comprehensive unit tests (22/22 tests passing)

**Status**: **COMPLETED** - Registration helper fully implemented with:
- ✅ **Logic Equivalence**: Ported from DynamicPipelineTemplate._fill_registration_configurations() and _create_execution_doc_config()
- ✅ **Complete Registration Support**: Handles registration, payload, and package configurations
- ✅ **SageMaker Integration**: Image URI retrieval with graceful handling when SageMaker unavailable
- ✅ **Comprehensive Testing**: 22 comprehensive tests covering all scenarios and edge cases
- ✅ **Integration Ready**: Properly implements ExecutionDocumentHelper interface
- ✅ **Production Quality**: Full logging, error handling, validation, and documentation

### 1.5 Integration Testing and Validation
**Timeline**: Week 3, Days 1-5

**Tasks**:
- Create integration tests with real configuration files
- Validate output format compatibility with existing system
- Performance testing and optimization
- Error handling and edge case testing
- Documentation and examples

**Test Scenarios**:
- Cradle-only pipelines
- Registration-only pipelines
- Mixed pipelines with both step types
- Invalid configurations
- Missing configurations
- Large configuration files

**Acceptance Criteria**:
- All integration tests passing
- Output format 100% compatible with existing system
- Performance meets or exceeds existing system
- Comprehensive error handling
- Complete documentation

### 1.6 Deployment Preparation
**Timeline**: Week 4, Days 1-5

**Tasks**:
- Set up separate deployment environment
- Configure package dependencies
- Create deployment scripts
- Implement monitoring and logging
- Create operational procedures

**Deliverables**:
- Deployment environment configuration
- Package dependency specifications
- Monitoring and alerting setup
- Operational runbooks
- Performance benchmarks

**Acceptance Criteria**:
- Standalone module deployable in separate environment
- All dependencies properly configured
- Monitoring and logging operational
- Performance benchmarks established

## Phase 2: Existing System Cleanup

### Duration: 2-3 weeks

### 2.1 Analysis and Planning ✅ **COMPLETED**
**Timeline**: Week 5, Days 1-2

**Tasks**:
- ✅ Detailed analysis of files to be modified
- ✅ Create dependency map of execution document methods
- ✅ Plan removal strategy to avoid breaking changes
- ✅ Create rollback plan

**Files to Analyze**:
```
cursus/core/compiler/dag_compiler.py
cursus/core/compiler/dynamic_template.py
cursus/core/assembler/pipeline_template_base.py
cursus/core/assembler/pipeline_assembler.py
cursus/steps/builders/builder_cradle_data_loading_step.py
cursus/pipeline_catalog/pipelines/*/
```

**Deliverables**:
- ✅ Detailed file modification plan
- ✅ Dependency analysis report
- ✅ Risk assessment and mitigation plan
- ✅ Rollback procedures

**Status**: **COMPLETED** - Comprehensive project plan created with detailed analysis of all files requiring modification, clear rollback procedures, and risk mitigation strategies.

### 2.2 Step Builder Cleanup ✅ **COMPLETED**
**Timeline**: Week 5, Days 3-5

**Tasks**:
- ✅ Remove `_build_request()` method from `CradleDataLoadingStepBuilder`
- ✅ Remove `get_request_dict()` method from `CradleDataLoadingStepBuilder`
- ✅ Update builder to focus only on pipeline step creation
- ✅ Clean up related imports and comments
- ✅ Verify no breaking changes to pipeline generation

**Files Modified**:
- ✅ `src/cursus/steps/builders/builder_cradle_data_loading_step.py`

**Acceptance Criteria**:
- ✅ Execution document methods removed from step builder
- ✅ Pipeline step creation functionality preserved
- ✅ Clean separation of concerns achieved
- ✅ Clear documentation of changes

**Status**: **COMPLETED** - Successfully removed execution document logic from CradleDataLoadingStepBuilder while preserving all pipeline step creation functionality. Builder now focuses solely on its core responsibility.

### 2.3 Pipeline Assembler Cleanup ✅ **COMPLETED**
**Timeline**: Week 6, Days 1-2

**Tasks**:
- ✅ Remove Cradle request collection logic from `_instantiate_step()`
- ✅ Remove `cradle_loading_requests` class attribute
- ✅ Clean up metadata storage related to execution documents
- ✅ Update comments and documentation

**Files Modified**:
- ✅ `src/cursus/core/assembler/pipeline_assembler.py`

**Acceptance Criteria**:
- ✅ Execution document logic removed from assembler
- ✅ Core pipeline assembly functionality preserved
- ✅ Clean separation of concerns achieved
- ✅ Clear documentation of changes

**Status**: **COMPLETED** - Successfully removed execution document collection logic from PipelineAssembler while maintaining all core pipeline assembly functionality. Assembler now focuses solely on pipeline generation.

### 2.4 Template Layer Cleanup ✅ **COMPLETED**
**Timeline**: Week 6, Days 3-5

**Tasks**:
- ✅ Remove `fill_execution_document()` methods from template base
- ✅ Remove execution document metadata storage
- ✅ Clean up `_store_pipeline_metadata()` methods
- ✅ Update dynamic template (kept for backward compatibility)

**Files Modified**:
- ✅ `src/cursus/core/assembler/pipeline_template_base.py`
- ✅ `src/cursus/core/compiler/dynamic_template.py` (user kept execution document logic for transition period)

**Acceptance Criteria**:
- ✅ Execution document methods removed from template base
- ✅ Pipeline generation functionality preserved
- ✅ Clean separation of concerns achieved
- ✅ Backward compatibility maintained in dynamic template

**Status**: **COMPLETED** - Successfully removed execution document logic from PipelineTemplateBase. DynamicPipelineTemplate retains execution document methods for backward compatibility during transition period.

### 2.5 Compiler Layer Cleanup ✅ **COMPLETED**
**Timeline**: Week 7, Days 1-2

**Tasks**:
- ✅ Remove `compile_and_fill_execution_doc()` method from DAG compiler
- ✅ Update compiler to focus only on pipeline compilation
- ✅ Clean up execution document related imports
- ✅ Update documentation with new two-step process

**Files Modified**:
- ✅ `src/cursus/core/compiler/dag_compiler.py`

**Acceptance Criteria**:
- ✅ Execution document methods removed from compiler
- ✅ Pipeline compilation functionality preserved
- ✅ Clear documentation of new workflow
- ✅ Migration guidance provided

**Status**: **COMPLETED** - Successfully removed `compile_and_fill_execution_doc()` method from DAG compiler. Users now follow a clean two-step process: compile pipeline, then generate execution document separately.

### 2.6 Integration Testing ⏳ **READY FOR IMPLEMENTATION**
**Timeline**: Week 7, Days 3-5

**Tasks**:
- Run full test suite to ensure no regressions
- Test all existing pipelines
- Validate pipeline generation still works
- Performance testing
- Documentation updates

**Acceptance Criteria**:
- All existing tests pass
- All existing pipelines generate successfully
- No performance regressions
- Documentation updated to reflect changes

**Status**: **READY** - All cleanup phases completed successfully. Ready for comprehensive integration testing to validate that pipeline generation continues to work correctly without execution document logic.

## Phase 3: Pipeline Catalog Integration

### Duration: 2-3 weeks

### 3.1 Pipeline Catalog Analysis
**Timeline**: Week 8, Days 1-2

**Tasks**:
- Analyze existing pipeline catalog structure
- Identify all pipelines requiring execution document generation
- Map pipelines to shared DAGs
- Plan integration strategy

**Analysis Scope**:
```
cursus/pipeline_catalog/
├── pipelines/
│   ├── pipeline_a/
│   ├── pipeline_b/
│   └── ...
└── shared_dags/
    ├── dag_1.py
    ├── dag_2.py
    └── ...
```

**Deliverables**:
- Pipeline catalog structure analysis
- Pipeline-to-DAG mapping
- Integration architecture plan
- Implementation timeline

### 3.2 Pipeline Execution Document Integration
**Timeline**: Week 8, Days 3-5

**Tasks**:
- Create simple `pipeline_exe` folder in pipeline catalog
- Add direct integration with standalone execution document generator
- Create simple utility functions for pipeline execution document generation
- Map existing pipelines to their configurations and shared DAGs

**Simplified Structure**:
```
cursus/pipeline_catalog/pipeline_exe/
├── __init__.py
├── generator.py              # Simple pipeline execution document generation
└── utils.py                  # Utility functions for pipeline mapping
```

**Acceptance Criteria**:
- Simple pipeline execution document module created
- Direct integration with standalone generator working
- Pipeline-to-config-to-DAG mapping functional
- Basic utility functions implemented

### 3.3 Pipeline Integration Implementation
**Timeline**: Week 9, Days 1-5

**Tasks**:
- Implement simple pipeline execution document generation functions
- Create direct mapping from pipeline names to configurations and DAGs
- Add simple wrapper functions for each pipeline
- Integrate with standalone execution document generator

**Simple Implementation Pattern**:
```python
def generate_execution_document_for_pipeline(pipeline_name: str) -> Dict[str, Any]:
    """Generate execution document for a specific pipeline."""
    config_path = get_config_path_for_pipeline(pipeline_name)
    dag = load_shared_dag_for_pipeline(pipeline_name)
    
    generator = ExecutionDocumentGenerator(config_path=config_path)
    execution_doc_template = create_execution_doc_template_for_pipeline(pipeline_name)
    
    return generator.fill_execution_document(dag, execution_doc_template)
```

**Acceptance Criteria**:
- Simple pipeline execution document generation working
- All pipelines can generate execution documents
- Direct integration with standalone generator functional
- No complex class hierarchies or over-engineering

### 3.4 Configuration Integration
**Timeline**: Week 10, Days 1-3

**Tasks**:
- Integrate pipeline configurations with execution document generation
- Ensure configuration compatibility
- Add configuration validation
- Update configuration loading utilities

**Configuration Flow**:
1. Pipeline handler loads pipeline-specific configuration
2. Configuration passed to standalone execution document generator
3. Generator processes configuration using helpers
4. Execution document generated and returned

**Acceptance Criteria**:
- Configuration integration working
- All pipeline configurations compatible
- Configuration validation in place
- No breaking changes to existing configurations

### 3.5 Testing and Validation
**Timeline**: Week 10, Days 4-5

**Tasks**:
- Create comprehensive test suite for pipeline execution document generation
- Test all pipelines in catalog
- Validate execution document format and content
- Performance testing
- Integration testing with existing systems

**Test Coverage**:
- All pipelines in catalog
- Various configuration scenarios
- Error conditions and edge cases
- Performance benchmarks
- Integration with downstream systems

**Acceptance Criteria**:
- All tests passing
- All pipelines generate valid execution documents
- Performance meets requirements
- Integration with downstream systems working

## Phase 4: Final Integration and Deployment

### Duration: 1-2 weeks

### 4.1 End-to-End Testing
**Timeline**: Week 11, Days 1-3

**Tasks**:
- Run complete end-to-end test suite
- Test all pipelines with new execution document generation
- Validate compatibility with downstream systems
- Performance testing under load
- Security and compliance validation

**Test Scenarios**:
- All pipeline catalog pipelines
- Various execution environments
- High-load scenarios
- Error recovery scenarios
- Security and access control

**Acceptance Criteria**:
- All end-to-end tests passing
- Performance meets or exceeds requirements
- Security and compliance validated
- No regressions in existing functionality

### 4.2 Documentation and Training
**Timeline**: Week 11, Days 4-5

**Tasks**:
- Update all documentation
- Create user guides for new system
- Update API documentation
- Create training materials
- Update operational procedures

**Documentation Updates**:
- Architecture documentation
- API reference documentation
- User guides and tutorials
- Operational runbooks
- Troubleshooting guides

**Acceptance Criteria**:
- All documentation updated
- User guides complete and tested
- Training materials ready
- Operational procedures updated

### 4.3 Production Deployment
**Timeline**: Week 12, Days 1-5

**Tasks**:
- Deploy standalone execution document generator
- Update pipeline catalog with new execution document generation
- Monitor system performance and stability
- Address any deployment issues
- Validate production functionality

**Deployment Steps**:
1. Deploy standalone execution document generator in separate environment
2. Update pipeline catalog with new execution document handlers
3. Enable new execution document generation for all pipelines
4. Monitor system health and performance
5. Validate all pipelines working correctly

**Acceptance Criteria**:
- Successful production deployment
- All pipelines generating execution documents correctly
- System performance stable
- No production issues
- Monitoring and alerting operational

## Risk Management

### High-Risk Items
1. **Package Dependency Issues**: Unsupported packages may cause deployment problems
   - **Mitigation**: Thorough testing in separate environment, fallback plans
2. **Configuration Compatibility**: Existing configurations may not work with new system
   - **Mitigation**: Comprehensive compatibility testing, migration tools
3. **Performance Degradation**: New system may be slower than existing system
   - **Mitigation**: Performance testing, optimization, benchmarking
4. **Integration Issues**: New system may not integrate properly with downstream systems
   - **Mitigation**: Integration testing, stakeholder validation

### Medium-Risk Items
1. **Test Coverage Gaps**: Some edge cases may not be covered
   - **Mitigation**: Comprehensive test planning, code review
2. **Documentation Gaps**: Users may not understand new system
   - **Mitigation**: Thorough documentation, user training
3. **Operational Complexity**: New system may be more complex to operate
   - **Mitigation**: Operational training, automation, monitoring

### Rollback Plan
1. **Phase 1 Rollback**: Remove standalone module, revert to existing system
2. **Phase 2 Rollback**: Restore execution document methods to existing system
3. **Phase 3 Rollback**: Remove pipeline catalog integration, use existing methods
4. **Phase 4 Rollback**: Revert to previous production configuration

## Success Metrics

### Technical Metrics
- **Performance**: Execution document generation time ≤ existing system
- **Reliability**: 99.9% success rate for execution document generation
- **Test Coverage**: ≥95% code coverage for new modules
- **Error Rate**: <1% error rate in production

### Business Metrics
- **Maintainability**: Reduced coupling between pipeline generation and execution documents
- **Extensibility**: Easy addition of new step types for execution documents
- **Operational Efficiency**: Reduced operational overhead for execution document issues
- **Developer Productivity**: Faster development of new execution document features

## Resource Requirements

### Development Team
- **Lead Developer**: 1 FTE for entire project
- **Backend Developers**: 2 FTE for Phases 1-3
- **Test Engineers**: 1 FTE for testing and validation
- **DevOps Engineer**: 0.5 FTE for deployment and infrastructure

### Infrastructure
- **Separate Environment**: For standalone execution document generator
- **Testing Environment**: For comprehensive testing
- **Monitoring Infrastructure**: For production monitoring
- **Documentation Platform**: For updated documentation

## Dependencies

### External Dependencies
- Access to `com.amazon.secureaisandboxproxyservice.models` packages
- Access to `secure_ai_sandbox_python_lib.utils.coral_utils` packages
- Separate deployment environment setup
- Stakeholder approval for architecture changes

### Internal Dependencies
- Existing configuration utilities
- Pipeline catalog structure
- Shared DAG implementations
- Test infrastructure

## Conclusion

This comprehensive project plan provides a structured approach to refactoring the execution document generation system. The phased approach ensures minimal risk while achieving the goals of separation of concerns, improved maintainability, and enhanced extensibility. The plan includes detailed timelines, acceptance criteria, risk management, and success metrics to ensure successful project delivery.

The project will result in a cleaner, more maintainable system with clear separation between pipeline generation and execution document generation, while preserving all existing functionality and improving the overall architecture of the system.
- Pipeline catalog structure
- Shared DAG implementations
- Test infrastructure

## Conclusion

This comprehensive project plan provides a structured approach to refactoring the execution document generation system. The phased approach ensures minimal risk while achieving the goals of separation of concerns, improved maintainability, and enhanced extensibility. The plan includes detailed timelines, acceptance criteria, risk management, and success metrics to ensure successful project delivery.

The project will result in a cleaner, more maintainable system with clear separation between pipeline generation and execution document generation, while preserving all existing functionality and improving the overall architecture of the system.
