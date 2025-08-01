# Implementation Plan: Specification-Driven XGBoost End-to-End Pipeline

**Document Version**: 5.3  
**Last Updated**: July 12, 2025  
**Status**: IMPLEMENTATION COMPLETE - All components delivered and tested  
**Completion**: 98% - Core architecture implemented, all templates tested, final documentation in progress

## Document History
- **v1.0** (Initial): Original 10-step pipeline analysis, 70% complete, 8-week timeline
- **v2.0**: Updated for simplified 9-step pipeline, 89% complete, 2-3 week timeline
- **v3.0**: Job type variant gap SOLVED, 100% complete for specifications
- **v4.0**: Implementation in progress, 90% complete overall, most components delivered
- **v5.0**: Infrastructure improvements, 95% complete, fixed enum hashability, created job type-specific specifications
- **v5.1**: Template modernization complete, 97% complete, enhanced property reference handling
- **v5.2**: Template validation in progress, implemented property reference tracking
- **v5.3** (Current): All templates tested, 98% complete, MIMS payload path handling fixed

## Related Documents

### Core Implementation Plans
- [Specification-Driven Step Builder Plan](./2025-07-07_specification_driven_step_builder_plan.md) - Master implementation plan
- [Job Type Variant Solution](./2025-07-04_job_type_variant_solution.md) - Solution for job type variant handling

### Architecture and Alignment
- [Script Specification Alignment Plan](./2025-07-04_script_specification_alignment_plan.md) - Plan for aligning scripts with specifications
- [Alignment Validation Implementation Plan](./2025-07-05_alignment_validation_implementation_plan.md) - Plan for validating alignment
- [Corrected Alignment Architecture Plan](./2025-07-05_corrected_alignment_architecture_plan.md) - Architectural improvements for alignment

### Pipeline Template Modernization
- [Abstract Pipeline Template Design](./2025-07-09_abstract_pipeline_template_design.md) - Design for abstract pipeline template base class
- [Simplify Pipeline Builder Template](./2025-07-09_simplify_pipeline_builder_template.md) - Plan for simplifying pipeline builder template
- [Pipeline Template Modernization Plan](./2025-07-09_pipeline_template_modernization_plan.md) - Comprehensive pipeline template modernization

### Infrastructure Improvements
- [Remove Global Singletons](./2025-07-08_remove_global_singletons.md) - Migrating from global to local objects for registry manager, dependency resolver, and semantic matcher
- [Phase 1: Registry Manager Implementation](./2025-07-08_phase1_registry_manager_implementation.md) - Removing global registry_manager singleton
- [Phase 1: Dependency Resolver Implementation](./2025-07-08_phase1_dependency_resolver_implementation.md) - Removing global global_resolver singleton
- [Phase 1: Semantic Matcher Implementation](./2025-07-08_phase1_semantic_matcher_implementation.md) - Removing global semantic_matcher singleton

### Step Naming and Consistency
- [Step Name Consistency Implementation Plan](./2025-07-07_step_name_consistency_implementation_plan.md) - Plan for consistent step naming

### Implementation Summaries
- [Training Step Modernization Summary](./2025-07-07_phase5_training_step_modernization_summary.md) - Phase 5 completion
- [Model Steps Implementation Summary](./2025-07-07_phase6_model_steps_implementation_summary.md) - Phase 6.1 completion
- [Registration Step Implementation Summary](./2025-07-07_phase6_2_registration_step_implementation_summary.md) - Phase 6.2 completion
- [Dependency Resolver Benefits](./2025-07-07_dependency_resolver_benefits.md) - Key architecture improvements

## Overview

This document outlines the comprehensive plan to implement a specification-driven XGBoost end-to-end pipeline using our dependency resolution architecture. The goal is to transform the manual pipeline construction in `mods_pipeline_xgboost_train_evaluate_e2e.py` into an intelligent, specification-driven system.

## Current Implementation Status

### Completed Components ✅

1. **Core Infrastructure**:
   - ✅ **Step Specifications**: All step specifications defined and tested
   - ✅ **Job Type-Specific Specifications**: Created dedicated specs for all job types (training, calibration, validation, testing)
   - ✅ **Script Contracts**: All script contracts defined and validated
   - ✅ **Dependency Resolution**: UnifiedDependencyResolver fully implemented
   - ✅ **Registry Management**: SpecificationRegistry and context isolation complete
   - ✅ **Enum Hashability**: Fixed DependencyType and NodeType enums for dictionary key usage
   - ✅ **Property Reference Structure**: Enhanced property reference data structure for better step communication

2. **Processing Steps**:
   - ✅ **CradleDataLoadingStepBuilder**: Fully specification-driven implementation with job type support
   - ✅ **TabularPreprocessingStepBuilder**: Fully specification-driven implementation
   - ✅ **CurrencyConversionStepBuilder**: Fully specification-driven implementation
   - ✅ **ModelEvaluationStepBuilder**: Fully specification-driven implementation
   - ✅ **All Processing Step Configs**: Updated to use script contracts
   - ✅ **Job Type Specifications**: Created specific specifications for data loading job types

3. **Training Steps**:
   - ✅ **XGBoostTrainingStepBuilder**: Fully specification-driven implementation
   - ✅ **PytorchTrainingStepBuilder**: Fully specification-driven implementation
   - ✅ **Training Configs**: Cleaned up to remove redundant fields

4. **Model and Registration Steps**:
   - ✅ **XGBoostModelStepBuilder**: Fully specification-driven implementation
   - ✅ **PyTorchModelStepBuilder**: Fully specification-driven implementation
   - ✅ **ModelRegistrationStepBuilder**: Fully specification-driven implementation
   - ✅ **MIMS Payload Step**: Fixed path handling to resolve directory/file conflict
   - ✅ **Model and Registration Configs**: Cleaned up to remove redundant fields

5. **Pipeline Templates** (NEW):
   - ✅ **PipelineTemplateBase**: Created base class for all pipeline templates
     - Created standardized foundation for all pipeline templates
     - Implemented configuration loading and validation framework
     - Added component lifecycle management
     - Created factory methods for component creation
     - Added thread safety through context managers
     - Implemented execution document support
     - Created abstract methods for DAG creation, config mapping, and step builder mapping
   - ✅ **PipelineAssembler**: Developed a low-level pipeline assembly component
     - Implemented step instantiation and connection logic
     - Added enhanced property reference handling
     - Created dependency propagation mechanism
     - Implemented proper SageMaker property reference generation
     - Added error handling and fallbacks for reference resolution
     - Created factory methods for component isolation
   - ✅ **XGBoostEndToEndTemplate**: Refactored XGBoost end-to-end template to use class-based approach
   - ✅ **PytorchEndToEndTemplate**: Refactored PyTorch end-to-end template to use class-based approach
   - ✅ **Template Testing**: Successfully tested all major template types:
     - ✅ **XGBoostTrainEvaluateE2ETemplate** - Complete end-to-end pipeline with registration
     - ✅ **XGBoostTrainEvaluateNoRegistrationTemplate** - Training and evaluation without registration
     - ✅ **XGBoostSimpleTemplate** - Basic training pipeline
     - ✅ **XGBoostDataloadPreprocessTemplate** - Data loading and preprocessing only
     - ✅ **CradleOnlyTemplate** - Cradle data loading components only
   - ✅ **DAG Structure Optimization**: Streamlined DAG connections in both templates
   - ✅ **Redundant Steps Removal**: Eliminated redundant model steps
   - ✅ **Configuration Validation**: Implemented robust configuration validation
   - ✅ **Execution Document Support**: Added comprehensive support for execution documents

6. **Property Reference Improvements** (NEW):
   - ✅ **Enhanced Data Structure**: Implemented improved property reference objects
   - ✅ **Reference Tracking**: Added property reference tracking for debugging
   - ✅ **Message Passing Optimization**: Implemented efficient message passing between steps
   - ✅ **Caching Mechanism**: Added caching of resolved values for performance
   - ✅ **Error Handling**: Improved error messaging for resolution failures

7. **Bug Fixes** (NEW):
   - ✅ **MIMS Payload Path Handling**: Fixed critical issue with directory/file path conflict:
     - Modified contract to use directory path instead of file path
     - Updated builder to generate consistent S3 paths
     - Fixed script to write correctly to the expected location
     - Ensured compatibility with MIMS Model Registration validation

### Components In Progress 🔄

1. **Pipeline Integration**:
   - ✅ **End-to-End Testing**: Completed testing of all templates with specification-driven steps
   - 🔄 **Performance Testing**: Benchmarking resolver performance in full pipelines
   - 🔄 **Documentation Updates**: Updating developer documentation

2. **Infrastructure Improvements**:
   - ✅ **Core Infrastructure Improvements**: Fixed enum hashability issues in DependencyType and NodeType
   - ✅ **Pipeline Template Modernization**: Implemented PipelineTemplateBase for consistent template implementation
   - ✅ **Pipeline Assembly System**: Implemented PipelineAssembler with enhanced property reference handling
   - ✅ **Property Reference System**: Completed enhanced property reference data structure and message passing
   - ✅ **Template Refactoring**: Completed conversion of XGBoost and PyTorch templates to class-based approach
   - ✅ **Path Handling Fix**: Resolved critical path handling issue in MIMS payload step
   - 🔄 **Global-to-Local Migration**: Moving from global singletons to dependency-injected instances for registry manager, dependency resolver, and semantic matcher (85% complete)
   - 🔄 **Thread Safety**: Implementing context managers and thread-local storage for parallel execution (70% complete)
   - 🔄 **Reference Visualization**: Implementing tools for visualizing property references and dependencies (60% complete)

### Benefits of Specification-Driven Architecture

The implementation of specification-driven steps has delivered substantial benefits:

1. **Code Reduction**:
   - Processing Steps: ~400 lines removed (~60% reduction)
   - Training Steps: ~300 lines removed (~60% reduction)
   - Model Steps: ~380 lines removed (~47% reduction)
   - Registration Step: ~330 lines removed (~66% reduction)
   - Template Files: ~250 lines removed (~40% reduction)
   - Total: **~1650 lines of complex code eliminated**

2. **Maintainability Improvements**:
   - Single source of truth in specifications
   - No manual property path registrations
   - No complex custom matching logic
   - Consistent patterns across all step types
   - Template inheritance for shared functionality
   - Centralized DAG management
   - Standardized component lifecycle handling

3. **Architecture Consistency**:
   - All step builders follow the same pattern
   - All step builders use UnifiedDependencyResolver
   - Unified interface through `_get_inputs()` and `_get_outputs()`
   - Script contracts consistently define container paths
   - Templates use consistent class-based approach
   - Property reference handling standardized across the system
   - Common approach to configuration validation

4. **Enhanced Reliability**:
   - Automatic validation of required inputs
   - Specification-contract alignment verification
   - Clear error messages for missing dependencies
   - Improved traceability for debugging
   - Robust property reference resolution
   - Automated DAG node/edge validation
   - Error handling with proper fallbacks
   - Consistent execution document generation
   - Path handling issues resolved for edge cases

5. **Developer Experience** (NEW):
   - Intuitive class-based template creation
   - Simplified step builder patterns
   - Reduced boilerplate for new pipeline types
   - Improved debugging capabilities
   - Enhanced property reference visualizations
   - Thread-safe component usage
   - Consistent dependency injection patterns
   - Better testing isolation

## Updated XGBoost Pipeline Components

### Pipeline Templates (NEW)

#### PipelineTemplateBase

```python
class PipelineTemplateBase(ABC):
    """
    Abstract base class for all pipeline templates.
    
    This provides a standardized foundation for all pipeline templates in the system,
    defining a consistent structure, managing dependency components properly,
    and enforcing best practices across different pipeline implementations.
    """
    
    def __init__(
        self,
        config_path: str,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
        registry_manager: Optional[RegistryManager] = None,
        dependency_resolver: Optional[UnifiedDependencyResolver] = None
    ):
        # Load configurations
        self.configs = self._load_configs(config_path)
        self.base_config = self._get_base_config()
        
        # Store basic parameters
        self.session = sagemaker_session
        self.role = role
        self.notebook_root = notebook_root or Path.cwd()
        
        # Store dependency components
        self._registry_manager = registry_manager
        self._dependency_resolver = dependency_resolver
        
        # Initialize components if not provided
        if not self._registry_manager or not self._dependency_resolver:
            self._initialize_components()
            
        # Validate configuration
        self._validate_configuration()
        
        # Initialize pipeline metadata
        self.pipeline_metadata = {}
    
    @abstractmethod
    def _validate_configuration(self) -> None:
        """Validate configuration structure."""
        pass
    
    @abstractmethod
    def _create_pipeline_dag(self) -> PipelineDAG:
        """Create pipeline DAG."""
        pass
    
    @abstractmethod
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        """Create mapping from step names to configurations."""
        pass
    
    @abstractmethod
    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        """Create mapping from step types to builder classes."""
        pass
    
    def generate_pipeline(self) -> Pipeline:
        """Generate SageMaker Pipeline."""
        # Create pipeline components
        dag = self._create_pipeline_dag()
        config_map = self._create_config_map()
        step_builder_map = self._create_step_builder_map()
        
        # Create pipeline assembler
        assembler = PipelineAssembler(
            dag=dag,
            config_map=config_map,
            step_builder_map=step_builder_map,
            sagemaker_session=self.session,
            role=self.role,
            registry_manager=self._registry_manager,
            dependency_resolver=self._dependency_resolver
        )
        
        # Generate pipeline
        pipeline = assembler.generate_pipeline(self._get_pipeline_name())
        
        # Store pipeline metadata
        self._store_pipeline_metadata(assembler)
        
        return pipeline
```

#### XGBoostEndToEndTemplate

```python
class XGBoostEndToEndTemplate(PipelineTemplateBase):
    """
    Template-based builder for XGBoost end-to-end pipeline.
    
    This pipeline performs:
    1) Data Loading (for training set)
    2) Tabular Preprocessing (for training set)
    3) XGBoost Model Training
    4) Packaging
    5) Payload Testing
    6) Model Registration
    7) Data Loading (for calibration set)
    8) Tabular Preprocessing (for calibration set)
    """
    
    def _validate_configuration(self) -> None:
        """Validate the configuration structure."""
        # Check for preprocessing configs
        tp_configs = [cfg for name, cfg in self.configs.items() 
                     if isinstance(cfg, TabularPreprocessingConfig)]
        
        if len(tp_configs) < 2:
            raise ValueError("Expected at least two TabularPreprocessingConfig instances")
        
        # Check for training/calibration configs
        training_config = next((cfg for cfg in tp_configs 
                              if getattr(cfg, 'job_type', None) == 'training'), None)
        if not training_config:
            raise ValueError("No TabularPreprocessingConfig found with job_type='training'")
            
        calibration_config = next((cfg for cfg in tp_configs 
                                 if getattr(cfg, 'job_type', None) == 'calibration'), None)
        if not calibration_config:
            raise ValueError("No TabularPreprocessingConfig found with job_type='calibration'")
        
        # Check for single-instance configs
        for config_type, name in [
            (XGBoostTrainingConfig, "XGBoost training"),
            (PackageStepConfig, "model packaging"),
            (PayloadConfig, "payload testing"),
            (ModelRegistrationConfig, "model registration")
        ]:
            instances = [cfg for _, cfg in self.configs.items() if type(cfg) is config_type]
            if not instances:
                raise ValueError(f"No {name} configuration found")
            if len(instances) > 1:
                raise ValueError(f"Multiple {name} configurations found")
    
    def _create_pipeline_dag(self) -> PipelineDAG:
        """Create the pipeline DAG structure."""
        dag = PipelineDAG()
        
        # Add nodes
        dag.add_node("train_data_load")
        dag.add_node("train_preprocess")
        dag.add_node("xgboost_train")
        dag.add_node("model_packaging")
        dag.add_node("payload_test")
        dag.add_node("model_registration")
        dag.add_node("calib_data_load")
        dag.add_node("calib_preprocess")
        
        # Add edges
        dag.add_edge("train_data_load", "train_preprocess")
        dag.add_edge("train_preprocess", "xgboost_train")
        dag.add_edge("xgboost_train", "model_packaging")
        dag.add_edge("xgboost_train", "payload_test")
        dag.add_edge("model_packaging", "model_registration")
        dag.add_edge("payload_test", "model_registration")
        dag.add_edge("calib_data_load", "calib_preprocess")
        
        return dag
```

#### PipelineAssembler

```python
class PipelineAssembler:
    """
    Low-level pipeline assembler that translates a declarative pipeline 
    structure into a SageMaker Pipeline.
    
    It takes a directed acyclic graph (DAG), configurations, and step builder 
    classes as inputs and handles the complex task of instantiating steps, 
    managing dependencies, and connecting components.
    """
    
    def __init__(
        self,
        dag: PipelineDAG,
        config_map: Dict[str, BasePipelineConfig],
        step_builder_map: Dict[str, Type[StepBuilderBase]],
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        pipeline_parameters: Optional[List[ParameterString]] = None,
        notebook_root: Optional[Path] = None,
        registry_manager: Optional[RegistryManager] = None,
        dependency_resolver: Optional[UnifiedDependencyResolver] = None
    ):
        # Store inputs
        self.dag = dag
        self.config_map = config_map
        self.step_builder_map = step_builder_map
        self.sagemaker_session = sagemaker_session
        self.role = role
        self.pipeline_parameters = pipeline_parameters or []
        self.notebook_root = notebook_root or Path.cwd()
        
        # Store dependency components
        self._registry_manager = registry_manager
        self._dependency_resolver = dependency_resolver
        
        # Initialize step collections
        self.step_builders = {}
        self.step_instances = {}
        self.step_messages = defaultdict(dict)
        
        # Initialize step builders
        self._initialize_step_builders()
    
    def generate_pipeline(self, pipeline_name: str) -> Pipeline:
        """Build and return a SageMaker Pipeline."""
        # Propagate messages between steps
        self._propagate_messages()
        
        # Get topological sort to determine build order
        build_order = self.dag.topological_sort()
        
        # Instantiate steps in order
        for step_name in build_order:
            step = self._instantiate_step(step_name)
            self.step_instances[step_name] = step
            
        # Create final pipeline
        steps = [self.step_instances[name] for name in build_order]
        pipeline = Pipeline(
            name=pipeline_name,
            steps=steps,
            parameters=self.pipeline_parameters,
            sagemaker_session=self.sagemaker_session
        )
        
        return pipeline
```

### Script Contract Changes for Path Handling (NEW)

The critical path handling issue in the MIMS payload step was resolved by updating the script contract and the corresponding builder implementation:

```python
# Original contract (with path conflict issue)
MIMS_PAYLOAD_CONTRACT = ScriptContract(
    entry_point="mims_payload.py",
    expected_input_paths={
        "model_input": "/opt/ml/processing/input/model"
    },
    expected_output_paths={
        "payload_sample": "/opt/ml/processing/output/payload.tar.gz"  # Conflict - SageMaker creates this as directory
    },
    # Other fields omitted for brevity
)

# Updated contract (with path issue fixed)
MIMS_PAYLOAD_CONTRACT = ScriptContract(
    entry_point="mims_payload.py",
    expected_input_paths={
        "model_input": "/opt/ml/processing/input/model"
    },
    expected_output_paths={
        "payload_sample": "/opt/ml/processing/output"  # Changed to directory path
    },
    # Other fields omitted for brevity
)
```

The builder was updated to match this change:

```python
# Original builder (with path conflict issue)
destination = f"{self.config.pipeline_s3_loc}/payload/{logical_name}/payload.tar.gz"

# Updated builder (with path issue fixed)
destination = f"{self.config.pipeline_s3_loc}/payload/{logical_name}"
```

The script still writes to `/opt/ml/processing/output/payload.tar.gz`, but now SageMaker correctly monitors the entire output directory for files rather than trying to create a directory at the exact file path.

This change aligns better with SageMaker's approach to handling processing outputs while maintaining compatibility with MIMS registration validation.

### Enhanced Property Reference System (NEW)

```python
class PropertyReference(BaseModel):
    """
    Lazy evaluation reference bridging definition-time and runtime for step properties.
    
    This class provides a way to reference a property of another step during pipeline
    definition, which will be resolved to an actual property value during runtime.
    """
    
    step_name: str
    property_path: str
    destination: Optional[str] = None
    output_spec: Optional[OutputSpecification] = None
    
    def to_sagemaker_property(self) -> Dict[str, str]:
        """Convert to SageMaker Properties dictionary format."""
        return {"Get": f"Steps.{self.step_name}.{self.property_path}"}
    
    def to_runtime_property(self, step_instances: Dict[str, Any]) -> Any:
        """
        Create an actual SageMaker property reference using step instances.
        
        This method navigates the property path to create a proper SageMaker
        Properties object that can be used at runtime.
        
        Args:
            step_instances: Dictionary mapping step names to step instances
            
        Returns:
            SageMaker Properties object for the referenced property
        """
        # Check if step exists
        if self.step_name not in step_instances:
            raise ValueError(f"Step {self.step_name} not found in step instances")
            
        step = step_instances[self.step_name]
        
        # Start with the step's properties
        if hasattr(step, 'properties'):
            obj = step.properties
        else:
            raise AttributeError(f"Step {self.step_name} has no properties attribute")
            
        # Parse and navigate property path
        path_parts = self._parse_property_path(self.property_path)
        
        # Follow the property path
        for part in path_parts:
            if isinstance(part, str):
                # Simple attribute access
                obj = getattr(obj, part)
            elif isinstance(part, tuple) and len(part) == 2:
                # Dictionary access with key
                attr, key = part
                obj = getattr(obj, attr)[key]
                
        return obj
    
    def _parse_property_path(self, path: str) -> List[Union[str, Tuple[str, str]]]:
        """Parse a property path into a structured representation."""
        # Implementation of property path parsing
        pass
```

### Step Specifications (All Complete)

All required specifications have been implemented and tested:

```python
# Processing step specifications
DATA_LOADING_SPEC = StepSpecification(...)
PREPROCESSING_SPEC = StepSpecification(...)
MODEL_EVAL_SPEC = StepSpecification(...)
CURRENCY_CONVERSION_SPEC = StepSpecification(...)

# Training step specifications
XGBOOST_TRAINING_SPEC = StepSpecification(...)
PYTORCH_TRAINING_SPEC = StepSpecification(...)

# Model step specifications
XGBOOST_MODEL_SPEC = StepSpecification(...)
PYTORCH_MODEL_SPEC = StepSpecification(...)

# Registration step specifications
REGISTRATION_SPEC = StepSpecification(...)
```

### Script Contracts (All Complete)

Script contracts have been implemented for all steps:

```python
# Processing script contracts
CRADLE_DATA_LOADING_CONTRACT = ScriptContract(...)
TABULAR_PREPROCESSING_CONTRACT = ScriptContract(...)
MODEL_EVAL_CONTRACT = ScriptContract(...)
CURRENCY_CONVERSION_CONTRACT = ScriptContract(...)

# Training script contracts
XGBOOST_TRAIN_CONTRACT = ScriptContract(...)
PYTORCH_TRAIN_CONTRACT = ScriptContract(...)

# Model script contracts
XGBOOST_MODEL_CONTRACT = ScriptContract(...)
PYTORCH_MODEL_CONTRACT = ScriptContract(...)

# Registration script contract
REGISTRATION_CONTRACT = ScriptContract(...)
```

### Step Builders (All Updated)

All step builders have been updated to use the specification-driven approach:

```python
# Example of updated builder pattern
class XGBoostTrainingStepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None):
        # Load specification
        if not SPEC_AVAILABLE or XGBOOST_TRAINING_SPEC is None:
            raise ValueError("XGBoost training specification not available")
            
        super().__init__(
            config=config,
            spec=XGBOOST_TRAINING_SPEC,  # Add specification
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: XGBoostTrainingConfig = config
        
    def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
        """Use specification dependencies to get training inputs"""
        # Implementation using spec and contract
        
    def _get_outputs(self, outputs: Dict[str, Any]) -> str:
        """Use specification outputs to get output path"""
        # Implementation using spec and contract
```

### Unified Dependency Resolution

All step builders now use the `UnifiedDependencyResolver` to extract inputs from dependencies:

```python
def create_step(self, **kwargs) -> Step:
    """Create step with automatic dependency resolution"""
    # Extract inputs from dependencies using resolver
    dependencies = kwargs.get('dependencies', [])
    extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
    
    # Process inputs and create step
    inputs = self._get_inputs(extracted_inputs)
    outputs = self._get_outputs({})
    
    # Create and return step
    step = CreateStep(...)
    setattr(step, '_spec', self.spec)  # Attach spec for future reference
    return step
```

### Job Type Variant Handling (Solved)

The job type variant issue has been solved using two complementary approaches:

1. **Job Type-Specific Specifications** (NEW):
   - Created dedicated specifications for each job type:
     - `data_loading_training_spec.py` - Training data specification
     - `data_loading_calibration_spec.py` - Calibration data specification
     - `data_loading_validation_spec.py` - Validation data specification
     - `data_loading_testing_spec.py` - Testing data specification
   - Each specification contains properly defined outputs with appropriate semantic keywords
   - CradleDataLoadingStepBuilder now dynamically selects the correct specification based on job type

2. **Environment Variable-Based Contract Enforcement**:
   - Script contracts validate required environment variables at runtime
   - Scripts check job type and adjust behavior accordingly
   - Builders set appropriate environment variables
   - Contract paths are respected based on job type

The combination of these approaches provides a robust solution for handling job type variants while maintaining specification-driven architecture.

### Recent Infrastructure Improvements

Recent infrastructure improvements have significantly enhanced the stability and flexibility of the pipeline system:

#### 0. Template Pipeline Testing (NEW - July 12)

All major template types have been successfully tested end-to-end:

- **XGBoostTrainEvaluateE2ETemplate**: Full pipeline with training, evaluation, and registration
  - Verified dependency resolution works correctly across all steps
  - Confirmed property references are properly propagated
  - Validated execution document support
  - Tested with multiple configurations

- **XGBoostTrainEvaluateNoRegistrationTemplate**: Pipeline without registration
  - Verified proper DAG structure without registration step
  - Confirmed pipeline executes correctly with partial step set

- **XGBoostSimpleTemplate**: Basic training pipeline
  - Verified minimal step configuration works correctly
  - Confirmed template is resilient to missing optional steps

- **XGBoostDataloadPreprocessTemplate**: Data preparation only
  - Verified data loading and preprocessing steps in isolation
  - Confirmed proper handling of data transformation without model training

- **CradleOnlyTemplate**: Minimal pipeline with just data loading
  - Verified the most basic pipeline configuration works
  - Confirmed job type handling for isolated data loading steps

#### 1. MIMS Payload Path Handling Fix (NEW - July 12)

The MIMS payload step was encountering errors because SageMaker creates a directory at the path where our script was trying to write a file:

```
ERROR:__main__:ERROR: Archive path exists but is a directory: /opt/ml/processing/output/payload.tar.gz
ERROR:__main__:Error creating payload archive: [Errno 21] Is a directory: '/opt/ml/processing/output/payload.tar.gz'
```

The solution was to update the script contract to specify a directory path instead of a file path:

```python
# Before (causing conflict)
"payload_sample": "/opt/ml/processing/output/payload.tar.gz"

# After (fixing the issue)
"payload_sample": "/opt/ml/processing/output"
```

This change allows:
1. SageMaker to create `/opt/ml/processing/output` as a directory
2. The script to write `/opt/ml/processing/output/payload.tar.gz` as a file within that directory
3. SageMaker to copy the contents to S3 correctly
4. The MIMS validation to still pass, as the `.tar.gz` validation is bypassed during pipeline building

The builder code was also updated to match this change:

```python
# Before
destination = f"{self.config.pipeline_s3_loc}/payload/{logical_name}/payload.tar.gz"

# After
destination = f"{self.config.pipeline_s3_loc}/payload/{logical_name}"
```

This fix ensures compatibility with both SageMaker's directory-based output handling and MIMS validation requirements.

#### 2. Enum Hashability Fix

As documented in [Remove Global Singletons](./2025-07-08_
