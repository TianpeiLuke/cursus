---
tags:
  - design
  - reference
  - execution_document
  - standalone_module
  - cradle_data_loading
  - mims_registration
keywords:
  - execution document generator
  - standalone module
  - PipelineDAG
  - cradle data loading
  - MIMS registration
  - independent system
  - separate environment
topics:
  - execution document generation
  - standalone architecture
  - modular design
  - system isolation
language: python
date of note: 2025-09-16
---

# Standalone Execution Document Generator Design

## Overview

This document outlines the design for a standalone execution document generator module that operates independently from the pipeline generation system. The module takes a PipelineDAG as input and generates execution documents by collecting configurations from Cradle data loading and MIMS registration steps.

## Reference

This design is based on the comprehensive analysis documented in:
- [Execution Document Filling Analysis](../4_analysis/execution_document_filling_analysis.md)

## Design Goals

### Primary Objectives
1. **Complete Independence**: Standalone module that doesn't depend on the pipeline generation system (dag_compiler, assembler, template layers)
2. **Simplified Input**: Takes only PipelineDAG and configuration data as input
3. **Environment Isolation**: Can use unsupported packages (`secure_ai_sandbox_python_lib`, `com.amazon.secureaisandboxproxyservice`) since it runs in a separate environment
4. **Focused Functionality**: Single responsibility - execution document generation only
5. **Maintainable Architecture**: Clean, testable, and extensible design

### Secondary Objectives
- Preserve existing execution document format and structure
- Support both Cradle data loading and MIMS registration steps
- Provide clear error handling and logging
- Enable easy testing and validation

## Architecture Design

### Module Structure
```
src/cursus/mods/exe_doc/
├── __init__.py
├── generator.py              # Main ExecutionDocumentGenerator class
├── base.py                   # Base classes and interfaces
├── cradle_helper.py          # Cradle data loading helper
├── registration_helper.py    # MIMS registration helper
└── utils.py                  # Utility functions
```

### Core Components

#### 1. ExecutionDocumentGenerator (Main Class)
**File**: `generator.py`

```python
class ExecutionDocumentGenerator:
    """
    Standalone execution document generator.
    
    Takes a PipelineDAG and configuration data as input, generates execution
    documents by collecting and processing step configurations independently
    from the pipeline generation system.
    """
    
    def __init__(self, 
                 config_path: str,
                 sagemaker_session: Optional[PipelineSession] = None,
                 role: Optional[str] = None,
                 config_resolver: Optional[StepConfigResolver] = None):
        """
        Initialize execution document generator.
        
        Args:
            config_path: Path to configuration file
            sagemaker_session: SageMaker session for AWS operations
            role: IAM role for AWS operations
            config_resolver: Custom config resolver for step name resolution
        """
        self.config_path = config_path
        self.sagemaker_session = sagemaker_session
        self.role = role
        self.config_resolver = config_resolver or StepConfigResolver()
        self.configs = self._load_configs()
        self.helpers = [
            CradleDataLoadingHelper(),
            RegistrationHelper(),
        ]
    
    def fill_execution_document(self, 
                              dag: PipelineDAG, 
                              execution_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for filling execution documents.
        
        Args:
            dag: PipelineDAG defining the pipeline structure
            execution_document: Template execution document to fill
            
        Returns:
            Filled execution document
        """
        # 1. Identify relevant steps in the DAG
        relevant_steps = self._identify_relevant_steps(dag)
        
        # 2. Collect configurations for relevant steps
        step_configs = self._collect_step_configurations(relevant_steps)
        
        # 3. Fill execution document
        return self._fill_document(execution_document, step_configs)
```

#### 2. Step Helpers (Processing Logic)

**Base Helper Interface**:
```python
class ExecutionDocumentHelper(ABC):
    """Base class for execution document helpers."""
    
    @abstractmethod
    def can_handle_step(self, step_name: str, config: Any) -> bool:
        """Check if this helper can handle the given step."""
        pass
    
    @abstractmethod
    def extract_step_config(self, step_name: str, config: Any) -> Dict[str, Any]:
        """Extract step configuration for execution document."""
        pass
```

**Cradle Helper** (`cradle_helper.py`):
```python
class CradleDataLoadingHelper(ExecutionDocumentHelper):
    """Helper for Cradle data loading execution document configuration."""
    
    def can_handle_step(self, step_name: str, config: Any) -> bool:
        return isinstance(config, CradleDataLoadConfig)
    
    def extract_step_config(self, step_name: str, config: CradleDataLoadConfig) -> Dict[str, Any]:
        """Extract Cradle configuration using existing methods."""
        # Can use com.amazon.secureaisandboxproxyservice and secure_ai_sandbox_python_lib
        # since this runs in a separate environment
        request = self._build_request(config)
        return self._get_request_dict(request)
    
    def _build_request(self, config: CradleDataLoadConfig) -> CreateCradleDataLoadJobRequest:
        """
        Build Cradle request using original model classes.
        
        Based on existing CradleDataLoadingStepBuilder._build_request() method.
        Uses com.amazon.secureaisandboxproxyservice.models directly.
        """
        # Implementation based on existing _build_request method:
        # 1. Create DataSourcesSpecification from config data sources
        # 2. Create TransformSpecification from config transforms
        # 3. Create OutputSpecification from config outputs
        # 4. Create CradleJobSpecification from config job settings
        # 5. Combine into CreateCradleDataLoadJobRequest
        pass
    
    def _get_request_dict(self, request: CreateCradleDataLoadJobRequest) -> Dict[str, Any]:
        """
        Convert request to dictionary using coral_utils.
        
        Based on existing CradleDataLoadingStepBuilder.get_request_dict() method.
        Uses secure_ai_sandbox_python_lib.utils.coral_utils directly.
        """
        from secure_ai_sandbox_python_lib.utils import coral_utils
        return coral_utils.to_dict(request)
```

**Registration Helper** (`registration_helper.py`):
```python
class RegistrationHelper(ExecutionDocumentHelper):
    """Helper for MIMS registration execution document configuration."""
    
    def can_handle_step(self, step_name: str, config: Any) -> bool:
        return isinstance(config, RegistrationConfig)
    
    def extract_step_config(self, step_name: str, config: RegistrationConfig) -> Dict[str, Any]:
        """Extract registration configuration using existing methods."""
        # Use supported packages (secure_ai_sandbox_workflow_python_sdk, mods_workflow_core)
        return self._create_execution_doc_config(config)
    
    def _create_execution_doc_config(self, config: RegistrationConfig) -> Dict[str, Any]:
        """
        Create execution document configuration for registration step.
        
        Based on existing DynamicPipelineTemplate._fill_registration_configurations() method.
        Uses supported packages (secure_ai_sandbox_workflow_python_sdk, mods_workflow_core).
        """
        # Implementation based on existing _fill_registration_configurations method:
        # 1. Find registration configuration and related configs (payload, package)
        # 2. Get image URI using sagemaker.image_uris.retrieve
        # 3. Create execution document config with required fields:
        #    - source_model_inference_image_arn
        #    - model_domain, model_objective
        #    - source_model_inference_content_types, response_types
        #    - source_model_inference_input/output_variable_list
        #    - model_registration_region, source_model_region
        #    - source_model_environment_variable_map
        #    - load_testing_info_map (if payload and package configs available)
        
        exec_config = {
            "source_model_inference_image_arn": self._get_image_uri(config),
        }
        
        # Add registration configuration fields
        for field in [
            "model_domain", "model_objective",
            "source_model_inference_content_types",
            "source_model_inference_response_types", 
            "source_model_inference_input_variable_list",
            "source_model_inference_output_variable_list",
            "model_registration_region", "source_model_region",
            "aws_region", "model_owner", "region"
        ]:
            if hasattr(config, field):
                exec_config[field] = getattr(config, field)
        
        # Add environment variables if entry point is available
        if hasattr(config, "inference_entry_point"):
            exec_config["source_model_environment_variable_map"] = {
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_PROGRAM": config.inference_entry_point,
                "SAGEMAKER_REGION": getattr(config, "aws_region", "us-east-1"),
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            }
        
        return exec_config
    
    def _get_image_uri(self, config: RegistrationConfig) -> str:
        """Get image URI using SageMaker image retrieval."""
        try:
            from sagemaker.image_uris import retrieve
            
            if all(hasattr(config, attr) for attr in [
                "framework", "aws_region", "framework_version", 
                "py_version", "inference_instance_type"
            ]):
                return retrieve(
                    framework=config.framework,
                    region=config.aws_region,
                    version=config.framework_version,
                    py_version=config.py_version,
                    instance_type=config.inference_instance_type,
                    image_scope="inference",
                )
        except Exception as e:
            self.logger.warning(f"Could not retrieve image URI: {e}")
        
        return "image-uri-placeholder"
```

#### 3. Configuration Management (Integrated)
**Note**: Configuration management is integrated directly into the `ExecutionDocumentGenerator` class rather than using a separate `ConfigurationCollector`. This simplifies the architecture and reduces unnecessary abstraction.

**Configuration Loading in Main Class**:
```python
class ExecutionDocumentGenerator:
    def _load_configs(self) -> Dict[str, BasePipelineConfig]:
        """Load configurations using existing utilities."""
        from ...steps.configs.utils import load_configs, build_complete_config_classes
        
        # Use existing configuration loading infrastructure
        complete_classes = build_complete_config_classes()
        return load_configs(self.config_path, complete_classes)
    
    def _get_config_for_step(self, step_name: str) -> Optional[BasePipelineConfig]:
        """Get configuration for a specific step using config resolver."""
        # Use the config_resolver to map step names to configurations
        return self.config_resolver.resolve_config_for_step(step_name, self.configs)
```

**When Configuration Collection Happens**:
1. **At Initialization**: All configurations are loaded from the config file
2. **During Step Identification**: Configurations are retrieved for each DAG step
3. **During Configuration Collection**: Specific configs are passed to helpers for processing

**No Separate ConfigurationCollector Needed** because:
- Configuration loading is straightforward and doesn't require complex logic
- The main generator class can handle config management directly
- The config_resolver parameter provides customization when needed
- Reduces unnecessary abstraction and complexity

## Detailed Design

### Input Interface

#### PipelineDAG Input
```python
def fill_execution_document(self, 
                          dag: PipelineDAG, 
                          execution_document: Dict[str, Any]) -> Dict[str, Any]:
```

**Input Requirements**:
- `dag`: PipelineDAG instance containing step names and relationships
- `execution_document`: Template execution document with `PIPELINE_STEP_CONFIGS` structure

#### Configuration Input
- Configuration file path provided during initialization
- Supports same configuration format as existing system
- Loads all step configurations at initialization

### Processing Flow

#### Step 1: Step Identification
```python
def _identify_relevant_steps(self, dag: PipelineDAG) -> List[str]:
    """
    Identify steps in the DAG that require execution document processing.
    
    Returns:
        List of step names that need execution document configuration
    """
    relevant_steps = []
    
    for step_name in dag.nodes:
        config = self._get_config_for_step(step_name)
        if config and self._is_execution_doc_relevant(config):
            relevant_steps.append(step_name)
    
    return relevant_steps

def _is_execution_doc_relevant(self, config: BasePipelineConfig) -> bool:
    """Check if a configuration requires execution document processing."""
    return (isinstance(config, CradleDataLoadConfig) or 
            isinstance(config, RegistrationConfig))
```

#### Step 2: Configuration Collection
```python
def _collect_step_configurations(self, step_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Collect execution document configurations for relevant steps.
    
    Returns:
        Dictionary mapping step names to their execution document configurations
    """
    step_configs = {}
    
    for step_name in step_names:
        config = self._get_config_for_step(step_name)
        if config:
            helper = self._find_helper_for_config(config)
            if helper:
                step_configs[step_name] = helper.extract_step_config(step_name, config)
    
    return step_configs
```

#### Step 3: Document Filling
```python
def _fill_document(self, 
                  execution_document: Dict[str, Any], 
                  step_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fill execution document with collected step configurations.
    
    Args:
        execution_document: Template execution document
        step_configs: Collected step configurations
        
    Returns:
        Filled execution document
    """
    if "PIPELINE_STEP_CONFIGS" not in execution_document:
        execution_document["PIPELINE_STEP_CONFIGS"] = {}
    
    pipeline_configs = execution_document["PIPELINE_STEP_CONFIGS"]
    
    for step_name, step_config in step_configs.items():
        if step_name not in pipeline_configs:
            pipeline_configs[step_name] = {}
        
        pipeline_configs[step_name]["STEP_CONFIG"] = step_config
        
        # Add STEP_TYPE if not present
        if "STEP_TYPE" not in pipeline_configs[step_name]:
            pipeline_configs[step_name]["STEP_TYPE"] = self._determine_step_type(step_name)
    
    return execution_document
```

### Package Dependencies

#### Allowed Packages (Separate Environment)
Since this module runs in a separate environment, it can use:

**Previously Unsupported Packages** (now allowed):
- `com.amazon.secureaisandboxproxyservice.models`: All model classes for Cradle requests
- `secure_ai_sandbox_python_lib.utils.coral_utils`: For request-to-dictionary conversion

**Supported Packages**:
- `secure_ai_sandbox_workflow_python_sdk`: For MIMS registration functionality
- `mods_workflow_core`: For core utilities and constants

**Core Dependencies**:
- Standard Python libraries
- Existing cursus configuration utilities
- PipelineDAG classes

### Configuration Management

#### Step Name Resolution
```python
def _resolve_step_name_to_config(self, step_name: str) -> Optional[BasePipelineConfig]:
    """
    Resolve step name to configuration using intelligent matching.
    
    Uses similar logic to existing config resolver but simplified for
    execution document generation only.
    """
    # Direct name match
    if step_name in self.configs:
        return self.configs[step_name]
    
    # Pattern matching for common naming conventions
    for config_name, config in self.configs.items():
        if self._names_match(step_name, config_name):
            return config
    
    return None
```

#### Configuration Loading
```python
def _load_configs(self) -> Dict[str, BasePipelineConfig]:
    """Load configurations using existing utilities."""
    from ...steps.configs.utils import load_configs, build_complete_config_classes
    
    # Use existing configuration loading infrastructure
    complete_classes = build_complete_config_classes()
    return load_configs(self.config_path, complete_classes)
```

## Implementation Details

### Error Handling
```python
class ExecutionDocumentGenerationError(Exception):
    """Base exception for execution document generation errors."""
    pass

class ConfigurationNotFoundError(ExecutionDocumentGenerationError):
    """Raised when configuration cannot be found for a step."""
    pass

class UnsupportedStepTypeError(ExecutionDocumentGenerationError):
    """Raised when step type is not supported for execution document generation."""
    pass
```

### Logging Strategy
```python
import logging

class ExecutionDocumentGenerator:
    def __init__(self, config_path: str):
        self.logger = logging.getLogger(__name__)
        # ... rest of initialization
    
    def fill_execution_document(self, dag: PipelineDAG, execution_document: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Starting execution document generation for DAG with {len(dag.nodes)} nodes")
        
        try:
            # ... processing logic
            self.logger.info(f"Successfully generated execution document for {len(step_configs)} steps")
            return filled_document
        except Exception as e:
            self.logger.error(f"Failed to generate execution document: {e}")
            raise
```

### Testing Strategy

#### Unit Tests
```python
class TestExecutionDocumentGenerator:
    def test_cradle_step_processing(self):
        """Test Cradle data loading step processing."""
        generator = ExecutionDocumentGenerator("test_config.json")
        dag = create_test_dag_with_cradle_step()
        execution_doc = {"PIPELINE_STEP_CONFIGS": {}}
        
        result = generator.fill_execution_document(dag, execution_doc)
        
        assert "cradle_step" in result["PIPELINE_STEP_CONFIGS"]
        assert "STEP_CONFIG" in result["PIPELINE_STEP_CONFIGS"]["cradle_step"]
    
    def test_registration_step_processing(self):
        """Test MIMS registration step processing."""
        # Similar test for registration steps
        pass
    
    def test_mixed_pipeline(self):
        """Test pipeline with both Cradle and registration steps."""
        # Test complete pipeline processing
        pass
```

#### Integration Tests
```python
class TestExecutionDocumentGeneratorIntegration:
    def test_with_real_config_file(self):
        """Test with actual configuration files."""
        pass
    
    def test_output_format_compatibility(self):
        """Test that output format matches existing system expectations."""
        pass
```

## Usage Examples

### Basic Usage
```python
from cursus.mods.exe_doc import ExecutionDocumentGenerator
from cursus.api.dag.base_dag import PipelineDAG
from sagemaker.workflow.pipeline_context import PipelineSession

# Initialize generator with basic configuration
generator = ExecutionDocumentGenerator("path/to/config.json")

# Or with additional parameters for AWS operations
session = PipelineSession()
role = "arn:aws:iam::123456789012:role/SageMakerRole"
generator = ExecutionDocumentGenerator(
    config_path="path/to/config.json",
    sagemaker_session=session,
    role=role
)

# Create or load DAG
dag = PipelineDAG()
dag.add_node("cradle_data_loading")
dag.add_node("model_registration")

# Create execution document template
execution_doc = {
    "PIPELINE_STEP_CONFIGS": {
        "cradle_data_loading": {"STEP_TYPE": ["PROCESSING_STEP", "CradleDataLoading"]},
        "model_registration": {"STEP_TYPE": ["PROCESSING_STEP", "ModelRegistration"]}
    }
}

# Generate filled execution document
filled_doc = generator.fill_execution_document(dag, execution_doc)
```

### Advanced Usage with Custom Helpers
```python
# Create custom helper for new step type
class CustomStepHelper(ExecutionDocumentHelper):
    def can_handle_step(self, step_name: str, config: Any) -> bool:
        return isinstance(config, CustomStepConfig)
    
    def extract_step_config(self, step_name: str, config: CustomStepConfig) -> Dict[str, Any]:
        return {"custom_field": config.custom_value}

# Add custom helper to generator
generator = ExecutionDocumentGenerator("config.json")
generator.add_helper(CustomStepHelper())

# Use as normal
filled_doc = generator.fill_execution_document(dag, execution_doc)
```

## Benefits of Standalone Design

### 1. Complete Independence
- No dependencies on pipeline generation system
- Can be deployed and run separately
- Isolated from pipeline generation changes

### 2. Simplified Dependencies
- Can use previously unsupported packages safely
- Reduced complexity compared to integrated approach
- Clear separation of concerns

### 3. Enhanced Testability
- Easy to unit test in isolation
- Mock inputs and validate outputs
- No complex system dependencies

### 4. Deployment Flexibility
- Can run in different environments
- Separate scaling and resource management
- Independent versioning and updates

### 5. Maintainability
- Single responsibility principle
- Clear interfaces and contracts
- Extensible architecture for new step types

## Migration Path

### Phase 1: Implementation
1. Create module structure
2. Implement core ExecutionDocumentGenerator class
3. Implement Cradle and Registration helpers
4. Add configuration management utilities

### Phase 2: Integration
1. Create integration points with existing system
2. Add comprehensive test suite
3. Validate output format compatibility
4. Performance testing and optimization

### Phase 3: Deployment
1. Set up separate deployment environment
2. Configure package dependencies
3. Implement monitoring and logging
4. Create operational procedures

### Phase 4: Migration
1. Update existing systems to use standalone generator
2. Remove execution document logic from pipeline generation system
3. Clean up deprecated code
4. Update documentation and training materials

## Conclusion

The standalone execution document generator provides a clean, maintainable solution for execution document generation that is completely independent from the pipeline generation system. By running in a separate environment, it can use previously unsupported packages while maintaining the same output format and functionality as the existing system.

This design achieves the primary goals of:
- Complete system independence
- Simplified architecture
- Enhanced maintainability
- Flexible deployment options
- Preserved functionality

The modular helper-based architecture ensures extensibility for future step types while maintaining clean separation of concerns and testability.
