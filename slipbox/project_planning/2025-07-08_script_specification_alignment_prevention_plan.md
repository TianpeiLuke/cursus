# Script-Specification Alignment Prevention Plan

## Problem Statement

The pipeline system has a **four-layer architecture** that requires precise alignment to prevent runtime failures:

1. **Producer Step Specifications** - Define outputs with logical names and property paths
2. **Consumer Step Specifications** - Define dependencies with logical names and semantic matching
3. **Script Contracts** - Define container paths where scripts expect inputs/outputs
4. **Step Builders** - Bridge specifications and contracts via SageMaker ProcessingInput/Output
5. **Actual Script Implementation** - The runtime behavior that uses the container paths

**Critical Insight**: The alignment is NOT between step specifications and script contracts directly, but rather:
- **Specifications** define the **channel names** (`input_name`) and **data flow** (`source` S3 URIs)
- **Contracts** define the **container paths** (`destination` for inputs, `source` for outputs)
- **Step Builders** create the **SageMaker ProcessingInput/Output** that maps `source` → `destination`

Misalignments can cause runtime failures, incorrect dependency resolution, and maintenance issues.

## Current Misalignment Issues Identified

### 1. **Logical Name Inconsistency**
- **Problem**: Step specifications and script contracts use different keys for the same logical concept
- **Example**: Spec uses `logical_name="processed_data"` but contract uses `expected_output_paths["ProcessedTabularData"]`
- **Impact**: Step builders cannot automatically map between spec and contract

### 2. **Property Path Inconsistency**
- **Problem**: OutputSpec property_path doesn't match the logical_name
- **Example**: `logical_name="processed_data"` but `property_path="...Outputs['ProcessedTabularData']..."`
- **Impact**: Runtime property access fails because the path doesn't match the logical name

### 3. **Missing Contract Coverage**
- **Problem**: Step specifications define dependencies/outputs that don't exist in script contracts
- **Example**: Spec defines `METADATA` and `SIGNATURE` dependencies but contract only has `DATA`
- **Impact**: Step builder cannot create ProcessingInput for missing contract paths

### 4. **Hardcoded Step Builder Paths**
- **Problem**: Step builders use hardcoded container paths instead of deriving from contracts
- **Example**: `"/opt/ml/processing/input/data"` hardcoded instead of using `contract.expected_input_paths['DATA']`
- **Impact**: Changes to contracts don't automatically propagate to step builders

### 5. **Cross-Step Semantic Mismatch**
- **Problem**: Producer step outputs don't match consumer step dependency logical names
- **Example**: Producer outputs `logical_name="training_data"` but consumer expects `logical_name="DATA"`
- **Impact**: Automatic dependency resolution fails

## Solution Architecture

### Phase 1: Logical Name Consistency Framework

#### 1.1 Enhanced Contract Alignment Validation
```python
# Updated StepSpecification.validate_contract_alignment() in src/pipeline_deps/base_specifications.py
def validate_contract_alignment(self) -> ValidationResult:
    """Validate that script contract aligns with step specification using logical names as keys"""
    if not self.script_contract:
        return ValidationResult.success("No contract to validate")
    
    errors = []
    warnings = []
    
    # Input alignment: DependencySpec.logical_name must be key in contract.expected_input_paths
    for dep in self.dependencies.values():
        if dep.required and dep.logical_name not in self.script_contract.expected_input_paths:
            errors.append(f"Required dependency '{dep.logical_name}' missing in contract expected_input_paths")
    
    # Output alignment: OutputSpec.logical_name must be key in contract.expected_output_paths  
    for output in self.outputs.values():
        if output.logical_name not in self.script_contract.expected_output_paths:
            errors.append(f"Output '{output.logical_name}' missing in contract expected_output_paths")
    
    # Property path consistency: OutputSpec.property_path must reference OutputSpec.logical_name
    for output in self.outputs.values():
        expected_property_path = f"properties.ProcessingOutputConfig.Outputs['{output.logical_name}'].S3Output.S3Uri"
        if output.property_path != expected_property_path:
            errors.append(f"OutputSpec '{output.logical_name}' property_path inconsistent. Expected: {expected_property_path}, Got: {output.property_path}")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
```

#### 1.2 Cross-Step Semantic Validation
```python
# Add to src/pipeline_deps/specification_registry.py
def validate_cross_step_compatibility(producer_spec: StepSpecification, consumer_spec: StepSpecification):
    """Validate that producer outputs can satisfy consumer dependencies"""
    errors = []
    
    for dep in consumer_spec.dependencies.values():
        if dep.required:
            # Find matching output in producer by logical name
            matching_output = producer_spec.get_output(dep.logical_name)
            if not matching_output:
                errors.append(f"Producer missing output '{dep.logical_name}' required by consumer")
            
            # Validate semantic compatibility
            if not dep.is_compatible_with_source(producer_spec.step_type):
                errors.append(f"Producer '{producer_spec.step_type}' not in compatible sources for '{dep.logical_name}'")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
```

#### 1.2 Specification Validation Against Contracts
```python
# Add to src/pipeline_deps/specification_registry.py
def validate_spec_against_contract(spec: StepSpecification) -> ValidationResult:
    """Ensure specification is compatible with its contract"""
    if not spec.script_contract:
        return ValidationResult.success("No contract defined")
    
    errors = []
    
    # Check that all required dependencies have corresponding contract inputs
    required_deps = [dep for dep in spec.dependencies if dep.required]
    contract_inputs = set(spec.script_contract.expected_input_paths.keys())
    
    for dep in required_deps:
        if dep.logical_name not in contract_inputs:
            errors.append(f"Required dependency '{dep.logical_name}' not in contract inputs")
    
    # Check that all outputs have corresponding contract outputs
    spec_outputs = set(output.logical_name for output in spec.outputs)
    contract_outputs = set(spec.script_contract.expected_output_paths.keys())
    
    missing_outputs = spec_outputs - contract_outputs
    if missing_outputs:
        errors.append(f"Specification outputs not in contract: {missing_outputs}")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
```

### Phase 2: Spec-Contract Driven Step Builders

#### 2.1 Enhanced Step Builder Base Class
```python
# Update src/pipeline_steps/builder_step_base.py
class StepBuilderBase:
    def __init__(self, config, spec: StepSpecification, contract: ScriptContract, ...):
        self.spec = spec
        self.contract = contract
        # Validate alignment during initialization
        self._validate_spec_contract_alignment()
    
    def _validate_spec_contract_alignment(self):
        """Validate that spec and contract are properly aligned"""
        errors = []
        
        # Check inputs: all required dependencies must have contract paths
        for dep in self.spec.dependencies.values():
            if dep.required and dep.logical_name not in self.contract.expected_input_paths:
                errors.append(f"Spec dependency '{dep.logical_name}' missing in contract inputs")
        
        # Check outputs: all outputs must have contract paths
        for output in self.spec.outputs.values():
            if output.logical_name not in self.contract.expected_output_paths:
                errors.append(f"Spec output '{output.logical_name}' missing in contract outputs")
        
        if errors:
            raise ValueError(f"Spec-Contract alignment errors: {errors}")
    
    def _get_spec_driven_processor_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Generate processor inputs using both spec and contract"""
        processing_inputs = []
        
        for dep in self.spec.dependencies.values():
            if dep.required or dep.logical_name in inputs:
                container_path = self.contract.expected_input_paths[dep.logical_name]
                processing_inputs.append(
                    ProcessingInput(
                        input_name=dep.logical_name,  # From spec
                        source=inputs[dep.logical_name],  # S3 URI from pipeline flow
                        destination=container_path     # From contract
                    )
                )
        
        return processing_inputs
    
    def _get_spec_driven_processor_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Generate processor outputs using both spec and contract"""
        processing_outputs = []
        
        for output_spec in self.spec.outputs.values():
            container_path = self.contract.expected_output_paths[output_spec.logical_name]
            processing_outputs.append(
                ProcessingOutput(
                    output_name=output_spec.logical_name,  # From spec (matches property path)
                    source=container_path,                 # From contract
                    destination=None  # SageMaker generates this
                )
            )
        
        return processing_outputs
```

#### 2.2 Contract-Aware Utility Functions
```python
# Enhanced src/pipeline_scripts/contract_utils.py
import os
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def validate_contract_environment(contract):
    """Validate SageMaker environment matches contract expectations"""
    errors = []
    
    # Check required environment variables
    for var in contract.required_env_vars:
        if var not in os.environ:
            errors.append(f"Missing required environment variable: {var}")
    
    # Check input paths exist (SageMaker mounts these)
    for logical_name, path in contract.expected_input_paths.items():
        if not os.path.exists(path):
            errors.append(f"Input path not found: {path} ({logical_name})")
    
    # Ensure output directories exist
    for logical_name, path in contract.expected_output_paths.items():
        os.makedirs(path, exist_ok=True)
        logger.info(f"Ensured output directory exists: {path}")
    
    if errors:
        raise RuntimeError(f"Contract validation failed: {errors}")
    
    logger.info("Contract environment validation passed")

def get_contract_paths(contract):
    """Get input/output paths from contract for easy access"""
    return {
        'inputs': contract.expected_input_paths,
        'outputs': contract.expected_output_paths
    }
```

#### 2.3 Updated Step Builder Pattern
```python
# Example: Updated TabularPreprocessingStepBuilder
class TabularPreprocessingStepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None):
        # Get spec and contract
        spec = PREPROCESSING_TRAINING_SPEC
        contract = _get_tabular_preprocess_contract()
        
        super().__init__(
            config=config,
            spec=spec,
            contract=contract,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
    
    def _get_processor_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Use spec-driven approach instead of hardcoded paths"""
        return self._get_spec_driven_processor_inputs(inputs)
    
    def _get_processor_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Use spec-driven approach instead of hardcoded paths"""
        return self._get_spec_driven_processor_outputs(outputs)
```

#### 2.4 Refactored Script Pattern
```python
# Standard pattern for all SageMaker scripts
def get_script_contract():
    """Get the contract for this script"""
    # Import at runtime to avoid circular imports
    from ..pipeline_script_contracts.tabular_preprocess_contract import TABULAR_PREPROCESS_CONTRACT
    return TABULAR_PREPROCESS_CONTRACT

def main():
    """Main entry point with contract validation"""
    # 1. Get and validate contract
    contract = get_script_contract()
    validate_contract_environment(contract)
    paths = get_contract_paths(contract)
    
    # 2. Use contract paths instead of hardcoded paths (using logical names)
    data_dir = paths['inputs']['DATA']  # From contract.expected_input_paths['DATA']
    output_dir = paths['outputs']['processed_data']  # From contract.expected_output_paths['processed_data']
    
    # 3. Access validated environment variables
    LABEL_FIELD = os.environ["LABEL_FIELD"]  # Contract ensures this exists
    TRAIN_RATIO = os.environ["TRAIN_RATIO"]
    
    # 4. Business logic remains unchanged
    # ... existing script logic
```

### Phase 3: Automated Alignment Enforcement

#### 3.1 Enhanced Pre-commit Validation Hook
```python
# Enhanced tools/validate_contracts.py
def validate_all_contracts():
    """Validate all script contracts against their specifications"""
    from src.pipeline_step_specs import (
        DATA_LOADING_TRAINING_SPEC,
        PREPROCESSING_TRAINING_SPEC,
        XGBOOST_TRAINING_SPEC,
        MODEL_EVAL_SPEC
    )
    
    specs_with_contracts = [
        DATA_LOADING_TRAINING_SPEC,
        PREPROCESSING_TRAINING_SPEC,
        XGBOOST_TRAINING_SPEC,
        MODEL_EVAL_SPEC
    ]
    
    all_valid = True
    for spec in specs_with_contracts:
        # Validate contract alignment
        result = spec.validate_contract_alignment()
        if not result.is_valid:
            print(f"❌ {spec.step_type}: {result.errors}")
            all_valid = False
        else:
            print(f"✅ {spec.step_type}: Contract aligned")
        
        # Validate property path consistency
        for output in spec.outputs.values():
            expected_path = f"properties.ProcessingOutputConfig.Outputs['{output.logical_name}'].S3Output.S3Uri"
            if output.property_path != expected_path:
                print(f"⚠️  {spec.step_type}: Property path inconsistency for '{output.logical_name}'")
                print(f"   Expected: {expected_path}")
                print(f"   Got: {output.property_path}")
                all_valid = False
    
    return all_valid

def validate_cross_step_compatibility():
    """Validate compatibility between connected steps"""
    # Example: Data Loading → Preprocessing
    from src.pipeline_step_specs import DATA_LOADING_TRAINING_SPEC, PREPROCESSING_TRAINING_SPEC
    
    result = validate_cross_step_compatibility(DATA_LOADING_TRAINING_SPEC, PREPROCESSING_TRAINING_SPEC)
    if not result.is_valid:
        print(f"❌ Cross-step compatibility: {result.errors}")
        return False
    else:
        print("✅ Cross-step compatibility validated")
        return True

if __name__ == "__main__":
    import sys
    contract_valid = validate_all_contracts()
    cross_step_valid = validate_cross_step_compatibility()
    
    if not (contract_valid and cross_step_valid):
        sys.exit(1)
```

#### 3.2 Runtime Contract Enforcement
```python
# Enhanced ScriptContract with runtime validation
class ScriptContract(BaseModel):
    # ... existing fields
    
    def enforce_at_runtime(self):
