# Pipeline Script Contracts

This module provides script contracts that define explicit I/O and environment requirements for pipeline scripts, bridging the gap between step specifications and script implementations with automated validation capabilities.

## Contract Types

### Base Contracts
- **base_script_contract.py** - Foundation contract for processing scripts
- **training_script_contract.py** - Specialized contract for training scripts
- **contract_validator.py** - Validation framework and AST analysis

### Processing Script Contracts
- **tabular_preprocess_contract.py** - Tabular data preprocessing
- **mims_package_contract.py** - MIMS model packaging
- **mims_payload_contract.py** - MIMS payload generation
- **model_evaluation_contract.py** - XGBoost model evaluation
- **currency_conversion_contract.py** - Currency conversion processing
- **risk_table_mapping_contract.py** - Risk table mapping operations

### Training Script Contracts
- **pytorch_train_contract.py** - PyTorch Lightning multimodal training
- **xgboost_train_contract.py** - XGBoost tabular training

## Key Features

1. **Explicit I/O Contracts** - Clear input/output path specifications
2. **Environment Validation** - Required and optional environment variables
3. **Framework Requirements** - Exact dependency versions from requirements.txt
4. **AST Analysis** - Static code analysis for compliance checking
5. **Automated Validation** - CLI and programmatic validation tools
6. **Dual Path Support** - Processing (`/opt/ml/processing/*`) and Training (`/opt/ml/input/*`, `/opt/ml/model/*`) paths

## Usage Examples

### Contract Definition
```python
from src.pipeline_script_contracts import ScriptContract

TABULAR_PREPROCESS_CONTRACT = ScriptContract(
    entry_point="tabular_preprocess.py",
    expected_input_paths={
        "input_data": "/opt/ml/processing/input/data",
        "metadata": "/opt/ml/processing/input/metadata"
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output/data"
    },
    required_env_vars=["LABEL_FIELD", "TRAIN_RATIO"],
    framework_requirements={
        "pandas": ">=1.3.0",
        "scikit-learn": ">=1.0.0"
    }
)
```

### Script Validation
```python
from src.pipeline_script_contracts import ScriptContractValidator

# Validate single script
validator = ScriptContractValidator('src/pipeline_scripts')
report = validator.validate_script('tabular_preprocess.py')
print(report.summary)

# Validate all scripts
reports = validator.validate_all_scripts()
summary = validator.generate_compliance_summary(reports)
```

### CLI Usage
```bash
# Validate specific script
python -m src.pipeline_script_contracts.contract_validator --script train.py

# Validate all scripts with detailed output
python -m src.pipeline_script_contracts.contract_validator --verbose
```

## Contract Coverage

### Processing Scripts (6 contracts)
- ✅ **tabular_preprocess.py** - Tabular data preprocessing with risk tables
- ✅ **mims_package.py** - MIMS model packaging 
- ✅ **mims_payload.py** - MIMS payload generation
- ✅ **model_evaluation_xgb.py** - XGBoost model evaluation
- ✅ **currency_conversion.py** - Currency conversion processing
- ✅ **risk_table_mapping.py** - Risk table mapping operations

### Training Scripts (2 contracts)
- ✅ **train.py** - PyTorch Lightning multimodal training (torch==2.1.2, lightning==2.1.3)
- ✅ **train_xgb.py** - XGBoost tabular training (xgboost==1.7.6)

## Validation Framework

### AST Analysis
The validation system uses Abstract Syntax Tree analysis to detect:
- Input/output path usage patterns
- Environment variable access
- Framework import statements
- Dynamic path construction

### Validation Results
```python
# Example validation output
=== PyTorch Training Script Validation ===
train.py vs TrainingScriptContract: ❌ NON-COMPLIANT
Errors: ["Script doesn't use expected input path: /opt/ml/input/data/train"]

=== XGBoost Training Script Validation ===  
train_xgb.py vs TrainingScriptContract: ❌ NON-COMPLIANT
Errors: ["Script doesn't use expected input path: /opt/ml/input/data/train"]
```

## Integration Points

### With Step Specifications
```python
@dataclass
class StepSpecification:
    script_contract: Optional[ScriptContract] = None
    
    def validate_script_compliance(self, script_path: str) -> ValidationResult:
        return self.script_contract.validate_implementation(script_path)
```

### With Step Builders
```python
class TabularPreprocessingStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        validation = TABULAR_PREPROCESS_CONTRACT.validate_implementation(script_path)
        if not validation.is_valid:
            raise ValueError(f"Script validation failed: {validation.errors}")
```

## Business Value

1. **Fail-Fast Validation** - Catch errors at build time, not runtime
2. **Explicit Documentation** - Script requirements are self-documenting
3. **Safe Refactoring** - Changes to specifications validate all implementations
4. **Framework Standardization** - Exact dependency versions prevent conflicts
5. **CI/CD Integration** - Automated validation prevents non-compliant deployments

## File Structure

```
src/pipeline_script_contracts/
├── __init__.py                     # Module exports
├── base_script_contract.py         # Base contract for processing scripts
├── training_script_contract.py     # Specialized contract for training scripts
├── contract_validator.py           # Validation framework
├── tabular_preprocess_contract.py  # Tabular preprocessing contract
├── mims_package_contract.py        # MIMS packaging contract
├── mims_payload_contract.py        # MIMS payload contract
├── model_evaluation_contract.py    # Model evaluation contract
├── currency_conversion_contract.py # Currency conversion contract
├── risk_table_mapping_contract.py  # Risk table mapping contract
├── pytorch_train_contract.py       # PyTorch training contract
└── xgboost_train_contract.py       # XGBoost training contract
```

This module successfully bridges the gap between architectural intent (Step Specifications) and implementation reality (actual scripts), ensuring they remain aligned throughout the development lifecycle with automated validation and explicit documentation.

## Documentation Coverage

### Core Framework Documentation
- **[Base Script Contract](base_script_contract.md)** - Foundation data structures and validation framework
- **[Contract Validator](contract_validator.md)** - Validation engine and compliance reporting

### Implementation Examples
- **[PyTorch Train Contract](pytorch_train_contract.md)** - Example PyTorch training contract implementation

## Related Documentation

### Script Contract Components
- [Base Script Contract](base_script_contract.md): Foundation contract for script validation
- [Contract Validator](contract_validator.md): Validation framework and AST analysis
- [PyTorch Train Contract](pytorch_train_contract.md): Example PyTorch training contract

### Pipeline Building
- [Pipeline Template Base](../pipeline_builder/pipeline_template_base.md): Core abstract class for templates
- [Pipeline Builder Overview](../pipeline_builder/README.md): Complete pipeline building system

### Specification System
- [Base Specifications](../pipeline_deps/base_specifications.md): Core specification structures
- [Dependency Resolver](../pipeline_deps/dependency_resolver.md): Dependency resolution system
- [Pipeline Steps](../pipeline_steps/README.md): Step specifications that use script contracts

### Integration Points
- [Pipeline DAG](../pipeline_dag/pipeline_dag.md): DAG structure for pipeline steps
- [Pipeline Dependencies](../pipeline_deps/README.md): Dependency resolution system

## Design Documentation

For architectural context and design decisions, see:
- **[Script Contract Design](../pipeline_design/script_contract.md)** - Script contract architecture and patterns
- **[Step Contract Design](../pipeline_design/step_contract.md)** - Step-level contract definitions
- **[Specification Driven Design](../pipeline_design/specification_driven_design.md)** - Overall design philosophy
- **[Step Builder Design](../pipeline_design/step_builder.md)** - Step builder integration with contracts
- **[Standardization Rules](../pipeline_design/standardization_rules.md)** - Naming and structure conventions
- **[Design Principles](../pipeline_design/design_principles.md)** - Core design principles and guidelines
