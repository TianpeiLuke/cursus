# Contract Validator

## Overview
The Contract Validator provides comprehensive validation tools for checking script implementations against their contracts and generating detailed compliance reports. It serves as the central validation engine for ensuring pipeline scripts meet their explicit I/O and environment requirements.

## Core Data Structures

### ContractValidationReport
Detailed report of contract validation results with comprehensive error analysis.

```python
class ContractValidationReport(BaseModel):
    """Report of contract validation results"""
    script_name: str
    contract_name: str
    is_compliant: bool
    errors: List[str] = []
    warnings: List[str] = []
    missing_inputs: List[str] = []
    missing_outputs: List[str] = []
    missing_env_vars: List[str] = []
    unexpected_inputs: List[str] = []
    
    @property
    def summary(self) -> str:
        """Generate a summary of the validation report"""
        status = "✅ COMPLIANT" if self.is_compliant else "❌ NON-COMPLIANT"
        return f"{self.script_name} vs {self.contract_name}: {status}"
```

### ScriptContractValidator
Main validation engine that orchestrates contract validation across multiple scripts.

```python
class ScriptContractValidator:
    """Validates script implementations against their contracts"""
    
    # Registry of all available contracts
    CONTRACTS = {
        "tabular_preprocess.py": TABULAR_PREPROCESS_CONTRACT,
        "mims_package.py": MIMS_PACKAGE_CONTRACT,
        "mims_payload.py": MIMS_PAYLOAD_CONTRACT,
        "model_evaluation_xgb.py": MODEL_EVALUATION_CONTRACT,
        "currency_conversion.py": CURRENCY_CONVERSION_CONTRACT,
        "risk_table_mapping.py": RISK_TABLE_MAPPING_CONTRACT,
        "train.py": PYTORCH_TRAIN_CONTRACT,
        "train_xgb.py": XGBOOST_TRAIN_CONTRACT,
    }
```

## Usage Examples

### Single Script Validation
```python
from src.pipeline_script_contracts.contract_validator import ScriptContractValidator

# Create validator for processing scripts
validator = ScriptContractValidator("src/pipeline_scripts")

# Validate a specific script
report = validator.validate_script("tabular_preprocess.py")

print(report.summary)
# Output: tabular_preprocess.py vs ScriptContract: ✅ COMPLIANT

if not report.is_compliant:
    print("Errors:")
    for error in report.errors:
        print(f"  - {error}")
    
    print("Missing inputs:", report.missing_inputs)
    print("Missing outputs:", report.missing_outputs)
    print("Missing env vars:", report.missing_env_vars)
```

### Batch Validation
```python
# Validate all scripts at once
validator = ScriptContractValidator("src/pipeline_scripts")
reports = validator.validate_all_scripts()

# Print summary for each script
for report in reports:
    print(report.summary)
    if report.warnings:
        print(f"  Warnings: {len(report.warnings)}")

# Generate comprehensive compliance report
compliance_summary = validator.generate_compliance_summary(reports)
print(compliance_summary)
```

### Training Script Validation
```python
# Validate training scripts in their respective directories
pytorch_validator = ScriptContractValidator('dockers/pytorch_bsm')
xgboost_validator = ScriptContractValidator('dockers/xgboost_atoz')

# Validate PyTorch training script
pytorch_report = pytorch_validator.validate_script('train.py')
print(f"PyTorch: {pytorch_report.summary}")

# Validate XGBoost training script
xgboost_report = xgboost_validator.validate_script('train_xgb.py')
print(f"XGBoost: {xgboost_report.summary}")
```

## Validation Process

### Validation Steps
1. **Contract Lookup** - Find contract for the specified script
2. **File Existence Check** - Verify script file exists in the directory
3. **Contract Validation** - Run the contract's validation logic
4. **Error Analysis** - Parse validation results for specific gap types
5. **Report Generation** - Create detailed validation report

### Validation Logic Flow
```python
def validate_script(self, script_name: str) -> ContractValidationReport:
    """Validate a single script against its contract"""
    
    # Step 1: Contract lookup
    if script_name not in self.CONTRACTS:
        return ContractValidationReport(
            script_name=script_name,
            contract_name="UNKNOWN",
            is_compliant=False,
            errors=[f"No contract defined for script: {script_name}"]
        )
    
    # Step 2: File existence check
    contract = self.CONTRACTS[script_name]
    script_path = self.scripts_directory / script_name
    
    if not script_path.exists():
        return ContractValidationReport(
            script_name=script_name,
            contract_name=contract.__class__.__name__,
            is_compliant=False,
            errors=[f"Script file not found: {script_path}"]
        )
    
    # Step 3: Contract validation
    validation_result = contract.validate_implementation(str(script_path))
    
    # Step 4: Create detailed report
    report = ContractValidationReport(
        script_name=script_name,
        contract_name=contract.__class__.__name__,
        is_compliant=validation_result.is_valid,
        errors=validation_result.errors,
        warnings=validation_result.warnings
    )
    
    # Step 5: Analyze specific gaps
    self._analyze_io_gaps(contract, validation_result, report)
    
    return report
```

## Error Analysis Framework

### Gap Analysis
The validator performs detailed analysis to categorize different types of validation failures:

```python
def _analyze_io_gaps(self, contract: ScriptContract, validation_result: ValidationResult, report: ContractValidationReport):
    """Analyze I/O and environment variable gaps"""
    
    # Parse errors to extract specific gap information
    for error in validation_result.errors:
        if "doesn't use expected input path" in error:
            # Extract the path from error message
            path_start = error.find(": ") + 2
            path_end = error.find(" (for ")
            if path_start > 1 and path_end > path_start:
                missing_path = error[path_start:path_end]
                report.missing_inputs.append(missing_path)
        
        elif "doesn't use expected output path" in error:
            # Extract the path from error message
            path_start = error.find(": ") + 2
            path_end = error.find(" (for ")
            if path_start > 1 and path_end > path_start:
                missing_path = error[path_start:path_end]
                report.missing_outputs.append(missing_path)
        
        elif "missing required environment variables" in error:
            # Extract environment variables from error message
            vars_start = error.find("[") + 1
            vars_end = error.find("]")
            if vars_start > 0 and vars_end > vars_start:
                vars_str = error[vars_start:vars_end]
                # Parse the list string
                missing_vars = [v.strip().strip("'\"") for v in vars_str.split(",")]
                report.missing_env_vars.extend(missing_vars)
    
    # Parse warnings for unexpected inputs
    for warning in validation_result.warnings:
        if "uses undeclared input path" in warning:
            path_start = warning.find(": ") + 2
            if path_start > 1:
                unexpected_path = warning[path_start:]
                report.unexpected_inputs.append(unexpected_path)
```

### Error Categories
The validator categorizes validation failures into specific types:

1. **Missing Inputs** - Expected input paths not found in script
2. **Missing Outputs** - Expected output paths not found in script
3. **Missing Environment Variables** - Required env vars not accessed by script
4. **Unexpected Inputs** - Script uses undeclared input paths
5. **File Not Found** - Script file doesn't exist
6. **No Contract** - No contract defined for the script

## Compliance Reporting

### Comprehensive Compliance Summary
```python
def generate_compliance_summary(self, reports: Optional[List[ContractValidationReport]] = None) -> str:
    """Generate a human-readable compliance summary"""
    
    if reports is None:
        reports = self.validate_all_scripts()
    
    compliant_count = sum(1 for r in reports if r.is_compliant)
    total_count = len(reports)
    
    summary_lines = [
        "=" * 60,
        "SCRIPT CONTRACT COMPLIANCE REPORT",
        "=" * 60,
        f"Overall Compliance: {compliant_count}/{total_count} scripts compliant",
        ""
    ]
    
    # Group by compliance status
    compliant_scripts = [r for r in reports if r.is_compliant]
    non_compliant_scripts = [r for r in reports if not r.is_compliant]
    
    # Add compliant scripts section
    if compliant_scripts:
        summary_lines.extend([
            "✅ COMPLIANT SCRIPTS:",
            "-" * 20
        ])
        for report in compliant_scripts:
            summary_lines.append(f"  • {report.script_name}")
            if report.warnings:
                summary_lines.append(f"    Warnings: {len(report.warnings)}")
        summary_lines.append("")
    
    # Add non-compliant scripts section
    if non_compliant_scripts:
        summary_lines.extend([
            "❌ NON-COMPLIANT SCRIPTS:",
            "-" * 25
        ])
        for report in non_compliant_scripts:
            summary_lines.append(f"  • {report.script_name}")
            summary_lines.append(f"    Errors: {len(report.errors)}")
            if report.missing_inputs:
                summary_lines.append(f"    Missing Inputs: {report.missing_inputs}")
            if report.missing_outputs:
                summary_lines.append(f"    Missing Outputs: {report.missing_outputs}")
            if report.missing_env_vars:
                summary_lines.append(f"    Missing Env Vars: {report.missing_env_vars}")
            if report.unexpected_inputs:
                summary_lines.append(f"    Unexpected Inputs: {report.unexpected_inputs}")
            summary_lines.append("")
    
    # Add recommendations
    summary_lines.extend([
        "=" * 60,
        "RECOMMENDATIONS:",
        "=" * 60,
        "1. Address missing I/O paths in non-compliant scripts",
        "2. Add required environment variable handling",
        "3. Document any intentional deviations from contracts",
        "4. Update contracts if script requirements have changed",
        ""
    ])
    
    return "\n".join(summary_lines)
```

### Sample Compliance Report
```
============================================================
SCRIPT CONTRACT COMPLIANCE REPORT
============================================================
Overall Compliance: 6/8 scripts compliant

✅ COMPLIANT SCRIPTS:
--------------------
  • tabular_preprocess.py
  • mims_package.py
  • mims_payload.py
  • model_evaluation_xgb.py
  • currency_conversion.py
  • risk_table_mapping.py

❌ NON-COMPLIANT SCRIPTS:
-------------------------
  • train.py
    Errors: 2
    Missing Inputs: ['/opt/ml/input/data/train']
    Missing Env Vars: ['MODEL_TYPE', 'EPOCHS']

  • train_xgb.py
    Errors: 1
    Missing Outputs: ['/opt/ml/model/model.pkl']

============================================================
RECOMMENDATIONS:
============================================================
1. Address missing I/O paths in non-compliant scripts
2. Add required environment variable handling
3. Document any intentional deviations from contracts
4. Update contracts if script requirements have changed
```

## Contract Registry

### Available Contracts
The validator maintains a registry of all available contracts:

```python
CONTRACTS = {
    # Processing Scripts
    "tabular_preprocess.py": TABULAR_PREPROCESS_CONTRACT,
    "mims_package.py": MIMS_PACKAGE_CONTRACT,
    "mims_payload.py": MIMS_PAYLOAD_CONTRACT,
    "model_evaluation_xgb.py": MODEL_EVALUATION_CONTRACT,
    "currency_conversion.py": CURRENCY_CONVERSION_CONTRACT,
    "risk_table_mapping.py": RISK_TABLE_MAPPING_CONTRACT,
    
    # Training Scripts
    "train.py": PYTORCH_TRAIN_CONTRACT,
    "train_xgb.py": XGBOOST_TRAIN_CONTRACT,
}
```

### Contract Coverage
- **Processing Scripts**: 6 contracts covering data processing operations
- **Training Scripts**: 2 contracts covering PyTorch and XGBoost training
- **Total Coverage**: 8 script contracts with comprehensive I/O validation

## CLI Interface

### Command Line Usage
The validator provides a CLI interface for integration with CI/CD pipelines:

```bash
# Validate specific script
python -m src.pipeline_script_contracts.contract_validator --script tabular_preprocess.py

# Validate all scripts with detailed output
python -m src.pipeline_script_contracts.contract_validator --verbose

# Validate scripts in custom directory
python -m src.pipeline_script_contracts.contract_validator --scripts-dir custom/scripts/path
```

### CLI Implementation
```python
def main():
    """CLI entry point for contract validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate pipeline scripts against their contracts")
    parser.add_argument("--script", help="Validate specific script (e.g., tabular_preprocess.py)")
    parser.add_argument("--scripts-dir", default="src/pipeline_scripts", help="Directory containing scripts")
    parser.add_argument("--verbose", action="store_true", help="Show detailed validation results")
    
    args = parser.parse_args()
    
    validator = ScriptContractValidator(args.scripts_dir)
    
    if args.script:
        # Validate single script
        report = validator.validate_script(args.script)
        print(report.summary)
        if args.verbose:
            if report.errors:
                print(f"Errors: {report.errors}")
            if report.warnings:
                print(f"Warnings: {report.warnings}")
    else:
        # Validate all scripts
        reports = validator.validate_all_scripts()
        summary = validator.generate_compliance_summary(reports)
        print(summary)
```

## Advanced Features

### Custom Script Directories
```python
# Validate scripts in different directories
processing_validator = ScriptContractValidator("src/pipeline_scripts")
pytorch_validator = ScriptContractValidator("dockers/pytorch_bsm")
xgboost_validator = ScriptContractValidator("dockers/xgboost_atoz")

# Each validator operates on its own directory
processing_reports = processing_validator.validate_all_scripts()
pytorch_reports = pytorch_validator.validate_all_scripts()
xgboost_reports = xgboost_validator.validate_all_scripts()
```

### Programmatic Integration
```python
# Integration with CI/CD pipeline
def validate_pipeline_scripts():
    validator = ScriptContractValidator()
    reports = validator.validate_all_scripts()
    
    non_compliant = [r for r in reports if not r.is_compliant]
    if non_compliant:
        print("❌ Script validation failed!")
        for report in non_compliant:
            print(f"  {report.script_name}: {len(report.errors)} errors")
        return False
    
    print("✅ All scripts are compliant with their contracts")
    return True

# Use in CI/CD
if not validate_pipeline_scripts():
    exit(1)  # Fail the build
```

### Filtering and Analysis
```python
# Filter reports by compliance status
def analyze_compliance_trends(validator: ScriptContractValidator):
    reports = validator.validate_all_scripts()
    
    # Group by script type
    processing_scripts = [r for r in reports if not r.script_name.startswith('train')]
    training_scripts = [r for r in reports if r.script_name.startswith('train')]
    
    # Calculate compliance rates
    processing_compliance = sum(1 for r in processing_scripts if r.is_compliant) / len(processing_scripts)
    training_compliance = sum(1 for r in training_scripts if r.is_compliant) / len(training_scripts)
    
    print(f"Processing scripts compliance: {processing_compliance:.1%}")
    print(f"Training scripts compliance: {training_compliance:.1%}")
    
    # Identify common issues
    all_missing_inputs = []
    all_missing_env_vars = []
    
    for report in reports:
        all_missing_inputs.extend(report.missing_inputs)
        all_missing_env_vars.extend(report.missing_env_vars)
    
    print(f"Most common missing inputs: {set(all_missing_inputs)}")
    print(f"Most common missing env vars: {set(all_missing_env_vars)}")
```

## Integration Points

### With CI/CD Pipelines
```yaml
# GitHub Actions example
- name: Validate Script Contracts
  run: |
    python -m src.pipeline_script_contracts.contract_validator
    if [ $? -ne 0 ]; then
      echo "Script contract validation failed"
      exit 1
    fi
```

### With Step Builders
```python
class ProcessingStepBuilder:
    def validate_script_compliance(self):
        validator = ScriptContractValidator()
        report = validator.validate_script(self.script_name)
        
        if not report.is_compliant:
            raise ValueError(f"Script {self.script_name} is not compliant: {report.errors}")
```

### With Pipeline Templates
```python
class PipelineTemplate:
    def validate_all_scripts(self):
        validator = ScriptContractValidator()
        reports = validator.validate_all_scripts()
        
        non_compliant = [r for r in reports if not r.is_compliant]
        if non_compliant:
            raise ValueError(f"Pipeline has {len(non_compliant)} non-compliant scripts")
```

## Best Practices

### 1. Validation Strategy
- Run validation in CI/CD pipelines before deployment
- Validate scripts after any changes to contracts or implementations
- Use verbose mode during development for detailed feedback

### 2. Error Handling
- Address all validation errors before deployment
- Document any intentional deviations from contracts
- Update contracts when script requirements legitimately change

### 3. Reporting
- Generate compliance reports regularly for monitoring
- Track compliance trends over time
- Use reports to identify common patterns and issues

### 4. Integration
- Integrate validation into development workflow
- Use programmatic validation in automated testing
- Fail builds on validation errors to enforce compliance

## Related Design Documentation

For architectural context and design decisions, see:
- **[Script Contract Design](../pipeline_design/script_contract.md)** - Script contract architecture and patterns
- **[Step Contract Design](../pipeline_design/step_contract.md)** - Step-level contract definitions
- **[Specification Driven Design](../pipeline_design/specification_driven_design.md)** - Overall design philosophy
- **[Design Principles](../pipeline_design/design_principles.md)** - Core design principles and guidelines
- **[Standardization Rules](../pipeline_design/standardization_rules.md)** - Naming and structure conventions

## Performance Considerations

### Validation Speed
- Single script validation: ~100-500ms depending on script size
- Batch validation: Linear scaling with number of scripts
- AST parsing is the primary performance bottleneck

### Memory Usage
- Each validation creates temporary AST objects
- Memory usage scales with script size and complexity
- Consider batch processing for large numbers of scripts

### Optimization Strategies
- Cache AST parsing results for repeated validations
- Use parallel processing for batch validation
- Pre-compile contracts for faster validation
