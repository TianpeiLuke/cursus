# Script Alignment Validation System

This directory contains a comprehensive alignment validation system for all scripts in `src/cursus/steps/scripts`. The system validates alignment across four critical levels to ensure consistency and correctness throughout the Cursus pipeline architecture.

## 🎯 Validation Levels

The alignment validation system checks four levels of alignment:

### Level 1: Script ↔ Contract Alignment
- Validates that script arguments match contract specifications
- Ensures all required paths are properly declared
- Checks for unused or undeclared arguments

### Level 2: Contract ↔ Specification Alignment  
- Verifies contract fields align with step specifications
- Validates field types and constraints
- Ensures all required fields are present

### Level 3: Specification ↔ Dependencies Alignment
- Checks dependency resolution and compatibility
- Validates specification requirements
- Ensures all dependencies are properly declared

### Level 4: Builder ↔ Configuration Alignment
- Validates step builder configuration
- Ensures proper field mapping and resolution
- Checks for configuration consistency

## 📁 Directory Structure

```
test/steps/scripts/alignment_validation/
├── README.md                           # This documentation
├── run_alignment_validation.py         # Main comprehensive validation runner
├── generate_validation_scripts.py      # Script generator for individual validators
├── validate_currency_conversion.py     # Individual validator for currency_conversion
├── validate_dummy_training.py          # Individual validator for dummy_training
├── validate_mims_package.py            # Individual validator for mims_package
├── validate_mims_payload.py            # Individual validator for mims_payload
├── validate_model_calibration.py       # Individual validator for model_calibration
├── validate_model_evaluation_xgb.py    # Individual validator for model_evaluation_xgb
├── validate_risk_table_mapping.py      # Individual validator for risk_table_mapping
├── validate_tabular_preprocess.py      # Individual validator for tabular_preprocess
└── reports/                            # Generated validation reports
    ├── json/                           # JSON format reports
    ├── html/                           # HTML format reports
    ├── individual/                     # Individual script reports
    └── validation_summary.json         # Overall validation summary
```

## 🚀 Usage

### Run Comprehensive Validation for All Scripts

```bash
cd test/steps/scripts/alignment_validation
python run_alignment_validation.py
```

This will:
- Discover all scripts in `src/cursus/steps/scripts`
- Run alignment validation across all 4 levels for each script
- Generate detailed JSON and HTML reports
- Create an overall validation summary

### Run Validation for Individual Scripts

```bash
# Validate specific script
python validate_currency_conversion.py
python validate_dummy_training.py
python validate_mims_package.py
# ... etc for other scripts
```

### Generate New Individual Validators

If new scripts are added to `src/cursus/steps/scripts`, regenerate the individual validators:

```bash
python generate_validation_scripts.py
```

## 📊 Report Formats

### JSON Reports
- Machine-readable format
- Complete validation results with metadata
- Suitable for automation and CI/CD integration
- Located in `reports/json/`

### HTML Reports  
- Human-readable format with visual styling
- Interactive dashboard with metrics
- Color-coded issue severity levels
- Located in `reports/html/`

### Validation Summary
- Overall statistics across all scripts
- Pass/fail rates by script and level
- Aggregated issue counts and severity distribution
- Located in `reports/validation_summary.json`

## 🔍 Understanding Results

### Overall Status
- **PASSING**: All 4 alignment levels pass validation
- **FAILING**: One or more alignment levels have issues
- **ERROR**: Validation could not be completed due to errors

### Issue Severity Levels
- **CRITICAL**: Must be fixed immediately, blocks functionality
- **ERROR**: Significant issues that should be addressed
- **WARNING**: Potential issues that should be reviewed
- **INFO**: Informational messages for awareness

### Level-by-Level Results
Each validation level provides:
- Pass/fail status
- List of issues found
- Detailed issue descriptions
- Recommendations for fixes

## 🛠️ Integration with CI/CD

The validation system is designed for automation:

```bash
# Run validation and exit with appropriate code
python run_alignment_validation.py
echo $?  # 0 = all passed, 1 = some failed, 2 = errors

# Run individual script validation
python validate_currency_conversion.py
echo $?  # 0 = passed, 1 = failed, 2 = error
```

## 📋 Validation Checklist

Before deploying changes to scripts, contracts, specifications, or builders:

1. ✅ Run comprehensive validation: `python run_alignment_validation.py`
2. ✅ Review HTML reports for any failing scripts
3. ✅ Address all CRITICAL and ERROR level issues
4. ✅ Consider addressing WARNING level issues
5. ✅ Verify overall validation summary shows acceptable pass rates
6. ✅ Update documentation if alignment patterns change

## 🔧 Troubleshooting

### Common Issues

**Import Errors**
- Ensure all dependencies are installed
- Check that Python path includes project root
- Verify all required modules are available

**Missing Files**
- Ensure corresponding contracts, specs, and builders exist
- Check file naming conventions match expected patterns
- Verify directory structure is correct

**Validation Failures**
- Review detailed issue descriptions in reports
- Follow provided recommendations
- Check for recent changes that might have broken alignment

### Getting Help

1. Check the detailed HTML reports for specific guidance
2. Review the validation summary for overall patterns
3. Examine individual script validators for focused analysis
4. Consult the Cursus documentation for alignment requirements

## 📈 Metrics and Monitoring

The validation system provides comprehensive metrics:

- **Script Coverage**: Percentage of scripts with validation
- **Pass Rate**: Percentage of scripts passing all levels
- **Issue Distribution**: Breakdown by severity and category
- **Trend Analysis**: Historical validation results over time

Use these metrics to:
- Monitor code quality over time
- Identify problematic patterns
- Track improvement efforts
- Ensure alignment standards are maintained

## 🎯 Best Practices

1. **Run Validation Early**: Validate changes before committing
2. **Address Issues Promptly**: Don't let alignment debt accumulate
3. **Review Reports Regularly**: Use HTML reports for detailed analysis
4. **Automate in CI/CD**: Include validation in automated pipelines
5. **Monitor Trends**: Track validation metrics over time
6. **Document Exceptions**: Record any intentional alignment deviations

---

*Generated by Cursus Alignment Validation System*
