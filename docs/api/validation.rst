Validation Framework
===================

The validation framework provides comprehensive validation capabilities for pipeline components, configurations, and step alignment in the Cursus framework.

.. currentmodule:: cursus.validation

Overview
--------

The validation system consists of several key components:

- **Alignment Validation** (:mod:`cursus.validation.alignment`): Validates alignment between contracts, specifications, and implementations
- **Builder Validation** (:mod:`cursus.validation.builders`): Validates step builder implementations and configurations
- **Interface Validation** (:mod:`cursus.validation.interface`): Validates interface contracts and API compliance
- **Naming Validation** (:mod:`cursus.validation.naming`): Validates naming conventions and consistency
- **Shared Utilities** (:mod:`cursus.validation.shared`): Common validation utilities and base classes

Alignment Validation (:mod:`cursus.validation.alignment`)
---------------------------------------------------------

.. automodule:: cursus.validation.alignment
   :members:
   :undoc-members:
   :show-inheritance:

Unified Alignment Tester
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.validation.alignment.unified_alignment_tester
   :members:
   :undoc-members:
   :show-inheritance:

Contract Alignment
~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.validation.alignment.contract_alignment
   :members:
   :undoc-members:
   :show-inheritance:

Specification Alignment
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.validation.alignment.specification_alignment
   :members:
   :undoc-members:
   :show-inheritance:

Builder Validation (:mod:`cursus.validation.builders`)
------------------------------------------------------

.. automodule:: cursus.validation.builders
   :members:
   :undoc-members:
   :show-inheritance:

Universal Step Builder Tester
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.validation.builders.universal_step_builder_tester
   :members:
   :undoc-members:
   :show-inheritance:

Builder Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.validation.builders.builder_config_validation
   :members:
   :undoc-members:
   :show-inheritance:

Interface Validation (:mod:`cursus.validation.interface`)
---------------------------------------------------------

.. automodule:: cursus.validation.interface
   :members:
   :undoc-members:
   :show-inheritance:

API Contract Validation
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.validation.interface.api_contract_validation
   :members:
   :undoc-members:
   :show-inheritance:

Interface Compliance
~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.validation.interface.interface_compliance
   :members:
   :undoc-members:
   :show-inheritance:

Naming Validation (:mod:`cursus.validation.naming`)
---------------------------------------------------

.. automodule:: cursus.validation.naming
   :members:
   :undoc-members:
   :show-inheritance:

Naming Convention Checker
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.validation.naming.naming_convention_checker
   :members:
   :undoc-members:
   :show-inheritance:

Consistency Validation
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.validation.naming.consistency_validation
   :members:
   :undoc-members:
   :show-inheritance:

Shared Utilities (:mod:`cursus.validation.shared`)
--------------------------------------------------

.. automodule:: cursus.validation.shared
   :members:
   :undoc-members:
   :show-inheritance:

Validation Base Classes
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.validation.shared.validation_base
   :members:
   :undoc-members:
   :show-inheritance:

Common Validators
~~~~~~~~~~~~~~~~~

.. automodule:: cursus.validation.shared.common_validators
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.validation import UnifiedAlignmentTester
   
   # Create alignment tester
   tester = UnifiedAlignmentTester()
   
   # Validate step alignment
   results = tester.validate_step_alignment("xgboost_training")
   
   if results.is_valid:
       print("Step alignment validation passed!")
   else:
       print("Validation errors:")
       for error in results.errors:
           print(f"  - {error}")

Step Builder Validation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.validation.builders import UniversalStepBuilderTester
   from cursus.steps.builders import XGBoostTrainingStepBuilder
   from cursus.steps.configs import XGBoostTrainingConfig
   
   # Create step builder tester
   tester = UniversalStepBuilderTester()
   
   # Create test configuration
   config = XGBoostTrainingConfig(
       input_path="s3://test-bucket/data/",
       output_path="s3://test-bucket/models/",
       training_instance_type="ml.m5.xlarge"
   )
   
   # Validate step builder
   builder = XGBoostTrainingStepBuilder(config=config)
   validation_result = tester.validate_builder(builder)
   
   if validation_result.is_valid:
       print("Step builder validation passed!")
   else:
       print(f"Validation failed: {validation_result.errors}")

Contract Validation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.validation.alignment import ContractAlignmentValidator
   from cursus.steps.contracts import XGBoostTrainingContract
   
   # Create contract validator
   validator = ContractAlignmentValidator()
   
   # Validate contract alignment
   contract = XGBoostTrainingContract()
   alignment_result = validator.validate_contract_alignment(
       contract,
       step_type="XGBoostTraining"
   )
   
   if alignment_result.is_valid:
       print("Contract alignment is valid")
   else:
       print(f"Alignment issues: {alignment_result.warnings}")

Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.validation.shared import ConfigurationValidator
   
   # Create configuration validator
   validator = ConfigurationValidator()
   
   # Validate configuration schema
   config_data = {
       "input_path": "s3://my-bucket/data/",
       "output_path": "s3://my-bucket/output/",
       "instance_type": "ml.m5.xlarge",
       "hyperparameters": {
           "max_depth": 6,
           "eta": 0.3
       }
   }
   
   validation_result = validator.validate_config(
       config_data,
       schema_type="XGBoostTrainingConfig"
   )
   
   if validation_result.is_valid:
       print("Configuration is valid")
   else:
       print(f"Configuration errors: {validation_result.errors}")

Batch Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.validation import ValidationSuite
   
   # Create comprehensive validation suite
   suite = ValidationSuite()
   
   # Add multiple validation checks
   suite.add_validator("alignment", UnifiedAlignmentTester())
   suite.add_validator("builders", UniversalStepBuilderTester())
   suite.add_validator("contracts", ContractAlignmentValidator())
   
   # Run all validations
   results = suite.run_all_validations()
   
   # Generate validation report
   report = suite.generate_report(results)
   print(report)

Custom Validation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.validation.shared import ValidationBase, ValidationResult
   
   class CustomValidator(ValidationBase):
       """Custom validation logic for specific requirements."""
       
       def validate(self, target, **kwargs) -> ValidationResult:
           """Implement custom validation logic."""
           errors = []
           warnings = []
           
           # Custom validation logic here
           if not self._check_custom_requirement(target):
               errors.append("Custom requirement not met")
           
           return ValidationResult(
               is_valid=len(errors) == 0,
               errors=errors,
               warnings=warnings
           )
       
       def _check_custom_requirement(self, target) -> bool:
           """Custom validation check."""
           # Implementation details
           return True
   
   # Use custom validator
   validator = CustomValidator()
   result = validator.validate(my_target)

Integration with CI/CD
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.validation import ValidationPipeline
   
   # Create validation pipeline for CI/CD
   pipeline = ValidationPipeline()
   
   # Configure validation stages
   pipeline.add_stage("syntax", syntax_validators)
   pipeline.add_stage("alignment", alignment_validators)
   pipeline.add_stage("integration", integration_validators)
   
   # Run validation pipeline
   pipeline_result = pipeline.run()
   
   # Exit with appropriate code for CI/CD
   if not pipeline_result.is_valid:
       print(f"Validation failed: {pipeline_result.summary}")
       exit(1)
   else:
       print("All validations passed!")
       exit(0)

See Also
--------

- :doc:`../guides/developer_guide/validation_framework_guide` - Comprehensive validation guide
- :doc:`../guides/developer_guide/validation_checklist` - Validation checklist for developers
- :doc:`steps` - Step builders and configurations
- :doc:`registry` - Registry system integration
- :doc:`../design/validation_framework` - Validation framework architecture
