"""
Unit tests for base specifications module.

This file serves as an entry point for running all base specification tests.
The tests have been split into multiple focused modules:

- test_dependency_spec.py: Tests for DependencySpec
- test_output_spec.py: Tests for OutputSpec
- test_step_specification.py: Tests for StepSpecification
- test_specification_registry_class.py: Tests for SpecificationRegistry
- test_enum_validation.py: Tests for enum validation
- test_pydantic_features.py: Tests for Pydantic features
- test_script_contract_integration.py: Tests for script contract integration
- test_step_specification_integration.py: Tests for StepSpecification integration
"""

import unittest
from test.pipeline_deps.test_dependency_spec import TestDependencySpec
from test.pipeline_deps.test_output_spec import TestOutputSpec
from test.pipeline_deps.test_step_specification import TestStepSpecification
from test.pipeline_deps.test_specification_registry_class import TestSpecificationRegistry
from test.pipeline_deps.test_enum_validation import TestEnumValidation
from test.pipeline_deps.test_pydantic_features import TestPydanticFeatures
from test.pipeline_deps.test_script_contract_integration import TestScriptContractIntegration
from test.pipeline_deps.test_step_specification_integration import TestStepSpecificationIntegration


if __name__ == '__main__':
    unittest.main()
