"""
Universal step builder test package.

This package contains test classes to validate step builder implementation compliance
at different levels of architectural requirements.
"""

from .base_test import UniversalStepBuilderTestBase
from .level1_interface_tests import InterfaceTests
from .level2_specification_tests import SpecificationTests
from .level3_path_mapping_tests import PathMappingTests
from .level4_integration_tests import IntegrationTests
from .universal_tests import UniversalStepBuilderTest
