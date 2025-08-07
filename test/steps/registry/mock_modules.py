"""
Mock modules for testing.

This module provides mock implementations of external dependencies
that may not be available in the test environment.
"""

import sys
from unittest.mock import MagicMock

# Create a mock for secure_ai_sandbox_workflow_python_sdk
cradle_mock = MagicMock()
cradle_mock.cradle_data_loading = MagicMock()
cradle_mock.cradle_data_loading.cradle_data_loading_step = MagicMock()
cradle_mock.cradle_data_loading.cradle_data_loading_step.CradleDataLoadingStep = MagicMock()
cradle_mock.utils = MagicMock()
cradle_mock.utils.constants = MagicMock()

# Constants needed by the CradleDataLoadConfig
cradle_mock.utils.constants.OUTPUT_TYPE_DATA = "DATA"
cradle_mock.utils.constants.OUTPUT_TYPE_METADATA = "METADATA"
cradle_mock.utils.constants.OUTPUT_TYPE_SIGNATURE = "SIGNATURE"
cradle_mock.utils.constants.DEFAULT_OUTPUT_FORMAT = "CSV"
cradle_mock.utils.constants.DEFAULT_CLUSTER_TYPE = "STANDARD"
cradle_mock.utils.constants.DEFAULT_CRADLE_ACCOUNT = "playground"

# Register all the mock modules
sys.modules['secure_ai_sandbox_workflow_python_sdk'] = cradle_mock
sys.modules['secure_ai_sandbox_workflow_python_sdk.cradle_data_loading'] = cradle_mock.cradle_data_loading
sys.modules['secure_ai_sandbox_workflow_python_sdk.cradle_data_loading.cradle_data_loading_step'] = cradle_mock.cradle_data_loading.cradle_data_loading_step
sys.modules['secure_ai_sandbox_workflow_python_sdk.utils'] = cradle_mock.utils
sys.modules['secure_ai_sandbox_workflow_python_sdk.utils.constants'] = cradle_mock.utils.constants

# Create a mock for coral models
for model_name in [
    'field', 'datasource', 'mdsdatasourceproperties', 'edxdatasourceproperties',
    'andesdatasourceproperties', 'datasourcesspecification', 'jobsplitoptions',
    'transformspecification', 'outputspecification', 'cradlejobspecification',
    'createcradledataloadjobrequest'
]:
    module_path = f'com.amazon.secureaisandboxproxyservice.models.{model_name}'
    model_mock = MagicMock()
    sys.modules[module_path] = model_mock
    
    # Add the base path parts too to ensure imports work
    base_path = 'com.amazon.secureaisandboxproxyservice'
    sys.modules[base_path] = MagicMock()
    sys.modules[f'{base_path}.models'] = MagicMock()

# Create a mock for coral utils
utils_mock = MagicMock()
utils_mock.coral_utils = MagicMock()
sys.modules['secure_ai_sandbox_python_lib'] = utils_mock
sys.modules['secure_ai_sandbox_python_lib.utils'] = utils_mock
sys.modules['secure_ai_sandbox_python_lib.utils.coral_utils'] = utils_mock.coral_utils
