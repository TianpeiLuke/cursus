"""
PyTorch Shared DAG Definitions

This module contains shared DAG definitions for PyTorch-based pipelines.
"""

__all__ = [
    "create_pytorch_training_dag",
    "create_pytorch_standard_e2e_dag",
    "create_pytorch_complete_e2e_dag",
    "create_pytorch_complete_e2e_dummy_dag",
    "create_bedrock_pytorch_e2e_dag",
]

# Import functions to make them available at package level
try:
    from .training_dag import (
        create_pytorch_training_dag,
        get_dag_metadata as get_training_metadata,
    )
except ImportError:
    pass

try:
    from .standard_e2e_dag import (
        create_pytorch_standard_e2e_dag,
        get_dag_metadata as get_standard_e2e_metadata,
    )
except ImportError:
    pass

try:
    from .complete_e2e_dag import (
        create_pytorch_complete_e2e_dag,
        get_dag_metadata as get_complete_e2e_metadata,
    )
except ImportError:
    pass

try:
    from .complete_e2e_dummy_dag import (
        create_pytorch_complete_e2e_dummy_dag,
        get_dag_metadata as get_complete_e2e_dummy_metadata,
    )
except ImportError:
    pass

try:
    from .bedrock_pytorch_e2e_dag import (
        create_bedrock_pytorch_e2e_dag,
        get_dag_metadata as get_bedrock_pytorch_e2e_metadata,
    )
except ImportError:
    pass
