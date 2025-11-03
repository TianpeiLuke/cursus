"""
Bedrock-Enhanced Pipeline DAGs

This package contains shared DAG definitions for pipelines that integrate
AWS Bedrock LLM capabilities for data enhancement and processing.

Available DAGs:
- bedrock_pytorch_e2e_dag: Complete end-to-end pipeline with Bedrock enhancement
- bedrock_simple_training_dag: Simple training pipeline with Bedrock enhancement
"""

from .bedrock_pytorch_e2e_dag import (
    create_bedrock_pytorch_e2e_dag,
    get_dag_metadata as get_bedrock_pytorch_e2e_metadata,
    validate_dag_structure as validate_bedrock_pytorch_e2e_structure,
)

from .bedrock_simple_training_dag import (
    create_bedrock_simple_training_dag,
    get_dag_metadata as get_bedrock_simple_training_metadata,
    validate_dag_structure as validate_bedrock_simple_training_structure,
)

__all__ = [
    # Bedrock PyTorch E2E DAG
    "create_bedrock_pytorch_e2e_dag",
    "get_bedrock_pytorch_e2e_metadata",
    "validate_bedrock_pytorch_e2e_structure",
    
    # Bedrock Simple Training DAG
    "create_bedrock_simple_training_dag",
    "get_bedrock_simple_training_metadata",
    "validate_bedrock_simple_training_structure",
]
