"""
Bedrock-Enhanced Pipeline DAGs

This package contains shared DAG definitions for pipelines that integrate
AWS Bedrock LLM capabilities for data enhancement and processing.

Available DAGs:

Real-time Processing:
- bedrock_pytorch_e2e_dag: E2E pipeline with Bedrock real-time processing
- bedrock_pytorch_with_label_ruleset_e2e_dag: E2E pipeline with Bedrock real-time + label ruleset

Batch Processing (Cost-Optimized):
- bedrock_batch_pytorch_e2e_dag: E2E pipeline with Bedrock batch processing
- bedrock_batch_pytorch_with_label_ruleset_e2e_dag: E2E pipeline with Bedrock batch + label ruleset

Simple Training:
- bedrock_simple_training_dag: Simple training pipeline with Bedrock enhancement
"""

# Real-time processing DAGs
from .bedrock_pytorch_e2e_dag import (
    create_bedrock_pytorch_e2e_dag,
    get_dag_metadata as get_bedrock_pytorch_e2e_metadata,
)

from .bedrock_pytorch_with_label_ruleset_e2e_dag import (
    create_bedrock_pytorch_with_label_ruleset_e2e_dag,
    get_dag_metadata as get_bedrock_pytorch_with_label_ruleset_e2e_metadata,
)

# Batch processing DAGs
from .bedrock_batch_pytorch_e2e_dag import (
    create_bedrock_batch_pytorch_e2e_dag,
    get_dag_metadata as get_bedrock_batch_pytorch_e2e_metadata,
    validate_dag_structure as validate_bedrock_batch_pytorch_e2e_structure,
)

from .bedrock_batch_pytorch_with_label_ruleset_e2e_dag import (
    create_bedrock_batch_pytorch_with_label_ruleset_e2e_dag,
    get_dag_metadata as get_bedrock_batch_pytorch_with_label_ruleset_e2e_metadata,
    validate_dag_structure as validate_bedrock_batch_pytorch_with_label_ruleset_e2e_structure,
)

# Simple training DAG
from .bedrock_simple_training_dag import (
    create_bedrock_simple_training_dag,
    get_dag_metadata as get_bedrock_simple_training_metadata,
    validate_dag_structure as validate_bedrock_simple_training_structure,
)

__all__ = [
    # Bedrock PyTorch E2E DAG (Real-time)
    "create_bedrock_pytorch_e2e_dag",
    "get_bedrock_pytorch_e2e_metadata",
    # Bedrock PyTorch with Label Ruleset E2E DAG (Real-time)
    "create_bedrock_pytorch_with_label_ruleset_e2e_dag",
    "get_bedrock_pytorch_with_label_ruleset_e2e_metadata",
    # Bedrock Batch PyTorch E2E DAG
    "create_bedrock_batch_pytorch_e2e_dag",
    "get_bedrock_batch_pytorch_e2e_metadata",
    "validate_bedrock_batch_pytorch_e2e_structure",
    # Bedrock Batch PyTorch with Label Ruleset E2E DAG
    "create_bedrock_batch_pytorch_with_label_ruleset_e2e_dag",
    "get_bedrock_batch_pytorch_with_label_ruleset_e2e_metadata",
    "validate_bedrock_batch_pytorch_with_label_ruleset_e2e_structure",
    # Bedrock Simple Training DAG
    "create_bedrock_simple_training_dag",
    "get_bedrock_simple_training_metadata",
    "validate_bedrock_simple_training_structure",
]
