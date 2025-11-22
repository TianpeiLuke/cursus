---
tags:
  - entry_point
  - documentation
  - model_architecture
  - machine_learning
  - design
  - overview
keywords:
  - xgboost
  - lightgbm
  - mtgbm
  - pytorch
  - pytorch lightning
  - multi-modal
  - trimodal
  - semi-supervised
  - temporal self-attention
  - model design
  - deep learning
topics:
  - model architecture design
  - algorithm implementation
  - multi-task learning
  - hybrid parallelism
  - model optimization
language: python
date of note: 2025-11-21
---

# Model Architecture Design Index

## Overview

This index card serves as the comprehensive navigation hub for model architecture designs and implementations in the Cursus framework. It catalogs design documents, analyses, and implementation patterns for various machine learning algorithms and model architectures used across prediction tasks.

## Quick Navigation

```
Model Architecture Designs
├── Tree-Based Models
│   ├── XGBoost Models
│   ├── LightGBM Models
│   └── Multi-Task GBDT (MTGBM/LightGBM-MT)
│
├── Deep Learning Models
│   ├── PyTorch Lightning Models
│   ├── Native PyTorch Models
│   ├── Temporal Self-Attention Models
│   └── Semi-Supervised Learning Models
│
└── Multi-Modal Architectures
    ├── Bimodal Models
    ├── Trimodal Models
    └── Fusion Mechanisms
```

---

## 1. Tree-Based Model Architectures

### 1.1 XGBoost Models

**Code Locations:**
- `projects/atoz_xgboost/docker/xgboost_training.py` - Main training implementation
- `projects/atoz_xgboost/docker/xgboost_inference.py` - Inference implementation
- `src/cursus/steps/specs/xgboost_training_spec.py` - Training specification
- `src/cursus/steps/builders/builder_xgboost_training_step.py` - Step builder

**Related Design Docs:**
- [XGBoost Model Inference Design](../1_design/xgboost_model_inference_design.md)
- [XGBoost Semi-Supervised Learning Training Design](../1_design/xgboost_semi_supervised_learning_training_design.md)
- [XGBoost Semi-Supervised Learning Pipeline Design](../1_design/xgboost_semi_supervised_learning_pipeline_design.md)

**Related Analysis:**
- [XGBoost Inference Latency Analysis](../4_analysis/xgboost_inference_latency_analysis.md)
- [XGBoost Inference Latency Optimization Analysis](../4_analysis/2025-11-20_xgboost_inference_latency_optimization_analysis.md)

**Key Features:**
- Gradient boosting decision trees
- Feature importance analysis
- Model calibration support
- Inference optimization

### 1.2 LightGBM Models

**Code Locations:**
- `projects/ab_lightgbm/docker/lightgbm_inference.py` - Inference implementation
- `src/cursus/steps/specs/lightgbm_training_spec.py` - Training specification
- `src/cursus/steps/configs/config_lightgbm_model_eval_step.py` - Evaluation config

**Related Design Docs:**
- [LightGBM Model Training Design](../1_design/lightgbm_model_training_design.md)
- [LightGBM Multi-Task Training Step Design](../1_design/lightgbm_multi_task_training_step_design.md)

**Key Features:**
- Histogram-based gradient boosting
- Categorical feature support
- Memory efficiency
- Fast training speed

### 1.3 Multi-Task GBDT (MTGBM)

**Code Locations:**
- `projects/cap_mtgbm/docker/models/` - Refactored MTGBM model implementations (base, factory, implementations, loss)
- `projects/cap_mtgbm/docker/lightgbmmt_inference.py` - MT inference
- `src/cursus/steps/specs/lightgbmmt_training_spec.py` - Training specification
- `projects/pfw_lightgbmmt_legacy/` - Legacy implementation

**Related Design Docs:**
- [MTGBM Multi-Task Learning Design](../1_design/mtgbm_multi_task_learning_design.md) - **PRIMARY** - Multi-task architecture
- [MTGBM Model Classes Refactoring Design](../1_design/mtgbm_model_classes_refactoring_design.md)
- [MTGBM Models Refactoring Design](../1_design/mtgbm_models_refactoring_design.md)
- [LightGBM-MT Model Inference Design](../1_design/lightgbmmt_model_inference_design.md)

**Key Features:**
- Multi-task learning framework
- Shared tree structures
- Task-specific leaf predictions
- Efficient multi-objective optimization

---

## 2. PyTorch Lightning Model Architectures

### 2.1 Text-Only Models

**Code Locations:**
- `projects/rnr_pytorch_bedrock/docker/lightning_models/text/pl_bert.py` - BERT implementation
- `projects/rnr_pytorch_bedrock/docker/lightning_models/text/pl_text_cnn.py` - TextCNN
- `projects/rnr_pytorch_bedrock/docker/lightning_models/text/pl_lstm.py` - LSTM

**Related Design Docs:**
- [PyTorch Model Evaluation Design](../1_design/pytorch_model_eval_design.md)

**Key Features:**
- Pre-trained transformer models
- Sequence modeling
- Text classification
- Fine-tuning strategies

### 2.2 Bimodal Models

**Code Locations:**
- `projects/rnr_pytorch_bedrock/docker/lightning_models/bimodal/pl_bimodal_bert.py` - BERT-based fusion
- `projects/rnr_pytorch_bedrock/docker/lightning_models/bimodal/pl_bimodal_cnn.py` - CNN-based fusion
- `projects/rnr_pytorch_bedrock/docker/lightning_models/bimodal/pl_bimodal_cross_attn.py` - Cross-attention
- `projects/rnr_pytorch_bedrock/docker/lightning_models/bimodal/pl_bimodal_gate_fusion.py` - Gated fusion
- `projects/rnr_pytorch_bedrock/docker/lightning_models/bimodal/pl_bimodal_moe.py` - Mixture of Experts

**Related Design Docs:**
- [PyTorch Model Evaluation Design](../1_design/pytorch_model_eval_design.md)

**Model Documentation:**
- [Bimodal BERT Model](../models/lightning_models/pl_bimodal_bert.md) - BERT + tabular concatenation fusion
- [Bimodal CNN Model](../models/lightning_models/pl_bimodal_cnn.md) - CNN + tabular fusion (faster, more efficient)
- [Bimodal Cross-Attention Model](../models/lightning_models/pl_bimodal_cross_attn.md) - Bidirectional cross-attention fusion
- [Bimodal Gate Fusion Model](../models/lightning_models/pl_bimodal_gate_fusion.md) - Adaptive gated fusion with learnable weights
- [Bimodal Mixture of Experts Model](../models/lightning_models/pl_bimodal_moe.md) - MoE with softmax-normalized expert routing

**Key Features:**
- Multi-modal feature fusion
- Cross-attention mechanisms
- Gated fusion networks
- Mixture of Experts architecture

### 2.3 Trimodal Models

**Code Locations:**
- `projects/rnr_pytorch_bedrock/docker/lightning_models/trimodal/pl_trimodal_bert.py` - BERT-based trimodal
- `projects/rnr_pytorch_bedrock/docker/lightning_models/trimodal/pl_trimodal_cross_attn.py` - Cross-attention trimodal
- `projects/rnr_pytorch_bedrock/docker/lightning_models/trimodal/pl_trimodal_gate_fusion.py` - Gated trimodal

**Related Design Docs:**
- [PyTorch Model Evaluation Design](../1_design/pytorch_model_eval_design.md)

**Model Documentation:**
- [Trimodal BERT Model](../models/lightning_models/pl_trimodal_bert.md) - BERT + BERT + tabular concatenation fusion (baseline)
- [Trimodal Cross-Attention Model](../models/lightning_models/pl_trimodal_cross_attn.md) - Bidirectional cross-attention between text modalities
- [Trimodal Gate Fusion Model](../models/lightning_models/pl_trimodal_gate_fusion.md) - Learnable gating for adaptive modality weighting

**Key Features:**
- Three-modal input processing
- Complex fusion strategies
- Hierarchical attention
- Multi-level feature extraction

### 2.4 PyTorch Lightning Training Utilities

**Code Locations:**
- `projects/rnr_pytorch_bedrock/docker/lightning_models/utils/pl_train.py` - Training utilities
- `projects/rnr_pytorch_bedrock/docker/lightning_models/utils/config_constants.py` - Configuration
- `src/cursus/steps/specs/pytorch_training_spec.py` - Training specification

**Related Design Docs:**
- [PyTorch Lightning Temporal Self-Attention Design](../1_design/pytorch_lightning_temporal_self_attention_design.md)

**Key Features:**
- Distributed training support
- Automatic mixed precision
- Model checkpointing
- TensorBoard integration

---

## 3. Native PyTorch Architectures

### 3.1 Native PyTorch Models

**Code Locations:**
- `projects/bsm_pytorch/docker/pytorch_training.py` - BSM training
- `projects/rnr_pytorch_bedrock/docker/pytorch_training.py` - RNR training
- `projects/rnr_pytorch_bedrock/docker/pytorch_model_eval.py` - Model evaluation

**Related Design Docs:**
- [Native PyTorch Hybrid Parallelism Implementation](../1_design/native_pytorch_hybrid_parallelism_implementation.md) - **PRIMARY** - Hybrid parallelism architecture
- [Native PyTorch Implementation Plan](../1_design/native_pytorch_implementation_plan.md)
- [Native PyTorch Migration Strategy](../1_design/native_pytorch_migration_strategy.md)

**Key Features:**
- Custom training loops
- Hybrid parallelism (data + model)
- Memory-efficient training
- Custom loss functions

### 3.2 Temporal Self-Attention Models

**Code Locations:**
- `projects/temporal_self_attention_pytorch/` - Current TSA implementation
- `projects/tsa/` - Legacy Time Series Attention models

**Related Design Docs:**
- [Temporal Self-Attention Model Design](../1_design/temporal_self_attention_model_design.md) - **PRIMARY** - TSA architecture
- [PyTorch Lightning Temporal Self-Attention Design](../1_design/pytorch_lightning_temporal_self_attention_design.md)

**Related Analysis:**
- [Temporal Self-Attention Scripts Analysis](../4_analysis/temporal_self_attention_scripts_analysis.md) - Comprehensive TSA implementation analysis
- [TSA Cursus Step Equivalency Analysis](../4_analysis/2025-10-20_tsa_cursus_step_equivalency_analysis.md) - TSA pipeline vs Cursus framework comparison
- [TSA SageMaker Pipeline DAG Analysis](../4_analysis/2025-10-20_tsa_sagemaker_pipeline_dag_analysis.md) - TSA pipeline DAG structure and execution

**Key Features:**
- Temporal attention mechanisms
- Sequence modeling
- Long-range dependencies
- Time-aware representations

---

## 4. Data Processing & Datasets

### 4.1 Dataset Implementations

**Code Locations:**
- `projects/bsm_pytorch/docker/processing/datasets/bsm_datasets.py` - BSM datasets
- `projects/rnr_pytorch_bedrock/docker/processing/datasets/bsm_datasets.py` - RNR datasets
- `projects/rnr_pytorch_bedrock/docker/processing/dataloaders/bsm_dataloader.py` - Custom dataloaders

**Related Design Docs:**
- [Multi-Sequence Feature Engineering Design](../1_design/multi_sequence_feature_engineering_design.md)
- [Multi-Sequence Preprocessing Design](../1_design/multi_sequence_preprocessing_design.md)
- [Temporal Sequence Normalization Design](../1_design/temporal_sequence_normalization_design.md)

**Key Features:**
- Custom PyTorch datasets
- Efficient data loading
- On-the-fly preprocessing
- Memory-mapped datasets

### 4.2 Data Validation

**Code Locations:**
- `projects/rnr_pytorch_bedrock/docker/processing/validation.py` - Input validation

**Related Design Docs:**
- [Multilabel Preprocessing Step Design](../1_design/multilabel_preprocessing_step_design.md)

**Key Features:**
- Schema validation
- Data quality checks
- Type enforcement
- Missing value handling

---

## 5. Model Hyperparameters

### 5.1 BSM Hyperparameters

**Code Locations:**
- `src/cursus/steps/hyperparams/hyperparameters_bsm.py` - BSM hyperparameter classes

**Related Design Docs:**
- [Hyperparameter Class Guide](../0_developer_guide/hyperparameter_class.md) - Base patterns

**Key Features:**
- Type-safe hyperparameter definitions
- Validation rules
- Default values
- Hyperparameter tuning support

### 5.2 Trimodal Hyperparameters

**Code Locations:**
- `src/cursus/steps/hyperparams/hyperparameters_trimodal.py` - Trimodal hyperparameters

**Related Design Docs:**
- [PyTorch Lightning Temporal Self-Attention Design](../1_design/pytorch_lightning_temporal_self_attention_design.md)

**Key Features:**
- Multi-modal specific parameters
- Fusion strategy configuration
- Architecture search space
- Learning rate scheduling

---

## 6. Model Calibration

### 6.1 Calibration Methods

**Code Locations:**
- `projects/atoz_xgboost/docker/scripts/model_calibration.py` - XGBoost calibration
- `src/cursus/steps/specs/model_calibration_spec.py` - Calibration specification
- `src/cursus/steps/specs/percentile_model_calibration_spec.py` - Percentile calibration

**Related Design Docs:**
- [Model Metrics Computation Design](../1_design/model_metrics_computation_design.md)

**Related Script Documentation:**
- [Model Calibration Script](../scripts/model_calibration_script.md) - Standard calibration implementation
- [Percentile Model Calibration Script](../scripts/percentile_model_calibration_script.md) - Percentile-based calibration
- [Model Metrics Computation Script](../scripts/model_metrics_computation_script.md) - Metrics computation

**Related Step Documentation:**
- [XGBoost Model Evaluation Step](../steps/xgboost_model_eval_step.md) - Model evaluation patterns

**Key Features:**
- Platt scaling
- Isotonic regression
- Percentile-based calibration
- Temperature scaling

---

## 7. Semi-Supervised Learning

**Code Locations:**
- *To be added when implemented*

**Related Design Docs:**
- [XGBoost Semi-Supervised Learning Training Design](../1_design/xgboost_semi_supervised_learning_training_design.md) - **PRIMARY** - Semi-supervised framework
- [XGBoost Semi-Supervised Learning Pipeline Design](../1_design/xgboost_semi_supervised_learning_pipeline_design.md)
- [Pseudo Label Merge Script Design](../1_design/pseudo_label_merge_script_design.md)

**Key Features:**
- Label propagation
- Self-training
- Co-training
- Consistency loss

---

## 8. Model Optimization & Performance

### 8.1 Inference Optimization

**Related Analysis:**
- [XGBoost Inference Latency Analysis](../4_analysis/xgboost_inference_latency_analysis.md)
- [XGBoost Inference Latency Optimization Analysis](../4_analysis/2025-11-20_xgboost_inference_latency_optimization_analysis.md)
- [Processor Optimization Summary](../4_analysis/processor_optimization_summary.md)

**Related Implementation Plans:**
- [Inference Latency Optimization Implementation Plan](../2_project_planning/2025-11-20_inference_latency_optimization_implementation_plan.md)

**Optimization Strategies:**
- Processor-level optimizations
- Model quantization
- Batching strategies
- Memory management

### 8.2 Training Optimization

**Related Design Docs:**
- [Native PyTorch Hybrid Parallelism Implementation](../1_design/native_pytorch_hybrid_parallelism_implementation.md)
- [PyTorch Lightning Temporal Self-Attention Design](../1_design/pytorch_lightning_temporal_self_attention_design.md)

**Optimization Strategies:**
- Distributed training
- Gradient checkpointing
- Learning rate scheduling
- Early stopping

---

## 9. Cross-Cutting Model Patterns

### 9.1 Multi-Task Learning Patterns

**Common Across:**
- MTGBM models
- Multi-output PyTorch models
- Shared representation learning

**Key Design Principles:**
- Task balancing
- Shared vs task-specific layers
- Loss weighting strategies
- Negative transfer mitigation

### 9.2 Attention Mechanisms

**Implementations:**
- Cross-attention (bimodal/trimodal)
- Self-attention (temporal)
- Multi-head attention
- Scaled dot-product attention

**Key Design Principles:**
- Query-key-value formulation
- Attention score computation
- Position encoding
- Dropout strategies

### 9.3 Fusion Mechanisms

**Types:**
- Early fusion (feature-level)
- Late fusion (decision-level)
- Intermediate fusion (hybrid)
- Gated fusion
- Attention-based fusion

**Key Design Principles:**
- Modality balance
- Gradient flow
- Information bottlenecks
- Interpretability

---

## 10. Pipeline Integration

### 10.1 Training Pipelines

**Code Locations:**
- `src/cursus/pipeline_catalog/shared_dags/xgboost/complete_e2e_with_testing_dag.py` - XGBoost E2E pipeline

**Related Design Docs:**
- [Pipeline Assembler](../1_design/pipeline_assembler.md)
- [Dynamic Template System](../1_design/dynamic_template_system.md)

**Integration Points:**
- Data preprocessing
- Model training
- Hyperparameter tuning
- Model evaluation
- Model registration

### 10.2 Inference Pipelines

**Code Locations:**
- Various `*_inference.py` files across projects

**Related Design Docs:**
- [XGBoost Model Inference Design](../1_design/xgboost_model_inference_design.md)
- [LightGBM-MT Model Inference Design](../1_design/lightgbmmt_model_inference_design.md)
- [Inference Handler Spec Design](../1_design/inference_handler_spec_design.md)

**Integration Points:**
- Input validation
- Feature engineering
- Batch inference
- Post-processing
- Output formatting

---

## 11. Project-Specific Implementations

### 11.1 AtoZ XGBoost Project

**Code Location:** `projects/atoz_xgboost/`

**Key Components:**
- XGBoost training
- Model calibration
- Risk table processing
- Numerical imputation

### 11.2 AB LightGBM Project

**Code Location:** `projects/ab_lightgbm/`

**Key Components:**
- LightGBM training
- Model inference
- Feature engineering

### 11.3 CAP MTGBM Project

**Code Location:** `projects/cap_mtgbm/`

**Key Components:**
- Multi-task GBDT
- LightGBM-MT inference
- Multi-objective optimization

### 11.4 RNR PyTorch Bedrock Project

**Code Location:** `projects/rnr_pytorch_bedrock/`

**Key Components:**
- Multi-modal models
- PyTorch Lightning training
- Bedrock integration
- Label ruleset execution

### 11.5 BSM PyTorch Project

**Code Location:** `projects/bsm_pytorch/`

**Key Components:**
- Native PyTorch training
- Custom datasets
- BSM-specific models

### 11.6 Temporal Self-Attention Project

**Code Location:** `projects/temporal_self_attention_pytorch/`, `projects/tsa/`

**Key Components:**
- Time series modeling
- Self-attention mechanisms
- Temporal patterns

---

## 12. Model Architecture Design Checklist

When creating a new model architecture design document, consider:

### Design Document Structure
- [ ] Model overview and motivation
- [ ] Architecture diagram
- [ ] Input/output specifications
- [ ] Layer-by-layer design
- [ ] Loss function design
- [ ] Training strategy
- [ ] Hyperparameter space
- [ ] Evaluation metrics
- [ ] Inference optimization

### Implementation Considerations
- [ ] Code organization
- [ ] Configuration management
- [ ] Checkpoint strategy
- [ ] Distributed training support
- [ ] Memory requirements
- [ ] Computational complexity
- [ ] Scalability considerations

### Integration Points
- [ ] Data preprocessing requirements
- [ ] Feature engineering needs
- [ ] Pipeline integration
- [ ] Monitoring and logging
- [ ] Model versioning
- [ ] A/B testing support

---

## 13. Future Model Architectures

**Planned Additions:**
- Graph neural networks
- Transformer-based architectures
- Reinforcement learning models
- Generative models
- Ensemble methods
- Meta-learning approaches

---

## 14. Related Entry Points

- [Cursus Package Overview](./cursus_package_overview.md) - Overall system architecture
- [Step Design and Documentation Index](./step_design_and_documentation_index.md) - Training/inference step patterns
- [Processing Steps Index](./processing_steps_index.md) - Data processing components
- [Core and MODS Systems Index](./core_and_mods_systems_index.md) - Pipeline orchestration

---

## 15. Quick Reference

### Common Model Selection Questions

**When to use XGBoost:**
- Tabular data
- Feature importance needed
- Smaller datasets
- CPU-based training

**When to use LightGBM:**
- Large-scale datasets
- Categorical features
- Memory constraints
- Fast iteration needed

**When to use MTGBM:**
- Multiple related prediction tasks
- Shared feature space
- Limited labeled data per task

**When to use PyTorch Lightning:**
- Complex neural architectures
- Multi-GPU training
- Experiment tracking
- Rapid prototyping

**When to use Native PyTorch:**
- Custom training loops
- Fine-grained control
- Novel architectures
- Research projects

**When to use Multi-Modal Models:**
- Multiple data types (text, image, tabular)
- Complementary information sources
- Cross-modal learning

---

## Maintenance Notes

**Last Updated:** 2025-11-21

**Update Triggers:**
- New model architecture implemented
- New design document added
- Performance optimization discovered
- New fusion mechanism developed
- Training strategy updated

**Maintenance Guidelines:**
- Add design docs as they are created
- Link to implementation code
- Track performance benchmarks
- Document architectural decisions
- Update project-specific sections
- Maintain cross-references

---

## Contributing

When adding a new model architecture:

1. Create design document in `slipbox/1_design/`
2. Add implementation in appropriate `projects/` directory
3. Update this index with links and descriptions
4. Add analysis documents to `slipbox/4_analysis/` if applicable
5. Create implementation plan in `slipbox/2_project_planning/` if needed
6. Update related entry points for cross-referencing
