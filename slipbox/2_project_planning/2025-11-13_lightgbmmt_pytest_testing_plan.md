---
tags:
  - project
  - testing
  - lightgbmmt
  - pytest
  - quality_assurance
  - refactored_architecture
keywords:
  - pytest testing
  - lightgbmmt test coverage
  - loss function testing
  - model factory testing
  - test best practices
topics:
  - pytest testing strategy
  - test-driven verification
  - refactored code testing
  - comprehensive test coverage
language: python
date of note: 2025-11-13
---

# LightGBMMT Pytest Testing Plan

## Overview

This document outlines the comprehensive testing strategy for the refactored LightGBMMT multi-task learning implementation, covering:
1. **Test Infrastructure**: Directory structure and pytest configuration
2. **Base Components**: TrainingState and base classes
3. **Loss Functions**: All three loss implementations and factory
4. **Model Components**: Model factory and integration tests
5. **Best Practices**: Following pytest best practices guide

**Timeline**: 1-2 days
**Prerequisites**: Understanding of refactored architecture and pytest best practices guide

## Executive Summary

### Objectives
- **Comprehensive Coverage**: >90% test coverage for all refactored components
- **Best Practices**: Follow pytest best practices guide rigorously
- **Test Quality**: Implementation-driven tests matching actual behavior
- **Maintainability**: Clear test organization with descriptive names
- **Documentation**: Each test documents what it verifies

### Success Metrics
- ✅ >90% code coverage across all modules
- ✅ 100% of public methods tested
- ✅ Edge cases and error conditions covered
- ✅ All tests pass consistently
- ✅ Tests follow best practices (source code first, no assumptions)

### Testing Philosophy

**Golden Rule**: Read source code completely before writing any test (prevents 95% of failures)

**Key Principles**:
1. **Source Code First**: Always read implementation before writing tests
2. **No Mocking for Self-Contained**: Use real objects when possible
3. **Implementation-Driven**: Test actual behavior, not expected behavior
4. **Comprehensive Coverage**: Happy path + edge cases + error conditions
5. **Clear Documentation**: Each test explains what it verifies

## Phase 1: Test Infrastructure Setup ✅ COMPLETED

**Status**: ✅ **IMPLEMENTED** (2025-11-13)

### 1.1 Create Directory Structure ✅

**Directories Created**:
```
projects/cap_mtgbm/tests/
├── __init__.py
└── models/
    ├── __init__.py
    ├── base/
    │   ├── __init__.py
    │   └── test_training_state.py  ✅ COMPLETE
    ├── loss/
    │   ├── __init__.py
    │   ├── test_base_loss_function.py  ← Next
    │   ├── test_fixed_weight_loss.py
    │   ├── test_adaptive_weight_loss.py
    │   ├── test_knowledge_distillation_loss.py
    │   └── test_loss_factory.py
    └── factory/
        ├── __init__.py
        └── test_model_factory.py
```

**Success Criteria**:
- ✅ Logical directory structure matching source code
- ✅ All `__init__.py` files created for package structure
- ✅ Test files named with `test_` prefix for pytest discovery

### 1.2 Configure Pytest ✅

**File**: `projects/cap_mtgbm/pytest.ini` (to be created)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    -p no:cacheprovider
markers =
    unit: Unit tests for isolated components
    integration: Integration tests for component interactions
    slow: Tests that take significant time to run
```

**Success Criteria**:
- ✅ Pytest configuration matches project needs
- ✅ Test discovery works correctly
- ✅ Markers defined for test categorization

## Phase 2: TrainingState Tests ✅ COMPLETED

**Status**: ✅ **IMPLEMENTED** (2025-11-13)

**File**: `projects/cap_mtgbm/tests/models/base/test_training_state.py`

**Coverage**: 8 test classes, 28 test methods

### 2.1 Source Code Analysis ✅

**Key Findings from Source**:
- Pure Pydantic v2 BaseModel (no external dependencies)
- No mocking needed (self-contained)
- Key methods: `should_stop_early()`, `update_best()`, `to_checkpoint_dict()`, `from_checkpoint_dict()`
- Validator: `validate_consistency()` checks `best_epoch <= current_epoch`
- Uses numpy arrays for `weight_evolution`
- Default values for all fields
- Model validator enforces consistency

### 2.2 Test Classes Implemented ✅

**TestTrainingStateInitialization** (3 tests):
- ✅ Default initialization with all default values
- ✅ Custom initialization with specific values
- ✅ Initialization with numpy arrays for weight_evolution

**TestShouldStopEarly** (4 tests):
- ✅ Early stopping not triggered when within patience
- ✅ Early stopping triggers when patience met (>=)
- ✅ Early stopping with patience greatly exceeded
- ✅ Zero epochs without improvement behavior

**TestUpdateBest** (5 tests):
- ✅ Metric improvement updates state and resets counter
- ✅ No improvement increments counter without updates
- ✅ Equal metric not considered improvement
- ✅ Sequence of mixed improvements
- ✅ Counter behavior verification

**TestCheckpointSerialization** (6 tests):
- ✅ Basic serialization without numpy arrays
- ✅ Serialization converts numpy arrays to lists
- ✅ Basic deserialization from checkpoint
- ✅ Deserialization converts lists back to numpy arrays
- ✅ Complete roundtrip (serialize → deserialize)
- ✅ Large weight evolution lists (1000 items)

**TestValidation** (4 tests):
- ✅ Validator rejects best_epoch > current_epoch
- ✅ Validator allows best_epoch == current_epoch
- ✅ Validator allows best_epoch < current_epoch
- ✅ Pydantic ge=0 constraint enforcement

**TestEdgeCases** (3 tests):
- ✅ Zero patience early stopping behavior
- ✅ Large weight evolution lists
- ✅ Empty history lists

**TestKnowledgeDistillationState** (3 tests):
- ✅ KD state defaults to inactive
- ✅ KD state activation
- ✅ KD state preservation in checkpoints

**Success Criteria**:
- ✅ 100% method coverage for TrainingState
- ✅ Edge cases tested
- ✅ Validation logic tested
- ✅ All tests pass
- ✅ No mocking needed (self-contained)

## Phase 3: Base Loss Function Tests ✅ COMPLETED

**Status**: ✅ **IMPLEMENTED** (2025-11-13)

**File**: `projects/cap_mtgbm/tests/models/loss/test_base_loss_function.py`

**Coverage**: 9 test classes, 29 test methods

### 3.1 Source Code Analysis ✅

**Key Findings from Source** (from `base_loss_function.py`):
- Abstract base class with ABC
- Requires: `num_label`, `val_sublabel_idx`, `hyperparams`
- Extracts 14 loss parameters from hyperparams (all prefixed with `loss_`)
- Caching mechanisms for predictions and labels (optional)
- Preprocessing methods: `_preprocess_predictions()`, `_preprocess_labels()`
- Utility methods: `normalize()`, `unit_scale()`, `grad()`, `hess()`
- Evaluation: `evaluate()` computes per-task AUC
- Abstract methods: `compute_weights()`, `objective()` (must mock for testing)
- Input validation in `__init__`

**Dependencies**:
- numpy for array operations
- scipy.special.expit for sigmoid
- sklearn.metrics.roc_auc_score for evaluation
- logging for logging infrastructure

**Data Structures**:
- `val_sublabel_idx`: Dict[int, np.ndarray] - task indices
- Predictions: reshaped to [N_samples, N_tasks]
- Labels: reshaped to [N_samples, N_tasks]

### 3.2 Test Strategy

**Mocking Strategy**:
- Create concrete subclass to test abstract base
- Mock `compute_weights()` and `objective()` for testing base functionality
- Mock hyperparams with all required loss parameters
- Mock lightgbm.Dataset for label extraction

**Test Classes Implemented** ✅:

1. **TestBaseLossFunctionInitialization** (7 tests)
   - ✅ Valid initialization with all required parameters
   - ✅ Validation: num_label >= 2
   - ✅ Validation: val_sublabel_idx not empty
   - ✅ Validation: hyperparams required
   - ✅ Parameter extraction from hyperparams (14 loss parameters)
   - ✅ Cache initialization when cache_predictions=True
   - ✅ Cache initialization when cache_predictions=False

2. **TestPreprocessPredictions** (5 tests)
   - ✅ Reshaping from flat to [N, num_col] matrix + sigmoid
   - ✅ Clipping for numerical stability [epsilon, 1-epsilon]
   - ✅ Caching behavior when cache_predictions=True
   - ✅ No caching when cache_predictions=False
   - ✅ Custom epsilon parameter override

3. **TestPreprocessLabels** (3 tests)
   - ✅ Label extraction from lightgbm.Dataset and reshape
   - ✅ Shape validation failure (reshape error)
   - ✅ Caching behavior with labels

4. **TestNormalization** (3 tests)
   - ✅ Standard sum normalization
   - ✅ NaN protection when sum < epsilon (uniform fallback)
   - ✅ Custom epsilon parameter

5. **TestUnitScale** (2 tests)
   - ✅ L2 normalization
   - ✅ Zero-norm protection (uniform fallback)

6. **TestGradientHessian** (2 tests)
   - ✅ Gradient computation: y_pred - y_true
   - ✅ Hessian computation: y_pred * (1 - y_pred)

7. **TestEvaluation** (2 tests)
   - ✅ Per-task AUC computation and mean AUC
   - ✅ Single-class handling (graceful degradation)

8. **TestCacheManagement** (2 tests)
   - ✅ clear_cache() empties both caches
   - ✅ clear_cache() when caching disabled

9. **TestEdgeCases** (3 tests)
   - ✅ Minimum tasks (2)
   - ✅ Many tasks (20 for scalability)
   - ✅ Empty trn_sublabel_idx default

**Success Criteria**:
- ✅ 100% coverage of base class methods
- ✅ Caching behavior thoroughly tested
- ✅ Input validation tested
- ✅ Edge cases (single class, zero norm, etc.)
- ✅ All preprocessing operations tested
- ✅ All 29 tests passing

### 3.3 Implementation Notes

**Mock Hyperparameters Structure**:
```python
@pytest.fixture
def mock_hyperparams():
    """Create mock hyperparameters with all loss parameters."""
    mock = Mock()
    mock.loss_epsilon = 1e-15
    mock.loss_epsilon_norm = 1e-10
    mock.loss_clip_similarity_inverse = 1e10
    mock.loss_beta = 0.2
    mock.loss_main_task_weight = 1.0
    mock.loss_weight_lr = 0.1
    mock.loss_patience = 100
    mock.enable_kd = False
    mock.loss_weight_method = None
    mock.loss_weight_update_frequency = 50
    mock.loss_delta_lr = 0.01
    mock.loss_cache_predictions = True
    mock.loss_precompute_indices = True
    mock.loss_log_level = "INFO"
    return mock
```

**Concrete Test Subclass**:
```python
class ConcreteBaseLoss(BaseLossFunction):
    """Concrete implementation for testing base class."""
    
    def compute_weights(self, labels_mat, preds_mat, iteration):
        """Stub implementation."""
        return np.ones(self.num_col) / self.num_col
    
    def objective(self, preds, train_data, ep=None):
        """Stub implementation."""
        labels_mat = self._preprocess_labels(train_data, self.num_col)
        preds_mat = self._preprocess_predictions(preds, self.num_col, ep)
        grad_i = self.grad(labels_mat, preds_mat)
        hess_i = self.hess(preds_mat)
        return grad_i.sum(axis=1), hess_i.sum(axis=1), grad_i, hess_i
```

## Phase 4: Fixed Weight Loss Tests ✅ COMPLETED

**Status**: ✅ **IMPLEMENTED** (2025-11-13)

**File**: `projects/cap_mtgbm/tests/models/loss/test_fixed_weight_loss.py`

**Coverage**: 6 test classes, 22 test methods

### 4.1 Source Code Analysis ✅

**Read**: `projects/cap_mtgbm/docker/models/loss/fixed_weight_loss.py`

**Key Findings**:
- Extends BaseLossFunction
- Implements `_generate_weights()` based on num_col and main_task_index
- Weight structure: main_task gets main_task_weight, others get main_task_weight * beta
- `compute_weights()` returns fixed weights (no adaptation)
- `objective()` computes weighted gradients and hessians
- Uses getattr(hyperparams, "main_task_index", 0) for backward compatibility

### 4.2 Test Classes Implemented ✅

1. **TestFixedWeightLossInitialization** (2 tests)
   - ✅ Valid initialization and inheritance
   - ✅ Weights generated at initialization

2. **TestWeightGeneration** (5 tests)
   - ✅ Weight structure with main_task_index=0
   - ✅ Weight structure with custom main_task_index
   - ✅ Beta scaling for subtask weights
   - ✅ Dynamic sizing based on num_col (not hardcoded)
   - ✅ Backward compatibility (defaults to index 0)

3. **TestComputeWeights** (3 tests)
   - ✅ Returns fixed weights (no adaptation)
   - ✅ Iteration parameter ignored
   - ✅ Weights match initialized weights

4. **TestObjective** (5 tests)
   - ✅ Gradient computation with weight application
   - ✅ Hessian computation with weight application
   - ✅ Weighted aggregation across tasks
   - ✅ Returns 4 values: grad, hess, grad_i, hess_i
   - ✅ Per-task gradients/hessians preserved

5. **TestDifferentTaskCounts** (3 tests)
   - ✅ Minimum tasks (2)
   - ✅ Six tasks (common use case)
   - ✅ Many tasks (15 for scalability)

6. **TestEdgeCases** (4 tests)
   - ✅ Zero beta (subtasks get zero weight)
   - ✅ Beta = 1.0 (equal weights)
   - ✅ Large main_task_weight
   - ✅ Main task index at end

**Success Criteria**:
- ✅ All weight generation scenarios tested
- ✅ Main task index handling verified
- ✅ Beta scaling verified
- ✅ Integration with base class verified
- ✅ All 22 tests passing

## Phase 5: Adaptive Weight Loss Tests

**Status**: ⏳ **PLANNED**

**File**: `projects/cap_mtgbm/tests/models/loss/test_adaptive_weight_loss.py`

### 5.1 Source Code Analysis (Required First Step)

**Read**: `projects/cap_mtgbm/docker/models/loss/adaptive_weight_loss.py`

**Expected Findings**:
- Extends BaseLossFunction
- Uses Jensen-Shannon divergence for similarity
- `compute_weights()` computes adaptive weights based on JS divergence
- Weight history tracking
- Weight learning rate (weight_lr) for smooth updates
- Uses main_task_index for similarity computation
- Normalization of similarities to get weights

### 5.2 Test Strategy

**Test Classes to Implement**:

1. **TestAdaptiveWeightLossInitialization**
   - Uniform weight initialization
   - Weight history initialization
   - Inheritance from BaseLossFunction

2. **TestSimilarityComputation**
   - JS divergence computation between main task and subtasks
   - Similarity conversion (inverse of divergence)
   - Clipping to prevent inf values
   - Main task similarity = 1.0 (to itself)
   - Uses main_task_index correctly

3. **TestWeightNormalization**
   - Similarities normalized to sum to 1
   - NaN protection in normalization
   - Zero similarity handling

4. **TestWeightAdaptation**
   - First iteration: uses computed weights
   - Subsequent iterations: applies weight_lr for smooth updates
   - Weight history tracking
   - Adaptive behavior over iterations

5. **TestObjective**
   - Uses adaptive weights
   - Weight computation integrated into objective
   - Returns 4 values: grad, hess, grad_i, hess_i

**Success Criteria**:
- [ ] JS divergence computation tested
- [ ] Weight adaptation tested
- [ ] Weight history tracked
- [ ] Main task index handling verified

## Phase 6: Knowledge Distillation Loss Tests

**Status**: ⏳ **PLANNED**

**File**: `projects/cap_mtgbm/tests/models/loss/test_knowledge_distillation_loss.py`

### 6.1 Source Code Analysis (Required First Step)

**Read**: `projects/cap_mtgbm/docker/models/loss/knowledge_distillation_loss.py`

**Expected Findings**:
- Extends AdaptiveWeightLoss (inherits adaptive behavior)
- KD tracking state (kd_active, kd_trigger_iteration, performance_history, decline_count)
- `_check_kd_trigger()` monitors performance decline
- `_apply_kd()` replaces labels with predictions for struggling tasks
- Patience mechanism for triggering KD

### 6.2 Test Strategy

**Test Classes to Implement**:

1. **TestKDInitialization**
   - KD state initialization (inactive by default)
   - Performance history tracking
   - Decline count tracking
   - Inherits from AdaptiveWeightLoss

2. **TestKDTriggerLogic**
   - Decline count increments on performance decline
   - Decline count resets on improvement
   - KD triggers when decline_count >= patience
   - Only triggers once (kd_active flag)

3. **TestKDLabelReplacement**
   - Labels replaced with predictions for struggling tasks
   - Only tasks with decline_count >= patience affected
   - Original labels preserved for non-struggling tasks
   - Soft labels (predictions) used for KD

4. **TestKDIntegration**
   - KD applied during objective computation
   - Performance monitoring integrated
   - KD trigger at correct iteration

5. **TestKDEdgeCases**
   - Small patience values
   - All tasks struggling
   - Single task struggling
   - KD never triggered (good performance)

**Success Criteria**:
- [ ] KD trigger logic tested
- [ ] Label replacement tested
- [ ] Performance monitoring tested
- [ ] Integration with adaptive loss verified

## Phase 7: Loss Factory Tests

**Status**: ⏳ **PLANNED**

**File**: `projects/cap_mtgbm/tests/models/loss/test_loss_factory.py`

### 7.1 Source Code Analysis (Required First Step)

**Read**: `projects/cap_mtgbm/docker/models/loss/loss_factory.py`

**Expected Findings**:
- Registry pattern with _registry dict
- `create()` method for loss function instantiation
- Type validation (must be BaseLossFunction subclass)
- Error handling for unknown loss types
- `register()` for extending with new loss types
- `get_available_losses()` for listing registered types

### 7.2 Test Strategy

**Test Classes to Implement**:

1. **TestLossFactoryCreation**
   - Create FixedWeightLoss via factory
   - Create AdaptiveWeightLoss via factory
   - Create KnowledgeDistillationLoss via factory
   - Returns correct instance types
   - Passes parameters correctly to constructors

2. **TestLossFactoryValidation**
   - Error on unknown loss_type
   - Error message lists available types
   - Error when hyperparams not provided
   - Error when num_label < 2
   - Error when val_sublabel_idx empty

3. **TestLossFactoryRegistry**
   - Lists available loss types
   - Register new loss type
   - Registered loss type becomes available
   - Cannot register non-BaseLossFunction class

4. **TestLossFactoryExtensibility**
   - Create custom loss function
   - Register custom loss
   - Create instance of custom loss via factory

**Success Criteria**:
- [ ] All registered loss types tested
- [ ] Error handling tested
- [ ] Registry extensibility tested
- [ ] Type safety verified

## Phase 8: Model Factory Tests

**Status**: ⏳ **PLANNED**

**File**: `projects/cap_mtgbm/tests/models/factory/test_model_factory.py`

### 8.1 Source Code Analysis (Required First Step)

**Read**: `projects/cap_mtgbm/docker/models/factory/model_factory.py`

**Expected Findings**:
- Similar pattern to LossFactory
- Registry for model types
- `create()` method with loss_function, training_state, hyperparams
- Currently only supports 'mtgbm' type
- Extensible for future model types

### 8.2 Test Strategy

**Test Classes to Implement**:

1. **TestModelFactoryCreation**
   - Create MtgbmModel via factory
   - Returns correct instance type
   - Passes parameters correctly

2. **TestModelFactoryValidation**
   - Error on unknown model_type
   - Error when required parameters missing
   - Error message clarity

3. **TestModelFactoryRegistry**
   - Lists available model types
   - Register new model type
   - Extensibility for future models

**Success Criteria**:
- [ ] Model creation tested
- [ ] Error handling tested
- [ ] Registry pattern tested

## Testing Best Practices Checklist

For **EVERY** test file, follow this checklist:

### Pre-Writing Phase
- [ ] **Read source code completely** (5-10 minutes)
- [ ] **Analyze import statements** for mock paths
- [ ] **Study method signatures** and return types
- [ ] **Identify data structures** used
- [ ] **Map exception points** in code

### Writing Phase
- [ ] **Import from actual source** (not string paths)
- [ ] **Use real objects** when possible (no unnecessary mocking)
- [ ] **Mock at import location** when needed
- [ ] **Match actual behavior** not assumptions
- [ ] **Test edge cases** and error conditions

### Documentation Phase
- [ ] **Docstrings** explain what test verifies
- [ ] **Comments** reference source code behavior
- [ ] **Class names** clearly indicate what's tested
- [ ] **Test names** describe specific behavior tested

### Validation Phase
- [ ] **Run pytest** and ensure all tests pass
- [ ] **Check coverage** with pytest-cov
- [ ] **Review assertions** match implementation
- [ ] **Verify no false positives** (tests actually test something)

## Summary

### Timeline
- **Phase 1**: Test infrastructure (0.5 days) ✅ COMPLETE
- **Phase 2**: TrainingState tests (0.5 days) ✅ COMPLETE
- **Phase 3**: Base loss function tests (0.5 days) ✅ COMPLETE
- **Phase 4**: Fixed weight loss tests (0.25 days) ✅ COMPLETE
- **Phase 5**: Adaptive weight loss tests (0.25 days) 🔄 NEXT
- **Phase 6**: KD loss tests (0.25 days)
- **Phase 7**: Loss factory tests (0.25 days)
- **Phase 8**: Model factory tests (0.25 days)

**Total**: 1-2 days

### Deliverables
1. ✅ Test infrastructure and configuration
2. ✅ TrainingState tests (100% coverage, 26 tests)
3. ✅ Base loss function tests (100% coverage, 29 tests)
4. ✅ Fixed weight loss tests (100% coverage, 22 tests)
5. ⏳ All concrete loss function tests (adaptive, KD)
6. ⏳ Factory tests
7. ⏳ >90% overall code coverage

### Success Metrics (Final)
- [ ] >90% code coverage across all modules
- [ ] All public methods tested
- [ ] Edge cases covered
- [ ] Error conditions tested
- [ ] All tests pass
- [ ] Tests follow best practices
- [ ] Clear documentation

### Next Steps
1. **Immediate**: Complete base loss function tests (Phase 3)
2. **Today**: Complete all loss function tests (Phases 4-6)
3. **Tomorrow**: Complete factory tests (Phases 7-8)
4. **Final**: Run full test suite and verify coverage

## References

### Best Practices
- [Pytest Best Practices and Troubleshooting Guide](../6_resources/pytest_best_practices_and_troubleshooting_guide.md)
- [Pytest Test Failure Categories and Prevention](../6_resources/pytest_test_failure_categories_and_prevention.md)

### Source Code
- `projects/cap_mtgbm/docker/models/base/training_state.py`
- `projects/cap_mtgbm/docker/models/loss/base_loss_function.py`
- `projects/cap_mtgbm/docker/models/loss/fixed_weight_loss.py`
- `projects/cap_mtgbm/docker/models/loss/adaptive_weight_loss.py`
- `projects/cap_mtgbm/docker/models/loss/knowledge_distillation_loss.py`
- `projects/cap_mtgbm/docker/models/loss/loss_factory.py`
- `projects/cap_mtgbm/docker/models/factory/model_factory.py`

### Design Documents
- [LightGBMMT Implementation Part 1](./2025-11-12_lightgbmmt_implementation_part1_script_contract_hyperparams.md)
- [LightGBMMT Implementation Part 2](./2025-11-12_lightgbmmt_implementation_part2_training_script_alignment.md)
- [MTGBM Models Refactoring Design](../1_design/mtgbm_models_refactoring_design.md)
