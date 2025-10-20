"""
Cursus Processing Module

This module provides access to various data processing utilities and processors
that can be used in preprocessing, inference, evaluation, and other ML pipeline steps.

The processors are organized by functionality:
- Base processor classes and composition utilities
- Text processing (tokenization, NLP)
- Numerical processing (imputation, binning)
- Categorical processing (label encoding)
- Domain-specific processors (BSM, risk tables, etc.)
"""

# Import base processor classes
from .processors import Processor, ComposedProcessor, IdentityProcessor

# Import specific processors
from .categorical.categorical_label_processor import CategoricalLabelProcessor
from .categorical.multiclass_label_processor import MultiClassLabelProcessor
from .numerical.numerical_imputation_processor import NumericalVariableImputationProcessor
from .numerical.numerical_binning_processor import NumericalBinningProcessor

# Import atomic processors
from .temporal.time_delta_processor import TimeDeltaProcessor
from .temporal.sequence_padding_processor import SequencePaddingProcessor
from .temporal.sequence_ordering_processor import SequenceOrderingProcessor
from .temporal.temporal_mask_processor import TemporalMaskProcessor
from .categorical.dictionary_encoding_processor import DictionaryEncodingProcessor
from .categorical.categorical_imputation_processor import CategoricalImputationProcessor
from .categorical.numerical_categorical_processor import NumericalCategoricalProcessor
from .categorical.categorical_validation_processor import CategoricalValidationProcessor
from .numerical.minmax_scaling_processor import MinMaxScalingProcessor
from .numerical.feature_normalization_processor import FeatureNormalizationProcessor

# Import text/NLP processors (with optional dependency handling)
try:
    from .nlp.bert_tokenize_processor import BertTokenizeProcessor
except ImportError:
    BertTokenizeProcessor = None

try:
    from .nlp.gensim_tokenize_processor import GensimTokenizeProcessor
except ImportError:
    GensimTokenizeProcessor = None

# Import domain-specific processors (with optional dependency handling)
try:
    from .nlp.bsm_processor import BSMProcessor
except ImportError:
    BSMProcessor = None

try:
    from .nlp.cs_processor import CSProcessor
except ImportError:
    CSProcessor = None

try:
    from .categorical.risk_table_processor import RiskTableProcessor
except ImportError:
    RiskTableProcessor = None

# Import data loading utilities (with optional dependency handling)
try:
    from .dataloaders.bsm_dataloader import BSMDataLoader
except ImportError:
    BSMDataLoader = None

try:
    from .datasets.bsm_datasets import BSMDatasets
except ImportError:
    BSMDatasets = None

# Export all available processors
__all__ = [
    # Base classes
    "Processor",
    "ComposedProcessor",
    "IdentityProcessor",
    # Core processors
    "CategoricalLabelProcessor",
    "MultiClassLabelProcessor",
    "NumericalVariableImputationProcessor",
    "NumericalBinningProcessor",
    # Atomic processors
    "TimeDeltaProcessor",
    "SequencePaddingProcessor",
    "SequenceOrderingProcessor",
    "TemporalMaskProcessor",
    "DictionaryEncodingProcessor",
    "CategoricalImputationProcessor",
    "NumericalCategoricalProcessor",
    "CategoricalValidationProcessor",
    "MinMaxScalingProcessor",
    "FeatureNormalizationProcessor",
]

# Add optional processors to __all__ if they're available
_optional_processors = [
    ("BertTokenizeProcessor", BertTokenizeProcessor),
    ("GensimTokenizeProcessor", GensimTokenizeProcessor),
    ("BSMProcessor", BSMProcessor),
    ("CSProcessor", CSProcessor),
    ("RiskTableProcessor", RiskTableProcessor),
    ("BSMDataLoader", BSMDataLoader),
    ("BSMDatasets", BSMDatasets),
]

for name, processor_class in _optional_processors:
    if processor_class is not None:
        __all__.append(name)
