# Feature Group Registry (Revised)

## Overview

The Feature Group Registry is an essential component of the streamlined configuration approach that organizes fields into logical groups based on business concepts rather than technical implementation details. It works in conjunction with the three-tier configuration architecture implemented in `src/pipeline_steps/config_base.py` to enable users to select entire groups of related features instead of specifying individual fields, significantly simplifying the configuration process.

## Integration with Three-Tier Architecture

The Feature Group Registry complements the three-tier configuration architecture by:

1. **Simplifying Essential User Inputs (Tier 1)**: Instead of selecting individual fields, users select business-meaningful feature groups
2. **Working with System Defaults (Tier 2)**: Integrating with default feature selections for common scenarios
3. **Supporting Field Derivation (Tier 3)**: Providing field classifications needed for deriving other fields

## Component Design

### Core Classes

#### `FeatureGroup` Class

Represents a logical grouping of related features with the following attributes:

- `name`: Human-readable display name
- `description`: Description of what this group represents
- `fields`: List of field names in this group
- `category`: Broad category (e.g., "buyer", "order", "product")
- `region_specific`: Whether fields have region-specific naming
- `categorical_fields`: List of categorical field names

Key methods:
- `get_fields(region)`: Get fields with region-specific adjustments
- `is_field_categorical(field_name)`: Check if a field is categorical
- `get_categorical_fields(region)`: Get categorical fields with region-specific naming

#### `FeatureGroupRegistry` Class

Central registry managing feature group definitions with the following capabilities:

- Maintains a default set of feature groups
- Provides a list of required fields that should always be included
- Offers methods to get fields based on selected groups
- Supports categorical field identification and extraction
- Enables mapping fields back to their logical groups

Key methods:
- `get_feature_groups(region)`: Get all feature groups with regional adjustments
- `get_feature_group(group_id, region)`: Get a specific feature group
- `get_fields_for_groups(selected_groups, region)`: Get all fields for selected groups
- `get_categorical_fields(selected_groups, region)`: Get categorical fields for selected groups
- `map_fields_to_groups(field_list)`: Map fields back to their feature groups
- `register_feature_group(group_id, feature_group)`: Register a custom feature group
- `is_field_categorical(field_name)`: Check if a field is categorical

### Default Feature Group Definitions

The registry provides a comprehensive set of default feature groups:

1. **Buyer Profile**: General buyer profile and history metrics
2. **Order Metrics**: Order-related counts and amounts
3. **Claims History**: A-to-Z claims metrics and history
4. **Refund History**: Refund metrics and history
5. **DNR Metrics**: Did Not Receive claim metrics
6. **Buyer Actions**: Abuse-specific actions on buyer account
7. **Buyer-Seller Messaging**: Communication metrics between buyer and seller
8. **Shipping Data**: Shipping status and tracking information
9. **Current Claim**: Data about the current claim being evaluated

Each group contains a defined set of fields and categorical field designations.

### Region-Specific Field Handling

For fields with region-specific naming patterns, the registry uses template placeholders:

```python
f"Abuse.bsm_stats_for_evaluated_mfn_concessions_by_customer_{region}.n_total_message_count"
```

These placeholders are automatically replaced with the actual region code at runtime.

### Categorical Field Identification

Fields are identified as categorical through:

1. **Explicit listing**: Fields explicitly listed in each feature group's `categorical_fields`
2. **Pattern matching**: Fields matching specific naming patterns like those ending with `_status`, `_type`, etc.

### Required Fields

The registry defines a set of fields that should always be included regardless of feature group selection:

```python
REQUIRED_FIELDS = [
    "order_id",
    "marketplace_id",
    "is_abuse",
]
```

## Integration with Notebook Design

The revised Essential Inputs Notebook directly utilizes the Feature Group Registry to simplify the user experience:

```python
# Select feature groups instead of individual fields
from src.config_field_manager.feature_group_registry import FeatureGroupRegistry

# Get feature groups for the region
feature_groups = FeatureGroupRegistry.get_feature_groups("NA")

# Display feature groups for selection
display(feature_groups)

# Collect user selections
selected_groups = {
    "buyer_profile": True,
    "order_behavior": True,
    "refund_claims": True,
    "messages": True,
    "shipping": True,
    "abuse_patterns": True
}

# Get fields for selected groups
fields = FeatureGroupRegistry.get_fields_for_groups(selected_groups, "NA")
cat_fields = FeatureGroupRegistry.get_categorical_fields(selected_groups, "NA")
tab_fields = [f for f in fields if f not in cat_fields]

# Use fields in configuration
training_config = XGBoostTrainingConfig.from_base_config(
    base_config,
    num_round=300,
    max_depth=10,
    min_child_weight=1,
    is_binary=True,
    full_field_list=fields,
    cat_field_list=cat_fields,
    tab_field_list=tab_fields
)
```

This approach:
- Allows users to think in business terms rather than technical field names
- Reduces the cognitive load of selecting individual fields
- Ensures appropriate categorical/tabular field classification
- Maintains region-specific field naming consistency

## Integration with Configuration Classes

The Feature Group Registry complements the configuration classes by providing field lists that can be used to initialize configuration objects:

```python
class XGBoostTrainingConfig(BasePipelineConfig):
    """Training configuration for XGBoost models"""
    
    # Essential user inputs
    num_round: int = Field(..., description="Number of boosting rounds")
    max_depth: int = Field(..., description="Maximum tree depth")
    min_child_weight: int = Field(..., description="Minimum child weight")
    
    # Field lists (can be derived from feature groups)
    full_field_list: List[str] = Field(..., description="Complete list of fields")
    cat_field_list: List[str] = Field(..., description="Categorical fields")
    tab_field_list: List[str] = Field(..., description="Tabular fields")
    
    # System inputs with defaults
    objective: Optional[str] = Field(default=None, description="XGBoost objective function")
    eval_metric: Optional[List[str]] = Field(default=None, description="Evaluation metrics")
    
    # Derived fields
    _input_tab_dim: Optional[int] = PrivateAttr(default=None)
    
    @property
    def input_tab_dim(self) -> int:
        """Derive input dimension from tabular fields"""
        if self._input_tab_dim is None:
            self._input_tab_dim = len(self.tab_field_list)
        return self._input_tab_dim
```

The Feature Group Registry provides the field lists needed for these configuration classes.

## Implementation Considerations

### Business Concept Alignment

The Feature Group Registry is designed to align with business concepts:
- Groups are named after business-meaningful categories
- Descriptions explain the business significance
- Categories create a hierarchy of related concepts
- Field selection reflects business decision points

### Extensibility

The registry is designed for extensibility:
- New feature groups can be registered at runtime
- Custom field selection can complement predefined groups
- Pattern-based categorical detection can be expanded

### Performance

The registry uses efficient data structures:
- Dictionary-based lookups for feature groups
- Set operations for field filtering and comparison
- Cached region-specific field lists

### Error Handling

The registry provides robust error handling:
- Graceful handling of unknown feature groups
- Clear error messages for configuration issues
- Fallbacks for region-specific field templates

## User Experience Benefits

1. **Reduced Cognitive Load**: Users think in business terms rather than technical field names
2. **Simplified Selection**: 9 feature groups vs. 80+ individual fields
3. **Consistent Classification**: Automatic categorization of fields as categorical or tabular
4. **Region-Specific Handling**: Automatic adaptation to regional field naming patterns
5. **Clear Business Context**: Groups provide explanations of what features represent

## Conclusion

The Feature Group Registry provides a critical layer of business meaning on top of the technical configuration system. By organizing fields into logical groups based on business concepts, it significantly simplifies the user experience while ensuring proper field selection and classification.

When combined with the three-tier configuration architecture implemented in the configuration classes, the Feature Group Registry creates a powerful yet simple interface for users to express their requirements. This hybrid approach leverages the strengths of both systems:

1. The configuration classes provide robust field categorization, default values, and derivation logic
2. The Feature Group Registry provides business meaning and simplified selection

Together, these components create a configuration system that is both powerful and easy to use, reducing the cognitive load on users while maintaining full flexibility and control.
