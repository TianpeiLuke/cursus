# Simplified User Input for XGBoost Evaluation Pipeline

## Problem Statement

The current configuration process in `template_config_xgb_eval_v2.ipynb` requires users to specify a lengthy and complex set of configuration parameters across multiple categories. This process is:

1. Time-consuming: Users must navigate through numerous configuration sections
2. Error-prone: Complex interdependencies between parameters are difficult to track
3. Cognitively demanding: Users need to understand the entire pipeline structure to provide the correct inputs

## Current Configuration Requirements

The `template_config_xgb_eval_v2.ipynb` notebook requires users to specify:

1. **Base Hyperparameters**:
   - Region selection
   - Field lists (full, categorical, tabular)
   - Label field name and ID field
   - Binary/multi-class classification settings
   - Class weights and metrics

2. **XGBoost-Specific Hyperparameters**:
   - Training parameters
   - Objective function
   - Evaluation metrics

3. **Data Loading Configs**:
   - Cradle data source specifications (MDS, EDX)
   - Data schema definitions
   - Transform SQL statements
   - Date ranges for training and calibration
   - Output specifications

4. **Processing Configs**:
   - Instance types
   - Volume sizes
   - Processing counts

5. **Training Configs**:
   - Instance types
   - Entry points
   - Training volume settings

6. **Model Evaluation Configs**:
   - Evaluation metrics
   - Processing settings

7. **Packaging Configs**:
   - Processing settings

8. **MIMS Registration Configs**:
   - Model metadata
   - Input/output variables
   - Content and response types

9. **Payload Testing Configs**:
   - TPS requirements
   - Latency thresholds
   - Error rate tolerances

## Proposed Solution

### 1. Tiered Configuration Approach

Create a three-tier system to organize configurations by importance and frequency of customization:

#### Tier 1: Essential Configuration (Always Visible)
- Region selection
- Data timeframes (training, calibration)
- Feature field selection
- Target variable
- Model type (binary/multiclass)

#### Tier 2: Standard Configuration (Common Customization)
- XGBoost hyperparameters
- Instance type selection
- Evaluation metrics
- Class weights

#### Tier 3: Advanced Configuration (Expert-Level)
- SQL transformations
- Schema overrides
- Custom registration settings
- Detailed infrastructure parameters

### 2. Template-Based Configuration

Provide ready-to-use templates for common use cases:

- **Basic XGBoost Binary Classification**: Default settings optimized for binary classification
- **XGBoost Multi-Class Classification**: Settings adjusted for multi-class problems
- **XGBoost with Custom Evaluation**: Configuration for specialized evaluation metrics
- **XGBoost for High-Volume Data**: Settings optimized for large datasets

Templates would pre-populate all necessary parameters while allowing specific overrides.

### 3. Smart Defaults with Derivation Logic

Implement intelligent default generation with:

- **Automatic Parameter Derivation**: Auto-generate dependent parameters from essential inputs
- **Context-Aware Defaults**: Set reasonable defaults based on model type, region, etc.
- **Transparent Derivation**: Show users how derived values were calculated
- **Override Capability**: Allow users to override automatic derivations when needed

Examples:
- Automatically generate model_params based on is_binary setting
- Set appropriate instance types based on estimated data volume
- Derive appropriate class_weights from data distribution

### 4. Grouped Parameter Sections

Reorganize parameters into logical functional groups:

#### Data Configuration
- Region selection
- Data sources
- Date ranges
- Field selection

#### Model Configuration
- Model type
- Hyperparameters
- Training parameters
- Evaluation metrics

#### Infrastructure Configuration
- Instance types
- Volume sizes
- Processing settings

#### Deployment Configuration
- Registration parameters
- Payload testing settings
- Inference settings

### 5. Simplified User Interface

Create a more user-friendly configuration experience with:

- **Wizard-like Interface**: Step-by-step guided configuration
- **Interactive Field Selection**: Visual selection of fields from available options
- **Real-time Validation**: Immediate feedback on parameter validity
- **Config Preview**: Visual preview of the complete configuration

## Implementation Approach

### New Configuration Structure

```python
# Example of new simplified configuration structure
config = {
    # Essential parameters (Tier 1)
    "region": "NA",  # Region selection (NA, EU, FE)
    
    # Data parameters
    "data": {
        "training_period": {
            "start_date": "2025-01-01",
            "end_date": "2025-04-17"
        },
        "calibration_period": {
            "start_date": "2025-04-17",
            "end_date": "2025-04-28"
        },
        "sources": {
            "mds": {
                "service_name": "AtoZ",
                "org_id": 0
            },
            "edx": {
                "provider": "trms-abuse-analytics",
                "subject": "qingyuye-notr-exp",
                "dataset": "atoz-tag",
                "etl_job_id": "24292902"  # Region-specific, can be derived
            }
        }
    },
    
    # Model parameters
    "model": {
        "type": "binary_classification",  # or "multiclass_classification"
        "target_field": "is_abuse",
        "id_field": "order_id",
        "marketplace_field": "marketplace_id",
        "feature_groups": {
            "buyer_metrics": True,  # Enable/disable pre-defined feature groups
            "order_metrics": True,
            "refund_metrics": True,
            "message_metrics": True
        },
        "custom_features": {
            "numerical": [],  # Additional custom numerical fields
            "categorical": []  # Additional custom categorical fields
        },
        "hyperparameters": {
            "num_round": 300,
            "max_depth": 10,
            "min_child_weight": 1
            # Other XGBoost params with sensible defaults
        }
    },
    
    # Infrastructure (with smart defaults - Tier 2)
    "infrastructure": {
        "training_instance": "ml.m5.4xlarge",
        "processing_instance": "ml.m5.xlarge",
        "enable_advanced_settings": False  # Toggle to show/hide advanced options
    },
    
    # Deployment settings (mostly derived - Tier 3)
    "deployment": {
        "model_domain": "AtoZ",
        "model_objective": "AtoZ_Claims_SM_Model_NA",  # Auto-derived from region and service
        "expected_tps": 2,
        "max_latency_ms": 800
    }
}
```

### Configuration Generator Functions

Create transformation functions to convert the simplified configuration into the full configuration required by the pipeline:

1. **Feature Field Generator**: Generate complete field lists based on selected feature groups
2. **Schema Generator**: Generate MDS and EDX schemas based on field selections
3. **SQL Generator**: Generate appropriate SQL transformations based on field selections
4. **Hyperparameter Optimizer**: Set optimal hyperparameters based on data characteristics
5. **Registration Configurator**: Generate registration configs based on model settings

### User Interface Improvements

The new notebook would implement a more streamlined user experience:

1. **Template Selection**: First cell allows selection of a template
2. **Essential Parameter Form**: Second cell displays only Tier 1 parameters
3. **Preview & Expand**: Show derived parameters with option to customize
4. **Validation & Generation**: Validate inputs and generate the full configuration
5. **Pipeline Execution**: Direct integration with pipeline execution

## Technical Architecture

### Component Structure

1. **Configuration Schema**: Define Pydantic models for the simplified configuration
2. **Transformation Engine**: Convert simplified config to full pipeline config
3. **Template Repository**: Store and load configuration templates
4. **Validation Layer**: Ensure configuration validity and compatibility
5. **UI Components**: Jupyter widgets for interactive configuration

### Core Functions

```python
def load_template(template_name: str) -> Dict:
    """Load a predefined configuration template"""
    pass

def generate_field_lists(feature_groups: Dict[str, bool], custom_features: Dict[str, List]) -> Dict[str, List]:
    """Generate complete field lists based on selected feature groups"""
    pass

def derive_training_parameters(config: Dict) -> Dict:
    """Derive appropriate training parameters based on configuration"""
    pass

def validate_configuration(config: Dict) -> Tuple[bool, List[str]]:
    """Validate configuration and return validation status with error messages"""
    pass

def generate_full_config(simple_config: Dict) -> Dict:
    """Transform simplified configuration into full pipeline configuration"""
    pass
```

## Benefits

1. **Reduced Time**: Minimize the time needed to configure a pipeline
2. **Fewer Errors**: Reduce configuration errors through validation and derivation
3. **Knowledge Transfer**: Enable less experienced users to configure pipelines correctly
4. **Maintainability**: Separate configuration logic from pipeline execution
5. **Standardization**: Encourage use of best practices through templates
6. **Flexibility**: Maintain full customization capability for advanced users

## Migration Path

To ensure a smooth transition:

1. **Parallel Support**: Maintain both configuration approaches initially
2. **Conversion Tools**: Provide tools to convert between formats
3. **Documentation**: Create detailed documentation and examples
4. **Training**: Provide training for users on the new approach
5. **Feedback Loop**: Continuously improve based on user feedback

## User Experience Analysis

When considering the integration of this simplified approach with the existing config field categorization system, several options emerge, each with different user experience implications:

### Option 1: Extended Field Categorization Framework with Tiered Approach
**User Experience Rating: Good**

This solution organizes fields into essential, standard, and advanced tiers, which helps manage complexity. However, it's primarily a backend architecture change that doesn't directly translate to dramatic UX improvements without a corresponding UI.

**Benefits:**
- Cleaner field organization internally
- Provides basis for progressive disclosure in UI
- Maintains full configuration power

**Limitations:**
- Still requires custom UI to leverage effectively
- Field categorization alone doesn't reduce input complexity
- Technical implementation more than user-facing feature

### Option 2: Template System on Top of Type-Aware Serialization
**User Experience Rating: Very Good**

Templates provide a significant UX improvement by giving users pre-built starting points tailored to common use cases.

**Benefits:**
- "One-click" starting points reduce configuration time dramatically
- Users can start from working configurations rather than building from scratch
- Templates can evolve based on usage patterns and feedback
- Promotes best practices and standardization

**Limitations:**
- Quality depends on available templates
- May still require significant customization for specific needs
- Doesn't address progressive disclosure of complex parameters

### Option 3: Smart Defaults with Circular Reference Handling
**User Experience Rating: Excellent**

This approach provides the most direct improvement to day-to-day user experience by automatically deriving parameters and reducing the cognitive load of configuration.

**Benefits:**
- Dramatically reduces required user input (potentially by 70%+)
- Parameters are derived intelligently based on context
- Transparent derivation explains how values were determined
- Safe handling of dependencies prevents cascading errors
- Users still maintain override capability for full control

**Limitations:**
- Complex implementation with many rules to maintain
- Some users may find automatic derivation confusing
- Requires excellent documentation and visualization

### Option 4: Feature Groups with Job Type Variants
**User Experience Rating: Very Good**

This approach simplifies field selection through functional grouping and provides job-specific customizations.

**Benefits:**
- Conceptual grouping makes field selection more intuitive
- Reduces long field lists to manageable functional groups
- Job variants allow customization without duplication
- Aligns with user mental model of feature selection

**Limitations:**
- Still requires understanding of feature relationships
- May not reduce overall configuration complexity enough by itself
- Potential cognitive load from matrix of groups × job types

### Recommended Approach for Optimal User Experience

For the best user experience, **Option 3: Smart Defaults with Circular Reference Handling** provides the most significant improvement. It directly addresses the core pain point of configuration complexity by automatically deriving many parameters, while still providing transparency and control.

The ideal implementation would combine all four approaches:
1. Smart defaults as the foundation (Option 3)
2. Template system for quick starts (Option 2)
3. Feature groups for intuitive organization (Option 4)
4. Tiered categorization for progressive disclosure (Option 1)

This comprehensive approach would provide both immediate relief (through templates and smart defaults) and long-term usability (through organized field groups and tiered disclosure).

## Conclusion

This simplified user input approach for the XGBoost evaluation pipeline would significantly improve the user experience by reducing complexity while maintaining all necessary configuration capabilities. The tiered approach ensures that both novice and experienced users can effectively configure pipelines according to their needs. By implementing smart defaults with circular reference handling as the primary UX improvement, while also incorporating templates, feature groups, and tiered categorization, we can create an intuitive and powerful configuration experience that dramatically reduces the time and expertise required to create pipeline configurations.
