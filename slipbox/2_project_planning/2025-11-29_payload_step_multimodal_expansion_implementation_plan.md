---
tags:
  - project
  - implementation
  - payload_generation
  - multimodal
  - pytorch
  - inference_testing
  - latency_testing
keywords:
  - payload generation
  - multimodal models
  - bimodal inference
  - trimodal inference
  - custom payload
  - text field defaults
  - inference testing
topics:
  - payload step expansion
  - multimodal payload generation
  - custom payload input support
  - inference testing optimization
language: python
date of note: 2025-11-29
---

# Payload Step Multi-Modal Expansion Implementation Plan

## Overview

This document covers the comprehensive implementation plan for expanding the MIMS payload generation step to support:
1. **Multi-modal models** (bimodal text+tabular, trimodal dual-text+tabular)
2. **Optional custom payload input** (user-provided samples via S3 or local path)
3. **Dict-based text field configuration** (flexible text sample customization)

**Timeline**: 2-3 weeks
**Prerequisites**: Understanding of payload step architecture, hyperparameter structure, multi-modal model inference

## Executive Summary

### Objectives
- **Multi-Modal Support**: Auto-detect and generate payloads for tabular, bimodal, and trimodal models
- **Custom Payload Input**: Allow users to provide their own payload samples (JSON/CSV)
- **Flexible Text Configuration**: Support dict-based text field value specification
- **Backward Compatibility**: Zero breaking changes for existing tabular-only usage
- **Maintain Architecture**: Follow specification-driven design (contract → spec → config → builder → script)

### Success Metrics
- ✅ Auto-detection of model type from hyperparameters (model_class field)
- ✅ Correct payload generation for bimodal models (single text + tabular)
- ✅ Correct payload generation for trimodal models (dual text + tabular)
- ✅ Optional custom payload input via S3/local path
- ✅ Dict-based text field defaults (TEXT_FIELD_DEFAULTS env var)
- ✅ Backward compatibility with existing tabular-only models
- ✅ Zero API changes to existing configs/builders

### Problem Statement

**Current State**:
- Payload script only generates samples for simple tabular models
- Extracts field lists from hyperparameters: `tab_field_list`, `cat_field_list`
- Generates basic key-value pairs with numeric/text defaults
- Cannot handle text fields requiring raw text (for tokenization)
- No support for multi-modal models with multiple text fields

**Requirements**:
- PyTorch multi-modal models need raw text inputs (not tokenized)
- Bimodal: Single text field (e.g., `chat`) + tabular features
- Trimodal: Dual text fields (e.g., `chat` + `shiptrack`) + tabular features
- Users want to provide custom payload samples for domain-specific testing
- Need flexible text sample configuration per field

**Solution**: Extend payload generation with model type detection and text field support

---

## Current Architecture Analysis

### Payload Step Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  Contract   │ --> │     Spec     │ --> │   Config    │ --> │   Builder    │ --> │   Script   │
│ (input/out) │     │ (deps/outs)  │     │ (user cfg)  │     │ (wire step)  │     │ (generate) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘     └────────────┘
```

### Current Files

1. **Contract** (`payload_contract.py`):
   - Entry point: `payload.py`
   - Input: `model_input` (required) → `/opt/ml/processing/input/model`
   - Output: `payload_sample` → `/opt/ml/processing/output`
   - Env vars: `CONTENT_TYPES`, `DEFAULT_NUMERIC_VALUE`, `DEFAULT_TEXT_VALUE`, `SPECIAL_FIELD_*`

2. **Spec** (`payload_spec.py`):
   - Dependencies: `model_input` (required, from training/model steps)
   - Outputs: `payload_sample` (generated samples archive)
   - Node type: INTERNAL

3. **Config** (`config_payload_step.py`):
   - User inputs: TPS, latency thresholds, content types
   - Default values: numeric (0.0), text ("DEFAULT_TEXT")
   - Special fields: Optional dict for custom field values

4. **Builder** (`builder_payload_step.py`):
   - Creates SKLearnProcessor
   - Maps inputs/outputs using spec
   - Passes env vars from config to script

5. **Script** (`payload.py`):
   - Extracts hyperparameters from model.tar.gz
   - Creates variable list from `tab_field_list` + `cat_field_list`
   - Generates JSON/CSV payloads
   - Archives samples to payload.tar.gz

### Current Limitations

1. **No text field detection** - Only handles tabular fields
2. **No model type awareness** - Doesn't check `model_class` from hyperparams
3. **No custom payload support** - Cannot load user-provided samples
4. **Single default mechanism** - Only `SPECIAL_FIELD_*` for overrides
5. **No multi-modal support** - Cannot generate dual text fields

---

## Phase 1: Contract & Spec Enhancement (Week 1, Days 1-2)

**Objective**: Add optional custom payload input to infrastructure

### 1.1 Update Contract

**File**: `src/cursus/steps/contracts/payload_contract.py`

**Changes**:

```python
PAYLOAD_CONTRACT = ScriptContract(
    entry_point="payload.py",
    expected_input_paths={
        "model_input": "/opt/ml/processing/input/model",
        "custom_payload_input": "/opt/ml/processing/input/custom_payload",  # NEW
    },
    expected_output_paths={
        "payload_sample": "/opt/ml/processing/output"
    },
    expected_arguments={},
    required_env_vars=[],
    optional_env_vars={
        # Existing
        "CONTENT_TYPES": "application/json",
        "DEFAULT_NUMERIC_VALUE": "0.0",
        "DEFAULT_TEXT_VALUE": "DEFAULT_TEXT",
        
        # NEW: Unified field defaults (replaces SPECIAL_FIELD_* pattern)
        "FIELD_DEFAULTS": "{}",  # JSON dict format: {"field_name": "field_value"}
        
        # DEPRECATED: SPECIAL_FIELD_* pattern (kept for backward compatibility)
        # Use FIELD_DEFAULTS instead
    },
    framework_requirements={"python": ">=3.7"},
    description="""
    MIMS payload generation script that:
    1. Extracts hyperparameters from model artifacts
    2. Detects model type (tabular/bimodal/trimodal) from model_class
    3. Generates sample payloads with text field support
    4. Optionally uses user-provided custom payload samples
    5. Archives payload files for deployment
    
    Input Structure:
    - /opt/ml/processing/input/model: Model artifacts containing hyperparameters.json
    - /opt/ml/processing/input/custom_payload: (Optional) User-provided payload samples
    
    Output Structure:
    - /opt/ml/processing/output/: payload.tar.gz with generated samples
    
    Environment Variables:
    - CONTENT_TYPES: Comma-separated list (default: "application/json")
    - DEFAULT_NUMERIC_VALUE: Default for numeric fields (default: "0.0")
    - DEFAULT_TEXT_VALUE: Default for text fields (default: "DEFAULT_TEXT")
    - TEXT_FIELD_DEFAULTS: JSON dict of field-specific text values (default: "{}")
    - SPECIAL_FIELD_<fieldname>: Per-field override with template support
    """,
)
```

**Success Criteria**:
- ✅ Added `custom_payload_input` to expected_input_paths
- ✅ Added `TEXT_FIELD_DEFAULTS` to optional_env_vars
- ✅ Updated description to reflect new capabilities

### 1.2 Update Spec

**File**: `src/cursus/steps/specs/payload_spec.py`

**Changes**:

```python
PAYLOAD_SPEC = StepSpecification(
    step_type=get_spec_step_type("Payload"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_payload_contract(),
    dependencies=[
        # Existing: Required model input
        DependencySpec(
            logical_name="model_input",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=True,
            compatible_sources=[
                "XGBoostTraining",
                "LightGBMTraining",
                "LightGBMMTTraining",
                "PyTorchTraining",
                "TrainingStep",
                "ModelStep",
            ],
            semantic_keywords=[
                "model",
                "artifacts",
                "trained",
                "output",
                "ModelArtifacts",
            ],
            data_type="S3Uri",
            description="Trained model artifacts for payload generation",
        ),
        
        # NEW: Optional custom payload input
        DependencySpec(
            logical_name="custom_payload_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,  # OPTIONAL
            compatible_sources=[
                "ProcessingStep",
                "S3Source",
                "UserProvided",
            ],
            semantic_keywords=[
                "payload",
                "sample",
                "custom",
                "user_provided",
                "inference_sample",
            ],
            data_type="S3Uri",
            description="Optional user-provided custom payload samples (JSON/CSV file or directory)",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="payload_sample",
            aliases=["GeneratedPayloadSamples"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['payload_sample'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Generated payload samples archive (payload.tar.gz)",
        )
    ],
)
```

**Success Criteria**:
- ✅ Added optional `custom_payload_input` dependency
- ✅ Set `required=False` to maintain backward compatibility
- ✅ Added appropriate semantic keywords for discovery

---

## Phase 2: Config & Builder Updates (Week 1, Days 3-4)

**Objective**: Add user-facing configuration fields

### 2.1 Update Config

**File**: `src/cursus/steps/configs/config_payload_step.py`

**Changes**:

```python
class PayloadConfig(ProcessingStepConfigBase):
    """
    Configuration for payload generation and testing.
    
    Supports tabular, bimodal, and trimodal model payload generation.
    """
    
    # ===== Essential User Inputs (Tier 1) =====
    
    expected_tps: int = Field(ge=1, description="Expected transactions per second")
    
    max_latency_in_millisecond: int = Field(
        ge=100, le=10000, description="Maximum acceptable latency in milliseconds"
    )
    
    # ===== System Inputs with Defaults (Tier 2) =====
    
    processing_entry_point: str = Field(
        default="payload.py", description="Entry point script for payload generation"
    )
    
    source_model_inference_content_types: List[str] = Field(
        default=["text/csv"],
        description="Content type for model inference input"
    )
    
    source_model_inference_response_types: List[str] = Field(
        default=["application/json"],
        description="Response type for model inference output"
    )
    
    # Default values for payload generation
    default_numeric_value: float = Field(
        default=0.0, description="Default value for numeric fields"
    )
    
    default_text_value: str = Field(
        default="DEFAULT_TEXT", description="Default value for text fields"
    )
    
    # Unified field defaults (replaces special_field_values)
    field_defaults: Optional[Dict[str, str]] = Field(
        default=None,
        description="""
        Optional dictionary mapping field names to sample values for payload generation.
        Works for all field types: text fields, numeric fields, categorical fields, etc.
        Supports template expansion (e.g., {timestamp} → actual timestamp).
        
        Examples:
        - Text fields: {"chat": "Hello, I need help", "shiptrack": "Shipped|In Transit"}
        - ID fields: {"order_id": "ORDER_{timestamp}"}
        - Numeric fields: {"price": "99.99", "quantity": "5"}
        - Any field: Maps directly to payload value
        """
    )
    
    # NEW: Custom payload path (S3 or local)
    custom_payload_path: Optional[str] = Field(
        default=None,
        description="""
        Optional path to user-provided custom payload sample file (JSON/CSV) or directory.
        Supports both S3 paths and local file paths.
        When provided, the script will use this instead of auto-generating payloads.
        Examples: 
        - S3: "s3://my-bucket/custom_payload_samples/sample.json"
        - Local: "/opt/ml/input/data/custom_payload/sample.json"
        - Local: "file:///path/to/payload.json"
        """
    )
    
    max_acceptable_error_rate: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Maximum acceptable error rate (0-1)"
    )
    
    # ... rest of config unchanged ...
```

**Success Criteria**:
- ✅ Added `text_field_defaults` field (optional dict)
- ✅ Added `custom_payload_s3_path` field (optional string)
- ✅ Maintained backward compatibility (all new fields optional)
- ✅ Clear documentation for each field

### 2.2 Update Builder

**File**: `src/cursus/steps/builders/builder_payload_step.py`

**Changes**:

```python
class PayloadStepBuilder(StepBuilderBase):
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs environment variables including new text field defaults.
        """
        env_vars = super()._get_environment_variables()
        
        # Existing env vars
        if hasattr(self.config, "source_model_inference_content_types"):
            env_vars["CONTENT_TYPES"] = ",".join(
                self.config.source_model_inference_content_types
            )
        
        if hasattr(self.config, "default_numeric_value"):
            env_vars["DEFAULT_NUMERIC_VALUE"] = str(self.config.default_numeric_value)
        
        if hasattr(self.config, "default_text_value"):
            env_vars["DEFAULT_TEXT_VALUE"] = str(self.config.default_text_value)
        
        # NEW: Unified FIELD_DEFAULTS as JSON string
        if hasattr(self.config, "field_defaults") and self.config.field_defaults:
            import json
            env_vars["FIELD_DEFAULTS"] = json.dumps(self.config.field_defaults)
            self.log_info(f"Added FIELD_DEFAULTS for {len(self.config.field_defaults)} fields")
        
        self.log_info("Payload environment variables configured")
        return env_vars
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the step using specification and contract.
        Adds support for custom_payload_path from config.
        
        Args:
            inputs: Input data sources keyed by logical name
        
        Returns:
            List of ProcessingInput objects
        """
        # NEW: Check if user provided custom payload path in config
        # If so, add it to inputs dict before processing (supports S3 or local paths)
        if (
            hasattr(self.config, "custom_payload_path")
            and self.config.custom_payload_path
        ):
            # Add custom_payload_input to inputs dict
            inputs["custom_payload_input"] = self.config.custom_payload_path
            self.log_info(
                f"Using custom payload from config: {self.config.custom_payload_path}"
            )
        
        # Continue with standard spec-based input processing from parent
        if not self.spec:
            raise ValueError("Step specification is required")

        if not self.contract:
            raise ValueError("Script contract is required for input mapping")

        processing_inputs = []

        # Process each dependency in the specification
        for _, dependency_spec in self.spec.dependencies.items():
            logical_name = dependency_spec.logical_name

            # Skip if optional and not provided
            if not dependency_spec.required and logical_name not in inputs:
                continue

            # Make sure required inputs are present
            if dependency_spec.required and logical_name not in inputs:
                raise ValueError(f"Required input '{logical_name}' not provided")

            # Get container path from contract
            container_path = None
            if logical_name in self.contract.expected_input_paths:
                container_path = self.contract.expected_input_paths[logical_name]
            else:
                raise ValueError(f"No container path found for input: {logical_name}")

            # Use the input value directly - property references handled by PipelineAssembler
            processing_inputs.append(
                ProcessingInput(
                    input_name=logical_name,
                    source=inputs[logical_name],
                    destination=container_path,
                )
            )

        return processing_inputs
    
    # ... rest of builder unchanged ...
```
### Phase 2: Configuration (Config/Builder) - Week 1, Days 3-4
- [ ] Add `text_field_defaults` field to PayloadConfig
- [ ] Add `custom_payload_path` field to PayloadConfig (supports both S3 and local paths)
- [ ] Update builder to serialize TEXT_FIELD_DEFAULTS to JSON env var
- [ ] **Update builder `_get_inputs()` to detect and use `custom_payload_path` from config**
- [ ] Ensure builder handles optional custom_payload_input from step inputs
- [ ] Add config validation for new fields
### Phase 2: Configuration (Config/Builder) - Week 1, Days 3-4
- [ ] Add `text_field_defaults` field to PayloadConfig
- [ ] Add `custom_payload_s3_path` field to PayloadConfig
- [ ] Update builder to serialize TEXT_FIELD_DEFAULTS to JSON env var
- [ ] **Update builder `_get_inputs()` to detect and use `custom_payload_s3_path` from config**
- [ ] Ensure builder handles optional custom_payload_input from step inputs
- [ ] Add config validation for new fields

**Success Criteria**:
- ✅ Added `TEXT_FIELD_DEFAULTS` env var from config
- ✅ JSON serialization of text_field_defaults dict
- ✅ Optional custom_payload_input handled automatically by parent
- ✅ Logging for debugging

---

## Phase 3: Script Core Enhancements (Week 1, Days 5 - Week 2, Days 1-2)

**Objective**: Implement multi-modal payload generation in script

### 3.1 Add Model Type Detection

**File**: `src/cursus/steps/scripts/payload.py`

**New Function**:

```python
def detect_model_type(hyperparams: Dict) -> str:
    """
    Detect model type from hyperparameters.
    
    Detection logic:
    1. Check for trimodal indicators (primary_text_name + secondary_text_name)
    2. Check for bimodal indicators (text_name field)
    3. Default to tabular (traditional XGBoost/LightGBM)
    
    Args:
        hyperparams: Dictionary loaded from hyperparameters.json
    
    Returns:
        'trimodal', 'bimodal', or 'tabular'
    """
    model_class = hyperparams.get("model_class", "").lower()
    
    # Check for trimodal
    if "trimodal" in model_class or (
        "primary_text_name" in hyperparams and "secondary_text_name" in hyperparams
    ):
        logger.info("Detected trimodal model (dual text + tabular)")
        return "trimodal"
    
    # Check for bimodal
    if "multimodal" in model_class or "text_name" in hyperparams:
        logger.info("Detected bimodal model (text + tabular)")
        return "bimodal"
    
    # Default to tabular
    logger.info("Detected tabular model")
    return "tabular"
```

**Success Criteria**:
- ✅ Correctly detects trimodal from `model_class` or field presence
- ✅ Correctly detects bimodal from `model_class` or `text_name`
- ✅ Defaults to tabular for existing models
- ✅ Logging for visibility

### 3.2 Add Text Field Defaults Loader

**New Function**:

```python
def get_text_field_defaults(environ_vars: Dict[str, str]) -> Dict[str, str]:
    """
    Load text field default values from environment.
    
    Priority (highest to lowest):
    1. SPECIAL_FIELD_* prefix (per-field overrides)
    2. TEXT_FIELD_DEFAULTS (JSON dict)
    3. Empty dict (use auto-generated defaults)
    
    Args:
        environ_vars: Environment variables dictionary
    
    Returns:
        Dictionary mapping field names to default text values
    """
    text_defaults = {}
    
    # Method 1: Load from JSON dictionary
    if "TEXT_FIELD_DEFAULTS" in environ_vars:
        try:
            text_defaults = json.loads(environ_vars["TEXT_FIELD_DEFAULTS"])
            logger.info(f"Loaded {len(text_defaults)} text field defaults from TEXT_FIELD_DEFAULTS")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse TEXT_FIELD_DEFAULTS: {e}")
    
    # Method 2: Load from SPECIAL_FIELD_ prefix (overrides JSON)
    for env_var, env_value in environ_vars.items():
        if env_var.startswith(ENV_SPECIAL_FIELD_PREFIX):
            field_name = env_var[len(ENV_SPECIAL_FIELD_PREFIX):].lower()
            text_defaults[field_name] = env_value
            logger.debug(f"Added SPECIAL_FIELD override for '{field_name}'")
    
    return text_defaults
```

**Success Criteria**:
- ✅ Loads TEXT_FIELD_DEFAULTS JSON dict
- ✅ SPECIAL_FIELD_* overrides JSON values
- ✅ Handles malformed JSON gracefully
- ✅ Returns empty dict if neither provided

### 3.3 Add Text Sample Generator

**New Function**:

```python
def generate_text_sample(
    field_name: str,
    text_field_defaults: Dict[str, str],
    default_text_value: str = "Sample text for inference testing"
) -> str:
    """
    Generate sample text for a text field with 3-tier priority.
    
    Priority (highest to lowest):
    1. User-provided value from text_field_defaults (exact or case-insensitive match)
    2. Intelligent default based on field name pattern
    3. Generic default from DEFAULT_TEXT_VALUE
    
    Args:
        field_name: Name of the text field
        text_field_defaults: User-provided text defaults dictionary
        default_text_value: Generic fallback default
    
    Returns:
        Sample text string for the field
    """
    # Priority 1: User-provided (exact match)
    if field_name in text_field_defaults:
        value = text_field_defaults[field_name]
        # Support template expansion (e.g., {timestamp})
        try:
            return value.format(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        except (KeyError, ValueError):
            return value
    
    # Case-insensitive fallback
    field_lower = field_name.lower()
    for key, value in text_field_defaults.items():
        if key.lower() == field_lower:
            try:
                return value.format(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            except (KeyError, ValueError):
                return value
    
    # Priority 2: Intelligent defaults based on field name
    if "chat" in field_lower or "dialogue" in field_lower or "conversation" in field_lower:
        return "Hello, I need help with my order. Can you assist me?"
    elif "shiptrack" in field_lower or "event" in field_lower or "tracking" in field_lower:
        return "Package shipped|In transit|Delivered"
    elif "description" in field_lower or "desc" in field_lower:
        return "Product description text for testing purposes"
    elif "comment" in field_lower or "note" in field_lower:
        return "Additional notes and comments for testing"
    elif "title" in field_lower or "subject" in field_lower:
        return "Sample title for testing"
    elif "message" in field_lower or "msg" in field_lower:
        return "Sample message content for testing"
    
    # Priority 3: Generic default
    return default_text_value
```

**Success Criteria**:
- ✅ User-provided values take precedence
- ✅ Intelligent defaults for common field names
- ✅ Template support for dynamic values ({timestamp})
- ✅ Graceful fallback to generic default

### 3.4 Add Custom Payload Loader

**New Function**:

```python
def load_custom_payload(
    custom_path: Path, 
    content_type: str
) -> Optional[Dict]:
    """
    Load user-provided custom payload sample.
    
    Supports:
    - JSON file: Load and return as dict
    - CSV file: Load first row as dict
    - Directory: Search for JSON/CSV files
    
    Args:
        custom_path: Path to custom payload file or directory
        content_type: Expected content type ('application/json' or 'text/csv')
    
    Returns:
        Dictionary with payload data if successful, None otherwise
    """
    if not custom_path.exists():
        logger.warning(f"Custom payload path not found: {custom_path}")
        return None
    
    try:
        # Handle directory: search for sample files
        if custom_path.is_dir():
            logger.info(f"Searching for payload samples in directory: {custom_path}")
            
            # Look for JSON files first
            json_files = list(custom_path.glob("*.json"))
            if json_files:
                logger.info(f"Found {len(json_files)} JSON files, using first: {json_files[0]}")
                with open(json_files[0], 'r') as f:
                    return json.load(f)
            
            # Look for CSV files
            csv_files = list(custom_path.glob("*.csv"))
            if csv_files:
                logger.info(f"Found {len(csv_files)} CSV files, using first: {csv_files[0]}")
                df = pd.read_csv(csv_files[0])
                if len(df) > 0:
                    return df.iloc[0].to_dict()
                else:
                    logger.warning("CSV file is empty")
                    return None
            
            logger.warning("No JSON or CSV files found in directory")
            return None
        
        # Handle file: load based on extension
        elif custom_path.is_file():
            logger.info(f"Loading custom payload from file: {custom_path}")
            
            if custom_path.suffix == '.json':
                with open(custom_path, 'r') as f:
                    payload = json.load(f)
                    logger.info(f"Loaded JSON payload with {len(payload)} fields")
                    return payload
            
            elif custom_path.suffix == '.csv':
                df = pd.read_csv(custom_path)
                if len(df) > 0:
                    payload = df.iloc[0].to_dict()
                    logger.info(f"Loaded CSV payload with {len(payload)} fields")
                    return payload
                else:
                    logger.warning("CSV file is empty")
                    return None
            
            else:
                logger.warning(f"Unsupported file extension: {custom_path.suffix}")
                return None
    
    except Exception as e:
        logger.error(f"Failed to load custom payload: {e}", exc_info=True)
        return None
    
    return None
```

**Success Criteria**:
- ✅ Loads JSON files correctly
- ✅ Loads CSV files (first row as dict)
- ✅ Searches directories for samples
- ✅ Error handling for malformed files

---

## Phase 4: Multi-Modal Payload Generation (Week 2, Days 3-5)

**Objective**: Update payload generation functions for multi-modal support

### 4.1 Update JSON Payload Generator

**File**: `src/cursus/steps/scripts/payload.py`

**Modified Function**:

```python
def generate_json_payload(
    input_vars: List[List[str]],
    hyperparams: Dict,
    default_numeric_value: float,
    default_text_value: str,
    text_field_defaults: Dict[str, str],
) -> str:
    """
    Generate JSON format payload with multi-modal support.
    
    Handles:
    - Tabular: Only numeric/categorical fields
    - Bimodal: text_name + numeric/categorical fields
    - Trimodal: primary_text_name + secondary_text_name + numeric/categorical fields
    
    Args:
        input_vars: List of [field_name, var_type] pairs for tabular features
        hyperparams: Full hyperparameters dict from model
        default_numeric_value: Default for numeric fields
        default_text_value: Generic default for text fields
        text_field_defaults: User-provided text field values
    
    Returns:
        JSON string with complete payload
    """
    payload = {}
    model_type = detect_model_type(hyperparams)
    
    # Add ID field if present
    id_name = hyperparams.get("id_name")
    if id_name:
        payload[id_name] = text_field_defaults.get(
            id_name,
            f"TEST_ID_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Add text fields based on model type
    if model_type == "bimodal":
        text_name = hyperparams.get("text_name")
        if text_name:
            payload[text_name] = generate_text_sample(
                text_name, text_field_defaults, default_text_value
            )
            logger.info(f"Added bimodal text field: {text_name}")
    
    elif model_type == "trimodal":
        primary_text_name = hyperparams.get("primary_text_name")
        secondary_text_name = hyperparams.get("secondary_text_name")
        
        if primary_text_name:
            payload[primary_text_name] = generate_text_sample(
                primary_text_name, text_field_defaults, default_text_value
            )
            logger.info(f"Added primary text field: {primary_text_name}")
        
        if secondary_text_name:
            payload[secondary_text_name] = generate_text_sample(
                secondary_text_name, text_field_defaults, default_text_value
            )
            logger.info(f"Added secondary text field: {secondary_text_name}")
    
    # Add tabular fields (existing logic)
    for field_name, var_type in input_vars:
        if var_type in ["TEXT", VariableType.TEXT]:
            # For categorical TEXT fields, check text_field_defaults first
            payload[field_name] = text_field_defaults.get(
                field_name,
                default_text_value
            )
        else:
            payload[field_name] = str(default_numeric_value)
    
    return json.dumps(payload)
```

**Success Criteria**:
- ✅ Detects model type correctly
- ✅ Adds appropriate text fields for each model type
- ✅ Maintains tabular field generation
- ✅ Uses text_field_defaults with fallbacks

### 4.2 Update CSV Payload Generator

**Modified Function**:

```python
def generate_csv_payload(
    input_vars: List[List[str]],
    hyperparams: Dict,
    default_numeric_value: float,
    default_text_value: str,
    text_field_defaults: Dict[str, str],
) -> str:
    """
    Generate CSV format payload with multi-modal support.
    
    Returns:
        Comma-separated string of values (no header)
    """
    values = []
    model_type = detect_model_type(hyperparams)
    
    # Add ID field if present
    id_name = hyperparams.get("id_name")
    if id_name:
        values.append(
            text_field_defaults.get(
                id_name,
                f"TEST_ID_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        )
    
    # Add text fields based on model type
    if model_type == "bimodal":
        text_name = hyperparams.get("text_name")
        if text_name:
            values.append(
                generate_text_sample(text_name, text_field_defaults, default_text_value)
            )
    
    elif model_type == "trimodal":
        primary_text_name = hyperparams.get("primary_text_name")
        secondary_text_name = hyperparams.get("secondary_text_name")
        
        if primary_text_name:
            values.append(
                generate_text_sample(primary_text_name, text_field_defaults, default_text_value)
            )
        
        if secondary_text_name:
            values.append(
                generate_text_sample(secondary_text_name, text_field_defaults, default_text_value)
            )
    
    # Add tabular fields
    for field_name, var_type in input_vars:
        if var_type in ["TEXT", VariableType.TEXT]:
            values.append(text_field_defaults.get(field_name, default_text_value))
        else:
            values.append(str(default_numeric_value))
    
    return ",".join(values)
```

**Success Criteria**:
- ✅ Detects model type correctly
- ✅ Adds text fields in correct order
- ✅ Maintains tabular field generation
- ✅ Uses text_field_defaults with fallbacks

### 4.3 Update main() Function

**Modified Function**:

```python
def main():
    """
    Main entry point with multi-modal and custom payload support.
    """
    try:
        # Setup
        environ_vars = get_environ_vars()
        input_paths = get_input_paths()
        output_dir = Path(get_output_path())
        working_directory = get_working_directory()
        
        payload_sample_dir = working_directory / "payload_sample"
        ensure_directory(output_dir)
        ensure_directory(payload_sample_dir)
        
        # Load hyperparameters
        model_input_path = Path(input_paths["model_input"])
        hyperparams = extract_hyperparameters(model_input_path)
        
        # Get configuration from environment
        content_types = environ_vars.get("CONTENT_TYPES", "application/json").split(",")
        default_numeric_value = float(environ_vars.get("DEFAULT_NUMERIC_VALUE", "0.0"))
        default_text_value = environ_vars.get("DEFAULT_TEXT_VALUE", "DEFAULT_TEXT")
        
        # NEW: Load text field defaults
        text_field_defaults = get_text_field_defaults(environ_vars)
        logger.info(f"Loaded {len(text_field_defaults)} text field defaults")
        
        # NEW: Check for custom payload input
        custom_payload_path = Path(input_paths.get("custom_payload_input", "/opt/ml/processing/input/custom_payload"))
        custom_payload = None
        
        if custom_payload_path.exists():
            logger.info(f"Found custom payload at: {custom_payload_path}")
            custom_payload = load_custom_payload(custom_payload_path, content_types[0])
        
        if custom_payload:
            # Use custom payload directly
            logger.info("Using user-provided custom payload")
            payload_file_paths = save_custom_payload(
                payload_sample_dir, custom_payload, content_types
            )
        else:
            # Generate from hyperparameters (with multi-modal support)
            model_type = detect_model_type(hyperparams)
            logger.info(f"Generating payload for {model_type} model")
            
            # Create variable list for tabular features
            var_type_list = create_model_variable_list(hyperparams)
            
            # Generate payloads with multi-modal support
            payload_file_paths = save_payloads(
                payload_sample_dir,
                var_type_list,
                hyperparams,
                content_types,
                default_numeric_value,
                default_text_value,
                text_field_defaults,
            )
        
        # Create archive
        archive_path = create_payload_archive(payload_file_paths, output_dir)
        
        # Log summary
        logger.info(f"MIMS payload generation complete.")
        logger.info(f"Number of payload samples: {len(payload_file_paths)}")
        logger.info(f"Payload archive: {archive_path}")
        
        return str(archive_path)
    
    except Exception as e:
        logger.error(f"Error in payload generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise
```

**Success Criteria**:
- ✅ Loads text field defaults from environment
- ✅ Checks for custom payload input
- ✅ Uses custom payload if provided
- ✅ Falls back to auto-generation with multi-modal support
- ✅ Maintains backward compatibility

---

## Implementation Checklist

### Phase 1: Infrastructure (Contract/Spec) - Week 1, Days 1-2 ✅ COMPLETED
- [x] Add `custom_payload_input` to contract expected_input_paths
- [x] Add `FIELD_DEFAULTS` to contract optional_env_vars (unified approach)
- [x] Update contract description
- [x] Add `custom_payload_input` dependency to spec (required=False)
- [x] Update spec with additional compatible training sources

### Phase 2: Configuration (Config/Builder) - Week 1, Days 3-4 ✅ COMPLETED
- [x] Add `field_defaults` field to PayloadConfig (unified approach)
- [x] Add `custom_payload_path` field to PayloadConfig (supports both S3 and local paths)
- [x] Update builder to serialize FIELD_DEFAULTS to JSON env var
- [x] Update builder `_get_inputs()` to detect and use `custom_payload_path` from config
- [x] Remove deprecated `special_field_values` field (clean implementation)
- [x] Remove deprecated `special_field_values` handling from builder

### Phase 3: Script Core Functions - Week 1, Day 5 - Week 2, Days 1-2 ✅ COMPLETED
- [x] Implement `detect_model_type()` function
- [x] Implement `get_field_defaults()` function (renamed from get_text_field_defaults for consistency)
- [x] Implement `generate_text_sample()` function with 3-tier priority
- [x] Implement `load_custom_payload()` function
- [x] Add pandas import if needed for CSV loading (conditional import)
- [x] Add ENV_FIELD_DEFAULTS constant

### Phase 4: Payload Generation - Week 2, Days 3-5 ✅ COMPLETED
- [x] Update `generate_json_payload()` for multi-modal support
- [x] Update `generate_csv_payload()` for multi-modal support
- [x] Update `generate_sample_payloads()` signature and logic
- [x] Update `save_payloads()` signature and logic
- [x] Update `main()` function with complete multi-modal workflow
- [x] Add custom payload loading support in main()
- [x] Integrate field_defaults throughout the pipeline
- [x] Maintain backward compatibility with special_field_values

### Phase 5: Testing & Validation - Week 3, Days 1-3
- [ ] Test with tabular-only models (backward compatibility)
- [ ] Test with bimodal PyTorch models
- [ ] Test with trimodal PyTorch models
- [ ] Test with custom payload input (JSON file)
- [ ] Test with custom payload input (CSV file)
- [ ] Test with custom payload input (directory)
- [ ] Test TEXT_FIELD_DEFAULTS environment variable
- [ ] Test SPECIAL_FIELD_* override pattern
- [ ] Test template expansion ({timestamp})

### Phase 6: Documentation - Week 3, Days 4-5
- [ ] Update inline code documentation
- [ ] Add usage examples in docstrings
- [ ] Create user guide for multi-modal payload configuration
- [ ] Update contract/spec/config documentation
- [ ] Add troubleshooting section

---

## Usage Examples

### Example 1: Auto-Generate Bimodal Payload with Custom Text

```python
from cursus.steps.configs.config_payload_step import PayloadConfig
from cursus.steps.builders.builder_payload_step import PayloadStepBuilder

# Configure with text field defaults
config = PayloadConfig(
    pipeline_name="my_bimodal_pipeline",
    bucket="my-ml-bucket",
    expected_tps=100,
    max_latency_in_millisecond=500,
    text_field_defaults={
        "chat": "Hello, I have a question about my order status",
        "order_id": "ORDER_12345"
    },
    default_numeric_value=0.0,
    default_text_value="DEFAULT"
)

# Build step
builder = PayloadStepBuilder(config, sagemaker_session=session, role=role)
payload_step = builder.create_step(
    inputs={
        "model_input": training_step.properties.ModelArtifacts.S3ModelArtifacts
    }
)
```

**Generated JSON Payload**:
```json
{
  "order_id": "ORDER_12345",
  "chat": "Hello, I have a question about my order status",
  "feature_1": "0.0",
  "feature_2": "0.0",
  "category_1": "DEFAULT"
}
```

### Example 2: Auto-Generate Trimodal Payload with Intelligent Defaults

```python
config = PayloadConfig(
    pipeline_name="my_trimodal_pipeline",
    bucket="my-ml-bucket",
    expected_tps=200,
    max_latency_in_millisecond=300,
    # No text_field_defaults - uses intelligent defaults
)

# Bimodal model with chat + shiptrack fields will auto-detect and generate:
# - "chat" → "Hello, I need help with my order. Can you assist me?"
# - "shiptrack" → "Package shipped|In transit|Delivered"
```

**Generated JSON Payload**:
```json
{
  "chat": "Hello, I need help with my order. Can you assist me?",
  "shiptrack": "Package shipped|In transit|Delivered",
  "feature_1": "0.0",
  "feature_2": "0.0"
}
```

### Example 3: User-Provided Custom Payload

```python
# Step 1: Upload custom payload to S3
# s3://my-bucket/custom_payloads/sample.json:
# {
#   "chat": "Domain-specific query text",
#   "shiptrack": "Real tracking events",
#   "feature_1": "1.5",
#   "feature_2": "2.3"
# }

# Step 2: Configure with custom payload path
config = PayloadConfig(
    pipeline_name="my_pipeline",
    bucket="my-ml-bucket",
    expected_tps=100,
    max_latency_in_millisecond=500,
    custom_payload_s3_path="s3://my-bucket/custom_payloads/sample.json"
)

# Step 3: Build step with custom payload input
payload_step = builder.create_step(
    inputs={
        "model_input": training_step.properties.ModelArtifacts.S3ModelArtifacts,
        "custom_payload_input": config.custom_payload_s3_path
    }
)

# Result: Uses the custom payload directly instead of auto-generating
```

### Example 4: Tabular-Only (Backward Compatible)

```python
# Existing config - no changes needed
config = PayloadConfig(
    pipeline_name="my_xgboost_pipeline",
    bucket="my-ml-bucket",
    expected_tps=500,
    max_latency_in_millisecond=200,
    default_numeric_value=0.0,
    default_text_value="DEFAULT"
)

# Works exactly as before - no text fields added
# Model type detection returns "tabular"
# Generated payload only contains tabular features
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/scripts/test_payload_multimodal.py`

```python
import pytest
from src.cursus.steps.scripts.payload import (
    detect_model_type,
    generate_text_sample,
    get_text_field_defaults,
    load_custom_payload
)

class TestModelTypeDetection:
    def test_detect_trimodal(self):
        hyperparams = {
            "model_class": "TrimodalBERT",
            "primary_text_name": "chat",
            "secondary_text_name": "shiptrack"
        }
        assert detect_model_type(hyperparams) == "trimodal"
    
    def test_detect_bimodal(self):
        hyperparams = {
            "model_class": "BimodalModel",
            "text_name": "chat"
        }
        assert detect_model_type(hyperparams) == "bimodal"
    
    def test_detect_tabular(self):
        hyperparams = {
            "model_class": "XGBoostClassifier"
        }
        assert detect_model_type(hyperparams) == "tabular"

class TestTextSampleGeneration:
    def test_user_provided_value(self):
        text_defaults = {"chat": "Custom message"}
        result = generate_text_sample("chat", text_defaults, "DEFAULT")
        assert result == "Custom message"
    
    def test_intelligent_default(self):
        text_defaults = {}
        result = generate_text_sample("chat", text_defaults, "DEFAULT")
        assert "order" in result.lower()
    
    def test_generic_fallback(self):
        text_defaults = {}
        result = generate_text_sample("unknown_field", text_defaults, "FALLBACK")
        assert result == "FALLBACK"

class TestTextFieldDefaultsLoader:
    def test_load_json_dict(self):
        env_vars = {
            "TEXT_FIELD_DEFAULTS": '{"chat": "Test", "shiptrack": "Events"}'
        }
        result = get_text_field_defaults(env_vars)
        assert result == {"chat": "Test", "shiptrack": "Events"}
    
    def test_special_field_override(self):
        env_vars = {
            "TEXT_FIELD_DEFAULTS": '{"chat": "Original"}',
            "SPECIAL_FIELD_chat": "Override"
        }
        result = get_text_field_defaults(env_vars)
        assert result["chat"] == "Override"

class TestCustomPayloadLoader:
    def test_load_json_file(self, tmp_path):
        payload_file = tmp_path / "sample.json"
        payload_file.write_text('{"chat": "Test", "feature_1": "1.0"}')
        
        result = load_custom_payload(payload_file, "application/json")
        assert result == {"chat": "Test", "feature_1": "1.0"}
    
    def test_load_csv_file(self, tmp_path):
        payload_file = tmp_path / "sample.csv"
        payload_file.write_text("chat,feature_1\nTest,1.0\n")
        
        result = load_custom_payload(payload_file, "text/csv")
        assert result["chat"] == "Test"
        assert result["feature_1"] == 1.0
```

### Integration Tests

**File**: `tests/integration/test_payload_step_multimodal.py`

```python
import pytest
from src.cursus.steps.configs.config_payload_step import PayloadConfig
from src.cursus.steps.builders.builder_payload_step import PayloadStepBuilder

class TestPayloadStepMultimodal:
    def test_bimodal_payload_generation(self, tmp_path, sample_bimodal_model):
        """Test end-to-end payload generation for bimodal model."""
        config = PayloadConfig(
            pipeline_name="test_bimodal",
            bucket="test-bucket",
            expected_tps=100,
            max_latency_in_millisecond=500,
            text_field_defaults={"chat": "Test message"}
        )
        
        builder = PayloadStepBuilder(config)
        # Run payload generation
        # Verify output contains chat field with correct value
    
    def test_trimodal_payload_generation(self, tmp_path, sample_trimodal_model):
        """Test end-to-end payload generation for trimodal model."""
        # Similar to bimodal test
    
    def test_custom_payload_input(self, tmp_path, custom_payload_file):
        """Test using custom payload instead of auto-generation."""
        # Verify custom payload is used verbatim
    
    def test_backward_compatibility(self, tmp_path, sample_xgboost_model):
        """Test that existing tabular models still work."""
        # Verify no breaking changes
```

### Performance Tests

```python
def test_payload_generation_performance():
    """Verify payload generation is fast enough."""
    import time
    
    config = PayloadConfig(...)
    
    start = time.perf_counter()
    # Generate payload
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    assert latency_ms < 100, f"Payload generation took {latency_ms:.2f}ms, expected < 100ms"
```

---

## Deployment Considerations

### Rollout Strategy

1. **Stage 1**: Deploy to dev environment
   - Test with all model types
   - Validate backward compatibility

2. **Stage 2**: Deploy to staging
   - Shadow testing with production models
   - Monitor for issues

3. **Stage 3**: Gradual production rollout
   - 25% → 50% → 100% of pipelines
   - Rollback plan if issues detected

### Monitoring

**CloudWatch Metrics**:
- Payload generation latency (P50, P95, P99)
- Custom payload usage rate
- Model type distribution (tabular/bimodal/trimodal)
- Failure rate by model type

**Alarms**:
- Payload generation failures > 5%
- P95 latency > 5 seconds
- Custom payload load failures

### Backward Compatibility Validation

```python
# Validation script
def validate_backward_compatibility():
    """
    Run with existing tabular models to ensure no breaking changes.
    """
    test_configs = [
        ("XGBoost", "existing_xgboost_pipeline"),
        ("LightGBM", "existing_lightgbm_pipeline"),
        ("LightGBMMT", "existing_lightgbmmt_pipeline"),
    ]
    
    for model_type, pipeline_name in test_configs:
        # Generate payload with old config
        # Compare with baseline
        # Assert identical output
        print(f"✅ {model_type} backward compatible")
```

---

## Summary

### Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Contract/Spec | 2 days | Updated contract, spec with optional inputs |
| Phase 2: Config/Builder | 2 days | New config fields, builder env var handling |
| Phase 3: Script Core | 3 days | 4 new core functions implemented |
| Phase 4: Payload Gen | 3 days | Updated JSON/CSV generators, main() |
| Phase 5: Testing | 3 days | Unit tests, integration tests, validation |
| Phase 6: Documentation | 2 days | Inline docs, user guide, examples |
| **Total** | **15 days (3 weeks)** | **Complete multi-modal payload support** |

### Expected Impact

**For PyTorch Multi-Modal Models**:
- ✅ Enables latency testing for bimodal models (single text + tabular)
- ✅ Enables latency testing for trimodal models (dual text + tabular)
- ✅ Provides flexible text sample configuration
- ✅ Supports domain-specific custom payloads

**For Existing Tabular Models**:
- ✅ Zero breaking changes
- ✅ Identical behavior as before
- ✅ Automatic performance (no config changes needed)

**For Users**:
- ✅ Dict-based text configuration (easier than SPECIAL_FIELD_*)
- ✅ Optional custom payload upload
- ✅ Intelligent text defaults for common field names
- ✅ Template support for dynamic values

### Next Steps

1. **Review & Approval**: Get stakeholder approval for design
2. **Implementation**: Follow 3-week timeline
3. **Testing**: Comprehensive validation with all model types
4. **Deployment**: Gradual rollout with monitoring
5. **Documentation**: User guide and examples

---

## References

### Implementation Files

**Contract/Spec/Config/Builder**:
- `src/cursus/steps/contracts/payload_contract.py` - Script contract
- `src/cursus/steps/specs/payload_spec.py` - Step specification
- `src/cursus/steps/configs/config_payload_step.py` - User configuration
- `src/cursus/steps/builders/builder_payload_step.py` - Step builder

**Script**:
- `src/cursus/steps/scripts/payload.py` - Payload generation script

**Multi-Modal Models**:
- `projects/bsm_pytorch/docker/lightning_models/` - BSM PyTorch models
- `projects/rnr_pytorch_bedrock/docker/lightning_models/` - RNR PyTorch models

### Related Documentation

- [PyTorch Training Step Implementation](./2025-07-06_pytorch_training_alignment_implementation_summary.md)
- [Inference Handler Design](../1_design/pytorch_inference_calibration_integration.md)
- [Hyperparameter Class Guide](../0_developer_guide/hyperparameter_class.md)
- [Step Specification Guide](../0_developer_guide/step_specification.md)

### Design Documents

- [Three-Tier Config Design](../1_design/config_tiered_design.md)
- [Specification-Driven Architecture](./2025-07-07_specification_driven_architecture_analysis.md)
- [Contract Alignment](./2025-07-04_contract_alignment_implementation_summary.md)

---

**Document Status**: ✅ Complete and ready for implementation
**Last Updated**: 2025-11-29
**Owner**: ML Platform Team
**Reviewers**: PyTorch Team, Infrastructure Team
