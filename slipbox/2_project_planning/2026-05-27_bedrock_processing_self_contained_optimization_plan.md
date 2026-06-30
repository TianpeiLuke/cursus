# Plan: BedrockProcessing Self-Contained Mode + Robustness Optimizations

**Date**: 2026-05-27
**Status**: ✅ COMPLETE — all phases implemented, pushed, published
**Commits**: `094cc58` (self-contained + structured output), `de7d1c1` (checkpoint/resume)
**Builds**: https://build.amazon.com/7856800673 (DONE), https://build.amazon.com/7856811776 (submitted)
**Motivation**: FZ Trail 29d10d-f (Munged Address migration) identified that requiring an upstream `BedrockPromptTemplateGeneration` step is over-engineered for simple rating tasks. Industry standard is prompt-as-config. Additionally, structured output and checkpoint/resume improve robustness at scale.

## Progress

| Phase | Status | Notes |
|-------|--------|-------|
| 1A: Spec | ✅ Done | `required=False` for prompt_templates + validation_schema |
| 1B: Contract | ✅ Done | +5 optional env vars |
| 1C: Config | ✅ Done | +5 fields, env var property updated, get_public_init_fields updated |
| 2A: Fallback functions | ✅ Done | `_load_templates_with_fallback`, `_load_schema_with_fallback` |
| 2B: main() modification | ✅ Done | Replaced hard requirements with fallback chain |
| 2C: use_structured_output config | ✅ Done | Added to config dict in main() |
| 2D: Structured output method | ✅ Done | `_invoke_bedrock_with_structured_output()` added |
| 2E: Routing in process_single_case | ✅ Done | Conditional dispatch based on config |
| 2F: Checkpoint/resume | ✅ Done | Both sequential + concurrent modes. Commit `de7d1c1` |
| 3: Verification | ✅ Done | ruff check passes, Pydantic validation passes, syntax valid |
| **ALL PHASES** | **✅ COMPLETE** | **Published to CursusPeru/cursus-peru-dev** |

## Scope

Make `BedrockProcessing` step usable **without** `BedrockPromptTemplateGeneration` as an upstream dependency, while adding production robustness optimizations. Fully backward-compatible — existing pipelines unchanged.

## Files Modified

| File | Location | Type of Change |
|------|----------|---------------|
| `bedrock_processing_spec.py` | `steps/specs/` | 2 lines: `required=True` → `required=False` |
| `config_bedrock_processing_step.py` | `steps/configs/` | +5 Optional fields, update env var property |
| `bedrock_processing_contract.py` | `steps/contracts/` | +5 optional_env_vars entries |
| `bedrock_processing.py` | `steps/scripts/` | +2 fallback functions, +1 structured output method, checkpoint/resume |

## Phase 1: Parallel Changes (Independent)

### 1A. Spec — Make Template Dependencies Optional

**File**: `steps/specs/bedrock_processing_spec.py`

Change at line 65 (`prompt_templates` DependencySpec):
```python
required=False,  # Optional — can use config-embedded templates instead
```

Change at line 79 (`validation_schema` DependencySpec):
```python
required=False,  # Optional — can use config-embedded validation schema instead
```

**Why safe**: Builder `_get_inputs()` already handles optional deps (line 345 of builder):
```python
if not dependency_spec.required and logical_name not in inputs:
    self.log_info("Optional input '%s' not provided, skipping", logical_name)
    continue
```

### 1B. Contract — Declare New Env Vars

**File**: `steps/contracts/bedrock_processing_contract.py`

Add to `optional_env_vars` dict (after `"USE_SECURE_PYPI": "false"`):
```python
# Config-embedded template support (self-contained mode)
"BEDROCK_USER_PROMPT_TEMPLATE": "",
"BEDROCK_SYSTEM_PROMPT": "",
"BEDROCK_INPUT_PLACEHOLDERS": "[]",
"BEDROCK_VALIDATION_SCHEMA": "{}",
# Structured output mode
"BEDROCK_USE_STRUCTURED_OUTPUT": "false",
```

### 1C. Config — Add Self-Contained Fields

**File**: `steps/configs/config_bedrock_processing_step.py`

Add 5 new Tier 2 fields (after `bedrock_log_truncations`, before `processing_entry_point`):

```python
# Config-embedded template (self-contained mode)
bedrock_user_prompt_template: Optional[str] = Field(
    default=None,
    description="User prompt template with {placeholder} syntax. If provided, BedrockPromptTemplateGeneration step is not needed.",
)

bedrock_system_prompt: Optional[str] = Field(
    default=None,
    description="System prompt for Bedrock API. Used only in config-embedded mode.",
)

bedrock_input_placeholders: Optional[List[str]] = Field(
    default=None,
    description="List of input placeholder names mapping to DataFrame columns.",
)

bedrock_validation_schema: Optional[Dict[str, Any]] = Field(
    default=None,
    description="JSON validation schema for Pydantic response model creation.",
)

# Structured output
bedrock_use_structured_output: bool = Field(
    default=False,
    description="Use tool_use for guaranteed schema compliance (0% parse failures).",
)
```

Update `bedrock_environment_variables` property — add after existing entries:
```python
"BEDROCK_USER_PROMPT_TEMPLATE": self.bedrock_user_prompt_template or "",
"BEDROCK_SYSTEM_PROMPT": self.bedrock_system_prompt or "",
"BEDROCK_INPUT_PLACEHOLDERS": json.dumps(self.bedrock_input_placeholders) if self.bedrock_input_placeholders else "[]",
"BEDROCK_VALIDATION_SCHEMA": json.dumps(self.bedrock_validation_schema) if self.bedrock_validation_schema else "{}",
"BEDROCK_USE_STRUCTURED_OUTPUT": str(self.bedrock_use_structured_output).lower(),
```

Update `get_public_init_fields` — add optional field forwarding:
```python
if self.bedrock_user_prompt_template is not None:
    bedrock_fields["bedrock_user_prompt_template"] = self.bedrock_user_prompt_template
if self.bedrock_system_prompt is not None:
    bedrock_fields["bedrock_system_prompt"] = self.bedrock_system_prompt
if self.bedrock_input_placeholders is not None:
    bedrock_fields["bedrock_input_placeholders"] = self.bedrock_input_placeholders
if self.bedrock_validation_schema is not None:
    bedrock_fields["bedrock_validation_schema"] = self.bedrock_validation_schema
bedrock_fields["bedrock_use_structured_output"] = self.bedrock_use_structured_output
```

## Phase 2: Script Changes (After Phase 1)

### 2A. Add Fallback Functions

**File**: `steps/scripts/bedrock_processing.py`
**Location**: After `load_validation_schema()` (line ~1387), before `process_split_directory()`

```python
def _load_templates_with_fallback(
    input_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """Load templates: upstream step → env var → error."""
    # Strategy 1: Upstream step
    if "prompt_templates" in input_paths:
        templates_path = input_paths["prompt_templates"]
        if Path(templates_path).exists():
            try:
                return load_prompt_templates(templates_path, log)
            except Exception as e:
                log(f"[WARN] Failed to load from upstream step: {e}, trying env var fallback")

    # Strategy 2: Environment variable config
    user_prompt = environ_vars.get("BEDROCK_USER_PROMPT_TEMPLATE", "")
    if user_prompt:
        log("[INFO] Using config-embedded template (self-contained mode)")
        placeholders_str = environ_vars.get("BEDROCK_INPUT_PLACEHOLDERS", "[]")
        try:
            placeholders = json.loads(placeholders_str)
        except json.JSONDecodeError:
            placeholders = []
        return {
            "system_prompt": environ_vars.get("BEDROCK_SYSTEM_PROMPT", ""),
            "user_prompt_template": user_prompt,
            "input_placeholders": placeholders,
        }

    raise ValueError(
        "No prompt templates available. Provide either:\n"
        "  1. Upstream BedrockPromptTemplateGeneration step, OR\n"
        "  2. BEDROCK_USER_PROMPT_TEMPLATE env var (config-embedded mode)"
    )


def _load_schema_with_fallback(
    input_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """Load schema: upstream step → env var → empty (graceful degradation)."""
    # Strategy 1: Upstream step
    if "validation_schema" in input_paths:
        schema_path = input_paths["validation_schema"]
        if Path(schema_path).exists():
            try:
                return load_validation_schema(schema_path, log)
            except Exception as e:
                log(f"[WARN] Failed to load schema from upstream: {e}, trying env var fallback")

    # Strategy 2: Environment variable config
    schema_str = environ_vars.get("BEDROCK_VALIDATION_SCHEMA", "{}")
    if schema_str and schema_str != "{}":
        try:
            schema = json.loads(schema_str)
            if "properties" in schema:
                log("[INFO] Using config-embedded validation schema (self-contained mode)")
                return schema
        except json.JSONDecodeError as e:
            log(f"[WARN] Failed to parse BEDROCK_VALIDATION_SCHEMA: {e}")

    # Strategy 3: Graceful degradation
    log("[WARN] No validation schema — using basic JSON parsing only")
    return {}
```

### 2B. Modify `main()` — Replace Hard Requirements

**Replace lines 1554-1576** (the two `if ... not in input_paths: raise ValueError` blocks) with:

```python
# Load prompt templates with fallback chain (upstream > env var > error)
templates = _load_templates_with_fallback(input_paths, environ_vars, log)
log(f"Loaded templates: system_prompt={bool(templates.get('system_prompt'))}, "
    f"user_prompt_template={bool(templates.get('user_prompt_template'))}")

# Load validation schema with fallback chain (upstream > env var > empty)
validation_schema = _load_schema_with_fallback(input_paths, environ_vars, log)
log(f"Loaded validation schema with {len(validation_schema.get('properties', {}))} properties")
```

### 2C. Add `use_structured_output` to Config Dict in `main()`

In the config dict (around line 1632), add after truncation entries:
```python
# Structured output mode
"use_structured_output": environ_vars.get(
    "BEDROCK_USE_STRUCTURED_OUTPUT", "false"
).lower() == "true",
```

### 2D. Add Structured Output Method to `BedrockProcessor`

**Location**: After `_invoke_bedrock()` method (after line 879)

```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _invoke_bedrock_with_structured_output(self, prompt: str) -> Dict[str, Any]:
    """Invoke Bedrock using tool_use for guaranteed schema compliance."""
    if self.concurrency_mode == "concurrent":
        self._enforce_rate_limit()
        client = self._get_thread_local_bedrock_client()
    else:
        client = self.bedrock_client

    tool_schema = {
        "type": "object",
        "properties": self.validation_schema.get("properties", {}),
        "required": self.validation_schema.get("required", []),
    }

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": int(self.config["max_tokens"]),
        "temperature": float(self.config["temperature"]),
        "top_p": float(self.config["top_p"]),
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{
            "name": "structured_response",
            "description": "Return the structured analysis response",
            "input_schema": tool_schema,
        }],
        "tool_choice": {"type": "tool", "name": "structured_response"},
    }

    if self.config.get("system_prompt"):
        request_body["system"] = self.config["system_prompt"]

    try:
        response = client.invoke_model(
            modelId=self.effective_model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )
        response_body = json.loads(response["body"].read())

        for block in response_body.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "structured_response":
                return {"content": [{"text": json.dumps(block["input"])}]}

        raise ValueError("No tool_use block found in response")

    except Exception as e:
        fallback_model = self.config.get("fallback_model_id")
        if fallback_model and "ValidationException" in str(e):
            logger.warning(f"Structured output: inference profile failed, falling back to: {fallback_model}")
            response = client.invoke_model(
                modelId=fallback_model,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
            response_body = json.loads(response["body"].read())
            for block in response_body.get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == "structured_response":
                    return {"content": [{"text": json.dumps(block["input"])}]}
            raise ValueError("No tool_use block found in fallback response")
        raise
```

### 2E. Modify `process_single_case` to Use Structured Output

**In `process_single_case`** (line ~999), replace:
```python
response = self._invoke_bedrock(prompt)
```
With:
```python
if self.config.get("use_structured_output") and self.validation_schema:
    response = self._invoke_bedrock_with_structured_output(prompt)
else:
    response = self._invoke_bedrock(prompt)
```

### 2F. Add Checkpoint/Resume to `process_batch`

**In `process_batch` method**, after batch_size initialization (line ~1066):

Add checkpoint loading:
```python
# Checkpoint/resume support
checkpoint_file = (output_dir if output_dir else Path(CONTAINER_PATHS["OUTPUT_DATA_DIR"])) / "_checkpoint.json"
completed_batches = 0
if checkpoint_file.exists():
    try:
        checkpoint = json.loads(checkpoint_file.read_text())
        completed_batches = checkpoint.get("completed_batches", 0)
        if completed_batches > 0:
            logger.info(f"Resuming from checkpoint: skipping {completed_batches} completed batches")
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
```

In the batch loop, add skip logic + checkpoint save:
```python
for i in range(0, len(df), batch_size):
    batch_num = i // batch_size + 1

    # Skip already-completed batches (checkpoint/resume)
    if batch_num <= completed_batches:
        # Load intermediate results for completed batches
        # ... (load from previously saved parquet)
        continue

    # ... existing processing logic ...

    # Save checkpoint after each batch
    if save_intermediate:
        checkpoint_file.write_text(json.dumps({"completed_batches": batch_num, "total_batches": total_batches}))
```

After loop completion:
```python
if checkpoint_file.exists():
    checkpoint_file.unlink()
```

## Phase 3: Verification

1. `ruff check` + `ruff format` on all modified files
2. `brazil-build release` to verify compilation
3. Unit tests:
   - Config: Pydantic validates new Optional fields
   - Fallback: no upstream → env var works
   - Fallback: upstream present → ignores env var
   - Structured output: tool_use response parsed correctly
   - Checkpoint: resume skips completed batches

## Dependency Graph

```
Phase 1A (spec)     ──┐
Phase 1B (contract)   ├── Independent, parallel
Phase 1C (config)    ──┘
                       │
                       ▼
Phase 2A-2F (script)  ── Sequential (depends on env var names from 1B/1C)
                       │
                       ▼
Phase 3 (verify)      ── After all changes
```

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Env var size limit (~16KB) | Document constraint; recommend upstream step for large templates |
| Stale checkpoint on crash | Checkpoint points to valid intermediate files; graceful on re-read |
| tool_use not supported by all models | Fallback: if `use_structured_output=true` but model errors, could add try/catch to fall back to prefilling mode |
| Backward compatibility | All new fields Optional with None/False defaults; existing pipelines unchanged |
