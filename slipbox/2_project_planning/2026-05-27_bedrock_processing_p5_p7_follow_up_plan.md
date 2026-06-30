# Plan: BedrockProcessing P5-P7 — Circuit Breaker, Adaptive Rate Limiting, Converse API

**Date**: 2026-05-27
**Status**: ✅ COMPLETE — committed `c0a3234`, pushed, published
**Build**: https://build.amazon.com/7856859633
**Prerequisite**: Commits `094cc58` (self-contained + structured output), `de7d1c1` (checkpoint/resume) already merged.
**File**: `src/cursus/steps/scripts/bedrock_processing.py`

## Scope

Three follow-up optimizations for the `BedrockProcessing` step — all localized to the script. No config/contract/spec/builder changes needed (these use existing config keys).

## Current State (What We're Building On)

The `BedrockProcessor` class (`__init__` at line ~447) currently has:
- `self.rate_limit_per_second` — static rate (default 10)
- `self.max_concurrent_workers` — static concurrency (default 5)
- `self.time_lock` + `self.last_request_times` — per-thread timing
- `_enforce_rate_limit()` — sleeps if interval too short (line 500-515)
- `@retry(stop=3, wait=exponential)` on `_invoke_bedrock()` — retries on ANY exception

The `_invoke_bedrock()` method (line 819):
- Uses `invoke_model()` with Anthropic-format request body
- Handles `ValidationException` → fallback model
- All other exceptions bubble up to tenacity retry

## P5: Circuit Breaker (~25 lines)

### What It Does

After N consecutive failures, STOP making requests for a cooldown period instead of hammering a failing service. Prevents cascading failures and respects Bedrock throttling.

### Design

Add to `BedrockProcessor.__init__()`:
```python
# Circuit breaker state
self.circuit_breaker_threshold = config.get("circuit_breaker_threshold", 5)
self.circuit_breaker_recovery_sec = config.get("circuit_breaker_recovery_sec", 60)
self._cb_consecutive_failures = 0
self._cb_state = "closed"  # closed (normal), open (blocking), half-open (testing)
self._cb_last_failure_time = 0.0
self._cb_lock = threading.Lock()
```

Add new method after `_enforce_rate_limit()`:
```python
def _circuit_breaker_check(self):
    """Check circuit breaker state before making a request."""
    with self._cb_lock:
        if self._cb_state == "closed":
            return  # Normal operation

        if self._cb_state == "open":
            elapsed = time.time() - self._cb_last_failure_time
            if elapsed >= self.circuit_breaker_recovery_sec:
                self._cb_state = "half-open"
                logger.info(
                    f"Circuit breaker: half-open after {elapsed:.0f}s cooldown, testing one request"
                )
                return
            else:
                remaining = self.circuit_breaker_recovery_sec - elapsed
                raise RuntimeError(
                    f"Circuit breaker OPEN: {remaining:.0f}s remaining before retry. "
                    f"({self._cb_consecutive_failures} consecutive failures)"
                )

        # half-open: allow one request through (already returned above)

def _circuit_breaker_record_success(self):
    """Record successful request — reset circuit breaker."""
    with self._cb_lock:
        if self._cb_state == "half-open":
            logger.info("Circuit breaker: closing (recovery confirmed)")
        self._cb_consecutive_failures = 0
        self._cb_state = "closed"

def _circuit_breaker_record_failure(self):
    """Record failed request — potentially trip circuit breaker."""
    with self._cb_lock:
        self._cb_consecutive_failures += 1
        self._cb_last_failure_time = time.time()

        if self._cb_consecutive_failures >= self.circuit_breaker_threshold:
            self._cb_state = "open"
            logger.warning(
                f"Circuit breaker TRIPPED: {self._cb_consecutive_failures} consecutive failures. "
                f"Blocking requests for {self.circuit_breaker_recovery_sec}s."
            )
```

### Integration in `_invoke_bedrock()`

At the START of `_invoke_bedrock()` (before rate limiting):
```python
self._circuit_breaker_check()
```

After successful response (line 856):
```python
self._circuit_breaker_record_success()
return json.loads(response["body"].read())
```

In the `except` block, before re-raising:
```python
self._circuit_breaker_record_failure()
```

Same pattern for `_invoke_bedrock_with_structured_output()`.

### Where to Insert

- `__init__` additions: After line 467 (`self.time_lock = threading.Lock()`)
- New methods: After `_enforce_rate_limit()` (after line 515)
- `_invoke_bedrock` integration: Lines 819-877
- `_invoke_bedrock_with_structured_output` integration: Lines 879-950

## P6: Adaptive Rate Limiting (~15 lines)

### What It Does

Automatically adjusts `rate_limit_per_second` based on observed throttling rate. Increases when healthy, decreases when throttled — no manual tuning needed.

### Design

Add to `BedrockProcessor.__init__()`:
```python
# Adaptive rate limiting state
self._adaptive_rate_enabled = config.get("adaptive_rate_limiting", False)
self._max_rate = config.get("rate_limit_per_second", 10)
self._min_rate = 1
self._throttle_window = []  # (timestamp, was_throttled) tuples
self._throttle_window_size = 100  # Track last N requests
```

Add new method:
```python
def _adapt_rate(self, was_throttled: bool):
    """Adjust rate limit based on recent throttle history."""
    if not self._adaptive_rate_enabled:
        return

    now = time.time()
    self._throttle_window.append((now, was_throttled))

    # Keep only recent window
    cutoff = now - 60  # Last 60 seconds
    self._throttle_window = [(t, th) for t, th in self._throttle_window if t > cutoff]

    if len(self._throttle_window) < 10:
        return  # Not enough data

    throttle_rate = sum(1 for _, th in self._throttle_window if th) / len(self._throttle_window)

    if throttle_rate > 0.05:
        # More than 5% throttled — halve the rate
        new_rate = max(self._min_rate, self.rate_limit_per_second // 2)
        if new_rate != self.rate_limit_per_second:
            logger.warning(
                f"Adaptive rate: throttle rate {throttle_rate:.1%}, "
                f"reducing {self.rate_limit_per_second} → {new_rate} req/s"
            )
            self.rate_limit_per_second = new_rate
    elif throttle_rate < 0.01 and self.rate_limit_per_second < self._max_rate:
        # Less than 1% throttled — increase by 2
        new_rate = min(self._max_rate, self.rate_limit_per_second + 2)
        if new_rate != self.rate_limit_per_second:
            logger.info(
                f"Adaptive rate: throttle rate {throttle_rate:.1%}, "
                f"increasing {self.rate_limit_per_second} → {new_rate} req/s"
            )
            self.rate_limit_per_second = new_rate
```

### Integration in `_invoke_bedrock()`

After successful response:
```python
self._adapt_rate(was_throttled=False)
```

In exception handling, if `ThrottlingException` detected:
```python
if "ThrottlingException" in str(e) or "TooManyRequestsException" in str(e):
    self._adapt_rate(was_throttled=True)
```

### Config/Contract (No Changes Needed)

The adaptive rate uses existing `rate_limit_per_second` as max. Add one new config key read in `main()`:
```python
"adaptive_rate_limiting": environ_vars.get(
    "BEDROCK_ADAPTIVE_RATE_LIMITING", "false"
).lower() == "true",
```

And add to contract `optional_env_vars`:
```python
"BEDROCK_ADAPTIVE_RATE_LIMITING": "false",
```

And config field:
```python
bedrock_adaptive_rate_limiting: bool = Field(
    default=False,
    description="Enable adaptive rate limiting that auto-tunes based on throttle rate.",
)
```

**Note**: This requires 1 line in contract, 1 field in config, 1 line in config env vars property — minimal.

## P7: Converse API Support (~30 lines)

### What It Does

Adds an alternative invocation path using the Bedrock `converse()` API instead of `invoke_model()`. The Converse API is model-agnostic — works with Claude, Nova, Llama, Mistral without format-specific request bodies.

### Design

Add new method to `BedrockProcessor`:
```python
def _invoke_bedrock_converse(self, prompt: str) -> Dict[str, Any]:
    """Invoke Bedrock using the model-agnostic Converse API."""
    if self.concurrency_mode == "concurrent":
        self._enforce_rate_limit()
        client = self._get_thread_local_bedrock_client()
    else:
        client = self.bedrock_client

    self._circuit_breaker_check()

    messages = [{"role": "user", "content": [{"text": prompt}]}]

    inference_config = {
        "maxTokens": int(self.config["max_tokens"]),
        "temperature": float(self.config["temperature"]),
        "topP": float(self.config["top_p"]),
    }

    kwargs = {
        "modelId": self.effective_model_id,
        "messages": messages,
        "inferenceConfig": inference_config,
    }

    if self.config.get("system_prompt"):
        kwargs["system"] = [{"text": self.config["system_prompt"]}]

    try:
        response = client.converse(**kwargs)
        self._circuit_breaker_record_success()
        self._adapt_rate(was_throttled=False)

        # Extract text from converse response format
        response_text = response["output"]["message"]["content"][0]["text"]

        # Return in same format as invoke_model for compatibility with _parse_response_with_pydantic
        return {"content": [{"text": response_text}]}

    except Exception as e:
        self._circuit_breaker_record_failure()
        if "ThrottlingException" in str(e) or "TooManyRequestsException" in str(e):
            self._adapt_rate(was_throttled=True)

        # Fallback to on-demand model
        fallback_model = self.config.get("fallback_model_id")
        if fallback_model and "ValidationException" in str(e):
            logger.warning(f"Converse API: falling back to {fallback_model}")
            kwargs["modelId"] = fallback_model
            response = client.converse(**kwargs)
            response_text = response["output"]["message"]["content"][0]["text"]
            self._circuit_breaker_record_success()
            return {"content": [{"text": response_text}]}
        raise
```

### Integration in `process_single_case()`

Update the routing logic (currently at line ~1067):
```python
# Current:
if self.config.get("use_structured_output") and self.validation_schema:
    response = self._invoke_bedrock_with_structured_output(prompt)
else:
    response = self._invoke_bedrock(prompt)

# New:
if self.config.get("use_structured_output") and self.validation_schema:
    response = self._invoke_bedrock_with_structured_output(prompt)
elif self.config.get("use_converse_api"):
    response = self._invoke_bedrock_converse(prompt)
else:
    response = self._invoke_bedrock(prompt)
```

### Config/Contract Additions

Contract `optional_env_vars`:
```python
"BEDROCK_USE_CONVERSE_API": "false",
```

Config field:
```python
bedrock_use_converse_api: bool = Field(
    default=False,
    description="Use Converse API (model-agnostic) instead of invoke_model (Anthropic-specific). Enables Nova/Llama/Mistral models.",
)
```

Config env var property:
```python
"BEDROCK_USE_CONVERSE_API": str(self.bedrock_use_converse_api).lower(),
```

Script `main()` config dict:
```python
"use_converse_api": environ_vars.get("BEDROCK_USE_CONVERSE_API", "false").lower() == "true",
```

## Dependency Graph

```
P5 (circuit breaker) ──┐
                        ├── Independent, can run in parallel
P6 (adaptive rate)  ────┘
         │
         ▼
P7 (converse API) ── Uses circuit breaker + adaptive rate in its implementation
         │
         ▼
Verify + commit
```

P5 and P6 are independent of each other but P7 calls both (`_circuit_breaker_check()` + `_adapt_rate()`), so P7 should come after P5+P6.

## Files Modified

| File | Change | Lines |
|------|--------|-------|
| `scripts/bedrock_processing.py` | +3 methods (CB) + 1 method (adaptive) + 1 method (converse) + routing | ~70 |
| `configs/config_bedrock_processing_step.py` | +2 bool fields, update env var property | ~10 |
| `contracts/bedrock_processing_contract.py` | +2 optional_env_vars | ~2 |
| **Total** | | **~82 lines** |

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Circuit breaker blocks all threads simultaneously | Uses lock; half-open state tests one request before closing |
| Adaptive rate oscillates | 60s window + 5% threshold prevents rapid swings; min_rate=1 prevents stalling |
| Converse API response format differs from invoke_model | Wrapped to return same `{"content": [{"text": ...}]}` format for `_parse_response_with_pydantic` compatibility |
| Converse API doesn't support assistant prefilling | Correct — no prefilling in converse mode. Prompt must explicitly request JSON. Use with `use_structured_output=True` (tool_use) or clear JSON instructions in prompt. |
| Thread safety of adaptive rate | `rate_limit_per_second` read is atomic in Python (GIL); window append uses list (GIL-safe for single-threaded append) |

## Verification

1. Syntax check + ruff format
2. Unit test: circuit breaker trips after N failures, recovers after cooldown
3. Unit test: adaptive rate decreases on throttle, increases on success
4. Unit test: converse API wraps response correctly
5. `brazil-build release` + `brazil pb build`
