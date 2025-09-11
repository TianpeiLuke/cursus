---
tags:
  - entry_point
  - tutorial
  - registry
  - step_registration
  - documentation
keywords:
  - registry system
  - step registration
  - step discovery
  - workspace awareness
  - unified registry
  - builder registry
  - step names registry
  - registry tutorials
topics:
  - registry system
  - step management
  - workspace integration
  - tutorial documentation
language: python
date of note: 2025-09-06
---

# Registry Tutorials

Comprehensive tutorials and documentation for the Cursus Registry System.

## Overview

The Cursus Registry System provides centralized step registration, discovery, and management capabilities with workspace awareness and hybrid backend support. These tutorials cover everything from basic usage to advanced customization.

## Tutorial Structure

### 📚 Quick Start
- **[Registry Quick Start](./registry_quick_start.md)** - 20-minute comprehensive tutorial covering all essential registry operations

### 📖 API Reference  
- **[Registry API Reference](./registry_api_reference.md)** - Complete API documentation with examples for all registry functions and classes

## What You'll Learn

### Core Concepts
- **Step Registration**: How to register steps permanently and temporarily
- **Step Discovery**: Finding and exploring available steps
- **Workspace Awareness**: Managing steps across different workspace contexts
- **JSON Export**: Publishing registry data for sharing and reuse
- **Validation**: Built-in validation and standardization enforcement

### Advanced Features
- **Hybrid Registry**: UnifiedRegistryManager with workspace integration
- **Builder Registry**: Automatic step builder discovery and registration
- **Custom Backends**: Creating custom registry storage backends
- **Batch Operations**: Efficient bulk registry operations
- **Monitoring**: Registry access and performance monitoring

## Getting Started

1. **New to Registry?** Start with [Registry Quick Start](./registry_quick_start.md)
2. **Need API Details?** Check [Registry API Reference](./registry_api_reference.md)
3. **Advanced Usage?** Explore the advanced sections in the API reference

## Prerequisites

- Basic understanding of Python and Cursus framework
- Familiarity with step builders and pipeline concepts
- Optional: Workspace setup for workspace-aware features

## Key Registry Components

### Core Registry (`cursus.registry`)
```python
from cursus.registry import get_step_names, register_step, get_available_steps
```

### Step Names Registry (`cursus.registry.step_names`)
```python
from cursus.registry.step_names import StepNamesRegistry
```

### Builder Registry (`cursus.registry.builder_registry`)
```python
from cursus.registry.builder_registry import StepBuilderRegistry
```

### Unified Registry (`cursus.registry.unified`)
```python
from cursus.registry.unified import UnifiedRegistryManager
```

## Common Use Cases

### 1. Explore Available Steps
```python
from cursus.registry import get_available_steps
steps = get_available_steps()
print(f"Available steps: {steps}")
```

### 2. Register Custom Step
```python
from cursus.registry import register_step

definition = {
    'config_class': 'CustomConfig',
    'builder_step_name': 'custom_builder',
    'spec_type': 'custom',
    'sagemaker_step_type': 'ProcessingStep',
    'description': 'Custom processing step'
}

register_step('custom_step', definition)
```

### 3. Export Registry to JSON
```python
from cursus.registry.export import export_registry_to_json
import json

registry_data = export_registry_to_json()
with open('my_registry.json', 'w') as f:
    json.dump(registry_data, f, indent=2)
```

### 4. Workspace-Aware Operations
```python
from cursus.registry.workspace import WorkspaceRegistryManager

ws_manager = WorkspaceRegistryManager('/path/to/workspace')
ws_steps = ws_manager.get_workspace_steps()
```

## Registry Architecture

```
Registry System
├── Core Registry (step_names.py)
│   ├── STEP_NAMES dictionary
│   ├── get_step_names()
│   └── Backward compatibility
├── Builder Registry (builder_registry.py)
│   ├── StepBuilderRegistry
│   ├── Auto-discovery
│   └── Validation
├── Unified Manager (unified.py)
│   ├── UnifiedRegistryManager
│   ├── Workspace awareness
│   └── Hybrid backend
└── Workspace Integration
    ├── Context management
    ├── Scope control
    └── Synchronization
```

## File Organization

```
slipbox/5_tutorials/registry/
├── README.md                    # This overview
├── registry_quick_start.md      # 20-minute tutorial
└── registry_api_reference.md    # Complete API reference
```

## Related Documentation

- [Step Builder Registry Guide](../../0_developer_guide/step_builder_registry_guide.md)
- [Hybrid Registry Design](../../1_design/hybrid_registry_standardization_enforcement_design.md)
- [Workspace-Aware Registry Design](../../1_design/workspace_aware_distributed_registry_design.md)
- [Workspace Tutorials](../workspace/)

## Support and Troubleshooting

### Common Issues

1. **Step Not Found**: Check step name spelling and registry scope
2. **Registration Failed**: Verify step definition format and validation
3. **Workspace Context**: Ensure workspace path is correctly set
4. **Import Errors**: Check module imports and dependencies

### Debug Functions

```python
from cursus.registry.debug import debug_step_resolution
from cursus.registry.utils import validate_registry_integrity

# Debug step resolution
debug_info = debug_step_resolution('step_name')

# Validate registry integrity
is_valid, issues = validate_registry_integrity()
```

### Getting Help

- Check the troubleshooting sections in tutorials
- Review error messages and validation feedback
- Use debug functions to trace issues
- Consult the API reference for detailed parameter information

---

*Happy registering! 🚀*
