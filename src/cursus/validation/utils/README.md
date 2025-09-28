# Validation Utilities

This directory contains utility modules for the cursus validation system, providing robust import resolution and other validation support functionality.

## Import Resolver

The `import_resolver.py` module provides a centralized, deployment-agnostic import resolution system that ensures cursus validation scripts work consistently across all deployment environments.

### Features

- **Universal Deployment Compatibility**: Works across PyPI, source, container, and serverless deployments
- **Multiple Resolution Strategies**: Employs 5 different strategies with intelligent fallbacks
- **Deployment-Agnostic Patterns**: Uses relative imports and StepCatalog integration
- **AST-Based Validation**: Safe component discovery without import dependencies
- **Robust Error Handling**: Comprehensive fallback mechanisms with graceful degradation

### Usage

#### Basic Usage

```python
from cursus.validation.utils.import_resolver import ensure_cursus_imports

# Ensure cursus imports work across all deployment scenarios
if not ensure_cursus_imports():
    print("❌ Failed to setup cursus imports")
    sys.exit(1)

# Now you can safely import cursus modules
from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
```

#### Advanced Usage

```python
from cursus.validation.utils.import_resolver import ensure_cursus_imports, get_project_info

# Setup imports with detailed information
success = ensure_cursus_imports()
if success:
    info = get_project_info()
    print(f"Setup successful using: {info}")
else:
    print("Import setup failed")
```

### Resolution Strategies

The import resolver uses the following strategies in order:

1. **Installed Package Import**: Try importing cursus as an installed package
2. **Relative Import Pattern**: Use deployment-agnostic relative imports with package parameter
3. **StepCatalog Integration**: Leverage existing unified discovery system
4. **Development Import Setup**: Add src directory to sys.path (fallback)
5. **Common Pattern Fallbacks**: Try common development directory patterns

### Architecture Compliance

The import resolver is fully compliant with the cursus package portability architecture:

- ✅ **Deployment Agnosticism**: Works identically across all environments
- ✅ **Relative Import Patterns**: Uses `importlib.import_module(relative_path, package=__package__)`
- ✅ **AST-Based Discovery**: Safe component validation before import attempts
- ✅ **StepCatalog Integration**: Leverages unified discovery system
- ✅ **Multiple Fallback Strategies**: Robust error handling with graceful degradation

### Integration with Validation Scripts

Validation scripts should use the import resolver at the beginning:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# Use centralized import resolver for robust import handling
from cursus.validation.utils.import_resolver import ensure_cursus_imports

# Ensure cursus imports work across all deployment scenarios
if not ensure_cursus_imports():
    print("❌ Failed to setup cursus imports")
    sys.exit(1)

# Now safe to import cursus modules
from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester

def main():
    # Validation logic here
    pass

if __name__ == "__main__":
    sys.exit(main())
```

### Benefits

#### Technical Benefits
- **Universal Portability**: Same validation scripts work across all deployment environments
- **Intelligent Resolution**: Automatic path discovery and resolution based on execution context
- **Performance**: Leverages existing O(1) StepCatalog operations when available
- **Reliability**: AST validation prevents import failures

#### Operational Benefits
- **Reduced Maintenance**: No manual path adjustments needed across environments
- **CI/CD Integration**: Validation scripts work seamlessly in automated pipelines
- **Developer Experience**: Simple one-line setup eliminates import configuration complexity
- **Error Clarity**: Clear error messages and fallback strategies

### Testing

The import resolver can be tested directly:

```bash
cd src/cursus/validation/utils
python import_resolver.py
```

This will run comprehensive tests of all resolution strategies and provide detailed information about the current setup.

### Troubleshooting

If import resolution fails:

1. **Check Project Structure**: Ensure you're in a cursus project with `pyproject.toml` and `src/cursus` structure
2. **Verify Installation**: Run `pip install -e .` to install the development package
3. **Check Logs**: The resolver provides detailed debug logging about which strategies were attempted
4. **Test Manually**: Run the import resolver test script to diagnose issues

### Future Enhancements

- Integration with workspace-aware discovery for multi-project environments
- Enhanced caching for improved performance in large-scale deployments
- Additional deployment scenario support (edge computing, specialized containers)
- Integration with external monitoring and observability systems
