# AutoPipe PyPI Package Conversion Plan

**Package Name**: `autopipe`  
**Tagline**: "Automatic SageMaker Pipeline Generation"  
**Version**: 1.0.0  
**License**: MIT  

## Executive Summary

Convert the nlp-pipeline repository into a production-ready PyPI package called `autopipe` that enables automatic SageMaker pipeline generation from DAG specifications. The package will emphasize the core value proposition of 10x faster development through intelligent automation.

## 1. Package Structure & Organization

### 1.1 Current Structure Analysis
```
src/
├── pipeline_api/          # Core API - Main entry point
├── pipeline_builder/      # Template system
├── pipeline_dag/          # DAG construction
├── pipeline_deps/         # Dependency resolution
├── pipeline_registry/     # Component registration
├── pipeline_steps/        # Step builders
├── pipeline_step_specs/   # Step specifications
├── config_field_manager/  # Configuration management
├── lightning_models/      # PyTorch models
├── processing/           # Data processing
└── [other modules...]
```

### 1.2 Proposed Package Structure
```
autopipe/
├── __init__.py           # Main exports and version
├── api/                  # High-level user API (from pipeline_api)
├── core/                 # Core functionality
│   ├── dag/             # DAG system (from pipeline_dag)
│   ├── builder/         # Pipeline builder (from pipeline_builder)
│   ├── deps/            # Dependency resolution (from pipeline_deps)
│   └── registry/        # Component registry (from pipeline_registry)
├── steps/               # Step implementations (from pipeline_steps + specs)
├── config/              # Configuration management (from config_field_manager)
├── models/              # ML models (from lightning_models)
├── processing/          # Data processing utilities
├── validation/          # Pipeline validation (from pipeline_validation)
├── cli/                 # Command-line interface
└── utils/               # Utilities and helpers
```

## 2. Package Configuration

### 2.1 pyproject.toml Configuration
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "autopipe"
version = "1.0.0"
description = "Automatic SageMaker Pipeline Generation from DAG Specifications"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Tianpei Xie", email = "your-email@example.com"}
]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Data Scientists",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
keywords = ["sagemaker", "pipeline", "dag", "machine-learning", "aws", "automation"]
requires-python = ">=3.8"

# Core dependencies (minimal for basic functionality)
dependencies = [
    "boto3>=1.39.0",
    "sagemaker>=2.240.0",
    "pydantic>=2.0.0",
    "networkx>=3.0",
    "pyyaml>=6.0",
    "click>=8.0.0",
]

[project.optional-dependencies]
# ML Framework dependencies
pytorch = [
    "torch>=2.0.0",
    "pytorch-lightning>=2.0.0",
    "torchmetrics>=1.0.0",
]
xgboost = [
    "xgboost>=2.0.0",
    "scikit-learn>=1.3.0",
]
nlp = [
    "transformers>=4.30.0",
    "spacy>=3.7.0",
    "tokenizers>=0.15.0",
]
# Development dependencies
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]
# Full installation
all = [
    "autopipe[pytorch,xgboost,nlp]"
]

[project.scripts]
autopipe = "autopipe.cli:main"

[project.urls]
Homepage = "https://github.com/TianpeiLuke/nlp-pipeline"
Documentation = "https://github.com/TianpeiLuke/nlp-pipeline/blob/main/README.md"
Repository = "https://github.com/TianpeiLuke/nlp-pipeline"
Issues = "https://github.com/TianpeiLuke/nlp-pipeline/issues"
```

### 2.2 Dependency Strategy
- **Core Dependencies**: Minimal set for basic DAG compilation
- **Optional Dependencies**: Heavy ML frameworks grouped by use case
- **Version Flexibility**: Minimum versions with reasonable upper bounds
- **Installation Options**:
  ```bash
  pip install autopipe                    # Core only
  pip install autopipe[pytorch]          # With PyTorch
  pip install autopipe[xgboost]          # With XGBoost  
  pip install autopipe[all]              # Everything
  pip install autopipe[dev]              # Development tools
  ```

## 3. Main Package API Design

### 3.1 Primary Entry Points
```python
# autopipe/__init__.py
from .api import (
    compile_dag,
    compile_dag_to_pipeline,
    PipelineDAGCompiler,
    create_pipeline_from_dag,
)
from .core.dag import PipelineDAG
from .version import __version__

__all__ = [
    "compile_dag",
    "compile_dag_to_pipeline", 
    "PipelineDAGCompiler",
    "create_pipeline_from_dag",
    "PipelineDAG",
    "__version__",
]
```

### 3.2 User-Friendly API
```python
# Simple usage
import autopipe
pipeline = autopipe.compile_dag(my_dag)

# Advanced usage
from autopipe import PipelineDAGCompiler
compiler = PipelineDAGCompiler(config_path="config.yaml")
pipeline = compiler.compile(my_dag, pipeline_name="fraud-detection")

# Fluent API (if available)
from autopipe.fluent import Pipeline
pipeline = (Pipeline("fraud-detection")
    .load_data("s3://data/")
    .train_xgboost()
    .evaluate()
    .deploy())
```

## 4. Command Line Interface

### 4.1 CLI Commands
```bash
# Compile DAG to pipeline
autopipe compile my_dag.py --output pipeline.json --name my-pipeline

# Validate DAG structure
autopipe validate my_dag.py

# Preview compilation results
autopipe preview my_dag.py

# Show supported step types
autopipe list-steps

# Generate example DAG
autopipe init --template xgboost --name my-project
```

### 4.2 CLI Implementation
```python
# autopipe/cli/__init__.py
import click
from ..api import compile_dag_to_pipeline, PipelineDAGCompiler

@click.group()
@click.version_option()
def main():
    """AutoPipe: Automatic SageMaker Pipeline Generation"""
    pass

@main.command()
@click.argument('dag_file')
@click.option('--output', '-o', help='Output file path')
@click.option('--name', '-n', help='Pipeline name')
def compile(dag_file, output, name):
    """Compile DAG file to SageMaker pipeline"""
    # Implementation here
    pass
```

## 5. Documentation Strategy

### 5.1 Package README
- **Quick Start**: 30-second example
- **Installation**: Different installation options
- **Basic Usage**: Core API examples
- **Advanced Features**: Power user capabilities
- **Links**: Full documentation, examples, contributing

### 5.2 API Documentation
- **Docstrings**: Comprehensive function/class documentation
- **Type Hints**: Full type annotation coverage
- **Examples**: Code examples in docstrings
- **Sphinx**: Auto-generated API docs (future)

## 6. Testing Strategy

### 6.1 Test Structure
```
tests/
├── unit/                 # Unit tests for individual modules
├── integration/          # Integration tests
├── examples/            # Example DAG tests
├── fixtures/            # Test data and fixtures
└── conftest.py          # Pytest configuration
```

### 6.2 Test Coverage Goals
- **Core API**: 95%+ coverage
- **DAG Compilation**: 90%+ coverage  
- **Step Builders**: 85%+ coverage
- **CLI**: 80%+ coverage

## 7. Build & Release Process

### 7.1 Build Configuration
```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Upload to TestPyPI (testing)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

### 7.2 Release Checklist
- [ ] Update version number
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Build package locally
- [ ] Test installation from built package
- [ ] Upload to TestPyPI
- [ ] Test installation from TestPyPI
- [ ] Upload to PyPI
- [ ] Create GitHub release
- [ ] Update documentation

## 8. Migration & Compatibility

### 8.1 Import Path Migration
```python
# Old imports (current)
from src.pipeline_api import compile_dag_to_pipeline
from src.pipeline_dag import PipelineDAG

# New imports (autopipe package)
from autopipe import compile_dag_to_pipeline
from autopipe.core.dag import PipelineDAG
```

### 8.2 Backward Compatibility
- **Alias Support**: Provide import aliases for common patterns
- **Deprecation Warnings**: Gradual migration path
- **Documentation**: Clear migration guide

## 9. Quality Assurance

### 9.1 Code Quality Tools
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing framework

### 9.2 CI/CD Pipeline (Future)
- **GitHub Actions**: Automated testing
- **Multiple Python Versions**: 3.8, 3.9, 3.10, 3.11
- **Multiple OS**: Linux, macOS, Windows
- **Automated Releases**: Tag-based releases

## 10. Implementation Timeline

### Phase 1: Core Package Structure (Week 1)
- [ ] Create pyproject.toml
- [ ] Restructure src/ to autopipe/
- [ ] Create main __init__.py with exports
- [ ] Basic CLI implementation
- [ ] Update imports throughout codebase

### Phase 2: Documentation & Testing (Week 2)  
- [ ] Package README.md
- [ ] API documentation
- [ ] Basic test suite
- [ ] Example scripts

### Phase 3: Build & Release (Week 3)
- [ ] Build configuration
- [ ] TestPyPI release
- [ ] Testing and refinement
- [ ] Production PyPI release

### Phase 4: Post-Release (Ongoing)
- [ ] Community feedback integration
- [ ] Documentation improvements
- [ ] Additional features
- [ ] Performance optimizations

## 11. Success Metrics

### 11.1 Technical Metrics
- **Installation Success Rate**: >95%
- **Import Success Rate**: >99%
- **Test Coverage**: >90%
- **Documentation Coverage**: >85%

### 11.2 Adoption Metrics
- **PyPI Downloads**: Track monthly downloads
- **GitHub Stars**: Community engagement
- **Issue Resolution**: Response time <48 hours
- **User Feedback**: Satisfaction surveys

## 12. Risk Mitigation

### 12.1 Dependency Conflicts
- **Flexible Versioning**: Avoid overly strict version pins
- **Optional Dependencies**: Keep core lightweight
- **Testing Matrix**: Test with different dependency versions

### 12.2 Breaking Changes
- **Semantic Versioning**: Follow semver strictly
- **Deprecation Policy**: 2-version deprecation cycle
- **Migration Guides**: Clear upgrade documentation

## 13. Next Steps

1. **Create pyproject.toml** with the configuration above
2. **Restructure package** from src/ to autopipe/
3. **Implement CLI** with basic commands
4. **Create package README** with quick start guide
5. **Build and test** package locally
6. **Release to TestPyPI** for validation
7. **Production release** to PyPI

---

**Package Vision**: Make AutoPipe the go-to solution for automatic SageMaker pipeline generation, emphasizing the 10x development speed improvement and intelligent automation that eliminates manual configuration complexity.
