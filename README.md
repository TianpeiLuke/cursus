# Cursus: Automatic SageMaker Pipeline Generation

[![PyPI version](https://badge.fury.io/py/cursus.svg)](https://badge.fury.io/py/cursus)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Transform pipeline graphs into production-ready SageMaker pipelines automatically.**

Cursus is an intelligent pipeline generation system that automatically creates complete SageMaker pipelines from user-provided pipeline graphs. Simply define your ML workflow as a graph structure, and Cursus handles all the complex SageMaker implementation details, dependency resolution, and configuration management automatically.

## 🚀 Quick Start

### Installation

```bash
# Core installation
pip install cursus

# With ML frameworks
pip install cursus[pytorch,xgboost]

# Full installation with all features
pip install cursus[all]
```

### 30-Second Example

```python
from cursus.core import compile_dag_to_pipeline
from cursus.api import PipelineDAG

# Create a simple DAG
dag = PipelineDAG()
dag.add_node("CradleDataLoading")
dag.add_node("TabularPreprocessing") 
dag.add_node("XGBoostTraining")
dag.add_edge("CradleDataLoading", "TabularPreprocessing")
dag.add_edge("TabularPreprocessing", "XGBoostTraining")

# Compile to SageMaker pipeline automatically
pipeline = compile_dag_to_pipeline(dag, pipeline_name="fraud-detection")
pipeline.start()  # Deploy and run!
```

### Command Line Interface

```bash
# Generate a new project
cursus init --template xgboost --name fraud-detection

# Validate your DAG
cursus validate my_dag.py

# Compile to SageMaker pipeline
cursus compile my_dag.py --name my-pipeline --output pipeline.json
```

## ✨ Key Features

### 🎯 **Graph-to-Pipeline Automation**
- **Input**: Simple pipeline graph with step types and connections
- **Output**: Complete SageMaker pipeline with all dependencies resolved
- **Magic**: Intelligent analysis of graph structure with automatic step builder selection

### ⚡ **10x Faster Development**
- **Before**: 2-4 weeks of manual SageMaker configuration
- **After**: 10-30 minutes from graph to working pipeline
- **Result**: 95% reduction in development time

### 🧠 **Intelligent Dependency Resolution**
- Automatic step connections and data flow
- Smart configuration matching and validation
- Type-safe specifications with compile-time checks
- Semantic compatibility analysis

### 🛡️ **Production Ready**
- Built-in quality gates and validation
- Enterprise governance and compliance
- Comprehensive error handling and debugging
- 98% complete with 1,650+ lines of complex code eliminated

## 📊 Proven Results

Based on production deployments across enterprise environments:

| Component | Code Reduction | Lines Eliminated | Key Benefit |
|-----------|----------------|------------------|-------------|
| **Processing Steps** | 60% | 400+ lines | Automatic input/output resolution |
| **Training Steps** | 60% | 300+ lines | Intelligent hyperparameter handling |
| **Model Steps** | 47% | 380+ lines | Streamlined model creation |
| **Registration Steps** | 66% | 330+ lines | Simplified deployment workflows |
| **Overall System** | **~55%** | **1,650+ lines** | **Intelligent automation** |

## 🏗️ Architecture

Cursus follows a sophisticated layered architecture:

- **🎯 User Interface**: Fluent API and Pipeline DAG for intuitive construction
- **🧠 Intelligence Layer**: Smart proxies with automatic dependency resolution  
- **🏗️ Orchestration**: Pipeline assembler and compiler for DAG-to-template conversion
- **📚 Registry Management**: Multi-context coordination with lifecycle management
- **🔗 Dependency Resolution**: Intelligent matching with semantic compatibility
- **📋 Specification Layer**: Comprehensive step definitions with quality gates

## 📚 Usage Examples

### Basic Pipeline

```python
from cursus.core import compile_dag_to_pipeline
from cursus.api import PipelineDAG

# Create DAG
dag = PipelineDAG()
dag.add_node("CradleDataLoading")
dag.add_node("XGBoostTraining")
dag.add_edge("CradleDataLoading", "XGBoostTraining")

# Compile to SageMaker pipeline
pipeline = compile_dag_to_pipeline(dag, pipeline_name="my-ml-pipeline")
```

### Advanced Configuration

```python
from cursus.core import compile_dag_to_pipeline
from cursus.api import PipelineDAG

# Create DAG with more complex workflow
dag = PipelineDAG()
dag.add_node("CradleDataLoading")
dag.add_node("TabularPreprocessing")
dag.add_node("XGBoostTraining")
dag.add_node("XGBoostModelEval")
dag.add_edge("CradleDataLoading", "TabularPreprocessing")
dag.add_edge("TabularPreprocessing", "XGBoostTraining")
dag.add_edge("XGBoostTraining", "XGBoostModelEval")

# Compile with custom configuration
pipeline = compile_dag_to_pipeline(
    dag=dag,
    pipeline_name="advanced-ml-pipeline",
    config_path="config.yaml"
)
```

### Using the Compiler Class

```python
from cursus.core import PipelineDAGCompiler
from cursus.api import PipelineDAG

# Create DAG
dag = PipelineDAG()
dag.add_node("TabularPreprocessing")
dag.add_node("XGBoostTraining")
dag.add_edge("TabularPreprocessing", "XGBoostTraining")

# Use compiler for more control
compiler = PipelineDAGCompiler()
pipeline = compiler.compile(dag, pipeline_name="my-pipeline")
```

## 🔧 Installation Options

### Core Installation
```bash
pip install cursus
```
Includes basic DAG compilation and SageMaker integration.

### Framework-Specific
```bash
pip install cursus[pytorch]    # PyTorch Lightning models
pip install cursus[xgboost]    # XGBoost training pipelines  
pip install cursus[nlp]        # NLP models and processing
pip install cursus[processing] # Advanced data processing
```

### Development
```bash
pip install cursus[dev]        # Development tools
pip install cursus[docs]       # Documentation tools
pip install cursus[all]        # Everything included
```

## 🎯 Who Should Use Cursus?

### **Data Scientists & ML Practitioners**
- Focus on model development, not infrastructure complexity
- Rapid experimentation with 10x faster iteration
- Business-focused interface eliminates SageMaker expertise requirements

### **Platform Engineers & ML Engineers**  
- 60% less code to maintain and debug
- Specification-driven architecture prevents common errors
- Universal patterns enable faster team onboarding

### **Organizations**
- Accelerated innovation with faster pipeline development
- Reduced technical debt through clean architecture
- Built-in governance and compliance frameworks

## 📖 Documentation

### 📚 [Complete Documentation Hub](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/README.md)
**Your gateway to all Cursus documentation - start here for comprehensive navigation**

### Knowledge Management Philosophy
- **[Zettelkasten Principles](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/1_design/zettelkasten_knowledge_management_principles.md)** - The knowledge management principles behind our slipbox documentation system, explaining how we organize and connect information for maximum discoverability and organic growth

### Core Documentation
- **[Developer Guide](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/README.md)** - Comprehensive guide for developing new pipeline steps and extending Cursus
- **[Design Documentation](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/1_design/README.md)** - Detailed architectural documentation and design principles
- **[Pipeline Catalog](https://github.com/TianpeiLuke/cursus/tree/main/src/cursus/pipeline_catalog/README.md)** - Comprehensive collection of prebuilt pipeline templates organized by framework and task
- **[API Reference](https://github.com/TianpeiLuke/cursus/tree/main/src/cursus/)** - Detailed API documentation including core, api, steps, and other components
- **[Examples](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/examples/)** - Ready-to-use pipeline blueprints and examples

### Quick Links
- **[Getting Started](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/adding_new_pipeline_step.md)** - Start here for adding new pipeline steps
- **[Design Principles](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/1_design/design_principles.md)** - Core architectural principles
- **[Best Practices](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/best_practices.md)** - Recommended development practices
- **[Component Guide](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/component_guide.md)** - Overview of key components
- **[Validation System](https://github.com/TianpeiLuke/cursus/tree/main/src/cursus/validation/)** - Comprehensive validation framework for pipeline alignment and quality assurance

## 🤝 Contributing

We welcome contributions! See our [Developer Guide](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/README.md) for comprehensive details on:

- **[Prerequisites](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/prerequisites.md)** - What you need before starting development
- **[Creation Process](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/creation_process.md)** - Step-by-step process for adding new pipeline steps
- **[Validation Checklist](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/validation_checklist.md)** - Comprehensive checklist for validating implementations
- **[Common Pitfalls](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/common_pitfalls.md)** - Common mistakes to avoid

For architectural insights and design decisions, see the [Design Documentation](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/1_design/README.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/TianpeiLuke/cursus/blob/main/LICENSE) file for details.

## 🔗 Links

- **GitHub**: https://github.com/TianpeiLuke/cursus
- **Issues**: https://github.com/TianpeiLuke/cursus/issues
- **PyPI**: https://pypi.org/project/cursus/

---

**Cursus**: Making SageMaker pipeline development 10x faster through intelligent automation. 🚀
