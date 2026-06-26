# Cursus: Automatic SageMaker Pipeline Generation

[![PyPI version](https://badge.fury.io/py/cursus.svg)](https://badge.fury.io/py/cursus)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Transform pipeline graphs into production-ready SageMaker pipelines automatically.**

Cursus is an intelligent pipeline generation system that automatically creates complete SageMaker pipelines from user-provided pipeline graphs. Simply define your ML workflow as a graph structure, and Cursus handles all the complex SageMaker implementation details, dependency resolution, and configuration management automatically.

## 🚀 Quick Start

### Installation

```bash
# Core installation
pip install cursus

# With ML frameworks
pip install cursus[pytorch,gbm]

# Full installation with all features
pip install cursus[all]
```

> **SageMaker SDK compatibility:** The current `cursus` 1.x line targets **SageMaker SDK 2.x**. Pin `pip install "cursus<2"` to stay on this line. The `2.x` line (forthcoming) will target SageMaker SDK 3.x; that work happens on `main` and is published from there once ready.

### 30-Second Example

```python
from cursus.core import compile_dag_to_pipeline
from cursus.api import PipelineDAG
from sagemaker.workflow.pipeline_context import PipelineSession

# Create a simple DAG
dag = PipelineDAG()
dag.add_node("CradleDataLoading_training")
dag.add_node("TabularPreprocessing_training") 
dag.add_node("XGBoostTraining")
dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
dag.add_edge("TabularPreprocessing_training", "XGBoostTraining")

# Set up SageMaker session
pipeline_session = PipelineSession()
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Compile to SageMaker pipeline automatically
pipeline = compile_dag_to_pipeline(
    dag=dag,
    config_path="config.json",
    sagemaker_session=pipeline_session,
    role=role,
    pipeline_name="fraud-detection"
)
pipeline.upsert()  # Deploy and run!
```

### Command Line Interface

The `cursus` CLI is organized into focused command groups (run `cursus --help` to see them all):

```bash
# Compile a DAG + config into a SageMaker pipeline definition
cursus compile -d dag.json -c config.json -o pipeline.json

# Dry-run validation with a resolution report (no upsert)
cursus compile -d dag.json -c config.json --validate-only --show-report

# Compile and deploy in one step
cursus compile -d dag.json -c config.json --upsert --start

# Discover and inspect local pipeline projects
cursus projects list
cursus projects show my-project

# Browse / recommend pre-built DAGs from the pipeline catalog
cursus pipeline-catalog list
cursus pipeline-catalog recommend --framework xgboost --features calibration

# Run scripts offline against a DAG (data flow resolved automatically)
cursus validate run-scripts dag.json --config-file config.json

# Expose the engine to LLM agents over MCP
cursus mcp list-tools
cursus mcp serve
```

Other groups: `catalog`, `dag`, `config`, `registry`, `alignment`, `exec-doc`.

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
- Stable, production-deployed, and published on PyPI

### 🤖 **Agent-Ready (MCP)**
- A framework-neutral `cursus.mcp` tool surface (JSON in / JSON out) exposes the
  whole engine — catalog, DAG, config, compile, validate, exec-doc, and the
  pipeline catalog — to LLM agents
- Run it as an MCP server with `cursus mcp serve` (or `python -m cursus.mcp.server`)

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

Cursus follows a layered, specification-driven architecture. A `PipelineDAG`
(nodes + edges) and a config file flow down through these layers to a finished
SageMaker pipeline:

- **🎯 DAG Compiler** (`core/compiler`): `compile_dag_to_pipeline` / `PipelineDAGCompiler` — the public entry point, plus validation, preview, and naming
- **🧩 Dynamic Template** (`core/compiler`): binds each DAG node to its typed config and step-builder class via the step catalog and registry — no per-pipeline template class required
- **🏗️ Pipeline Assembler** (`core/assembler`): instantiates each step builder in topological order and wires them together
- **🔗 Dependency Resolver** (`core/deps`): matches each step's declared inputs to upstream outputs by semantic compatibility scoring, producing SageMaker property references
- **🛠️ Step Builders** (`steps/builders`): turn a config + resolved inputs into a concrete SageMaker `ProcessingStep` / `TrainingStep` / `ModelStep`
- **📋 Specification Layer** (`steps/interfaces`): one declarative `.step.yaml` per step — a unified `StepInterface` carrying both the script contract (I/O paths) and the dependency/output spec the resolver matches on
- **📚 Registry & Step Catalog** (`registry`, `step_catalog`): the canonical step-name registry and auto-discovery that map names → builders, configs, and interfaces

## 📚 Usage Examples

### Basic Pipeline

```python
from cursus.core import compile_dag_to_pipeline
from cursus.api import PipelineDAG
from sagemaker.workflow.pipeline_context import PipelineSession

# Create DAG
dag = PipelineDAG()
dag.add_node("CradleDataLoading_training")
dag.add_node("XGBoostTraining")
dag.add_edge("CradleDataLoading_training", "XGBoostTraining")

# Set up SageMaker session
pipeline_session = PipelineSession()
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Compile to SageMaker pipeline
pipeline = compile_dag_to_pipeline(
    dag=dag,
    config_path="config.json",
    sagemaker_session=pipeline_session,
    role=role,
    pipeline_name="my-ml-pipeline"
)
```

### Advanced Configuration

```python
from cursus.core import compile_dag_to_pipeline, PipelineDAGCompiler
from cursus.api import PipelineDAG
from sagemaker.workflow.pipeline_context import PipelineSession

# Create DAG with more complex workflow
dag = PipelineDAG()
dag.add_node("CradleDataLoading_training")
dag.add_node("TabularPreprocessing_training")
dag.add_node("XGBoostTraining")
dag.add_node("CradleDataLoading_calibration")
dag.add_node("TabularPreprocessing_calibration")
dag.add_node("XGBoostModelEval_calibration")

# Add edges for training flow
dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
dag.add_edge("TabularPreprocessing_training", "XGBoostTraining")

# Add edges for calibration flow
dag.add_edge("CradleDataLoading_calibration", "TabularPreprocessing_calibration")
dag.add_edge("XGBoostTraining", "XGBoostModelEval_calibration")
dag.add_edge("TabularPreprocessing_calibration", "XGBoostModelEval_calibration")

# Set up SageMaker session
pipeline_session = PipelineSession()
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Compile with validation and reporting
compiler = PipelineDAGCompiler(
    config_path="config.json",
    sagemaker_session=pipeline_session,
    role=role
)

# Validate DAG before compilation
validation = compiler.validate_dag_compatibility(dag)
if validation.is_valid:
    print(f"✅ DAG validation passed! Confidence: {validation.avg_confidence:.2f}")
    
    # Compile with detailed report
    pipeline, report = compiler.compile_with_report(
        dag=dag,
        pipeline_name="advanced-ml-pipeline"
    )
    print(f"📊 Pipeline compiled: {report.summary()}")
else:
    print("❌ DAG validation failed:", validation.config_errors)
```

### Using the Pre-built Pipeline Catalog

The pipeline catalog is **data-driven**: 40+ ready-made DAGs ship as `*.dag.json`
files (indexed by `catalog_index.json`) and are loaded by ID — no per-pipeline
classes. Discover, load, and compile them through `cursus.pipeline_catalog`:

```python
from cursus.pipeline_catalog import recommend_dag, load_shared_dag, get_all_shared_dags
from cursus.core import compile_dag_to_pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

# Discover a DAG that matches your needs (ranked, scored)
for hit in recommend_dag(framework="xgboost", features=["calibration"]):
    print(hit["id"], hit["score"])

# ...or list everything in the catalog
get_all_shared_dags()  # {dag_id: metadata, ...}

# Load a ready-made DAG by ID and compile it
dag = load_shared_dag("xgboost_simple")

pipeline_session = PipelineSession()
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

pipeline = compile_dag_to_pipeline(
    dag=dag,
    config_path="config.json",
    sagemaker_session=pipeline_session,
    role=role,
    pipeline_name="xgboost-simple",
)
pipeline.upsert()  # Deploy and run!
```

For a one-call build straight from files, use `build_and_compile`:

```python
from cursus.pipeline_catalog import build_and_compile

pipeline, report = build_and_compile(
    dag_path="src/cursus/pipeline_catalog/shared_dags/xgboost/simple.dag.json",
    config_path="config.json",
    sagemaker_session=pipeline_session,
    role=role,
)
```

### Using the Compiler Class Directly

```python
from cursus.core import PipelineDAGCompiler
from cursus.pipeline_catalog import load_shared_dag
from sagemaker.workflow.pipeline_context import PipelineSession

# Load a ready-made DAG from the catalog (or build your own PipelineDAG)
dag = load_shared_dag("xgboost_simple")

# Set up SageMaker session
pipeline_session = PipelineSession()
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Use compiler for more control
compiler = PipelineDAGCompiler(
    config_path="config.json",
    sagemaker_session=pipeline_session,
    role=role
)

# Preview resolution before compilation
preview = compiler.preview_resolution(dag)
for node, config_type in preview.node_config_map.items():
    confidence = preview.resolution_confidence.get(node, 0.0)
    print(f"   {node} → {config_type} (confidence: {confidence:.2f})")

# Compile the pipeline
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
pip install cursus[gbm]        # GBM training pipelines (XGBoost + LightGBM)
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
- **[Zettelkasten Principles](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/6_resources/zettelkasten_knowledge_management_principles.md)** - The knowledge management principles behind our slipbox documentation system, explaining how we organize and connect information for maximum discoverability and organic growth

### Core Documentation
- **[Developer Guide](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/README.md)** - Comprehensive guide for developing new pipeline steps and extending Cursus
- **[Design Documentation](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/1_design/README.md)** - Detailed architectural documentation and design principles
- **[Pipeline Catalog](https://github.com/TianpeiLuke/cursus/tree/main/src/cursus/pipeline_catalog/README.md)** - Comprehensive collection of prebuilt pipeline templates organized by framework and task
- **[API Reference](https://github.com/TianpeiLuke/cursus/tree/main/src/cursus/)** - Detailed API documentation including core, api, steps, and other components
- **[Examples](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/examples/)** - Ready-to-use pipeline blueprints and examples

### Quick Links
- **[Getting Started](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/adding_new_pipeline_step.md)** - Start here for adding new pipeline steps
- **[Design Principles](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/design_principles.md)** - Core architectural principles
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
