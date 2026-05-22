"""
Cursus: Automatic SageMaker Pipeline Generation

Transform pipeline graphs into production-ready SageMaker pipelines automatically.
An intelligent pipeline generation system that automatically creates complete SageMaker
pipelines from user-provided pipeline graphs with intelligent dependency resolution
and configuration management.

Key Features:
- 🎯 Graph-to-Pipeline Automation: Automatically generate complete SageMaker pipelines
- ⚡ 10x Faster Development: Minutes to working pipeline vs. weeks of manual configuration
- 🧠 Intelligent Dependency Resolution: Automatic step connections and data flow
- 🛡️ Production Ready: Built-in quality gates and validation
- 📈 Proven Results: 60% average code reduction across pipeline components

Basic Usage:
    >>> import cursus
    >>> pipeline = cursus.compile_dag(my_dag)

    >>> from cursus import PipelineDAGCompiler
    >>> compiler = PipelineDAGCompiler()
    >>> pipeline = compiler.compile(my_dag, pipeline_name="fraud-detection")

Advanced Usage:
    >>> from cursus.core.dag import PipelineDAG
    >>> from cursus.api import compile_dag_to_pipeline
    >>>
    >>> dag = PipelineDAG()
    >>> # ... build your DAG
    >>> pipeline = compile_dag_to_pipeline(dag, config_path="config.yaml")
"""

# Package metadata — pyproject.toml is the single source of truth.
# - Version: VERSION file at repo root is checked first (live source of truth,
#   declared in pyproject as `version = {file = "VERSION"}`), so an updated
#   VERSION wins over stale installed metadata. Falls back to importlib.metadata
#   when running an installed wheel without the source tree.
# - Title / description / author: read from importlib.metadata, which picks
#   them up from pyproject.toml at install time. No values are hardcoded here.
def _resolve_metadata():
    from pathlib import Path

    title = version = description = author = None

    try:
        from importlib.metadata import metadata as _meta_lookup

        _m = _meta_lookup("cursus")
        title = _m.get("Name")
        version = _m.get("Version")
        description = _m.get("Summary")
        author = _m.get("Author") or _m.get("Author-email")
    except Exception:
        pass

    # VERSION file takes priority for version (source-of-truth in dev mode).
    _v_file = Path(__file__).resolve().parent.parent.parent / "VERSION"
    if _v_file.exists():
        try:
            text = _v_file.read_text().strip()
            if text:
                version = text
        except (OSError, IOError):
            pass

    return title, version, description, author


__title__, __version__, __description__, __author__ = _resolve_metadata()

# Core API exports - main user interface
try:
    from .core.compiler import (
        DynamicPipelineTemplate,
        PipelineDAGCompiler,
        compile_dag_to_pipeline,
    )
    from .core.compiler import compile_dag_to_pipeline as compile_dag
except ImportError as e:
    # Graceful degradation if dependencies are missing
    import warnings

    warnings.warn(f"Some Cursus features may not be available: {e}")

    def compile_dag(*args, **kwargs):
        raise ImportError(
            "Core Cursus dependencies not available. Please install with: pip install cursus[all]"
        )

    def compile_dag_to_pipeline(*args, **kwargs):
        raise ImportError(
            "Core Cursus dependencies not available. Please install with: pip install cursus[all]"
        )

    class PipelineDAGCompiler:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Core Cursus dependencies not available. Please install with: pip install cursus[all]"
            )


# Core data structures
try:
    from .api.dag import EnhancedPipelineDAG, PipelineDAG
except ImportError:

    class PipelineDAG:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DAG functionality not available. Please install with: pip install cursus[all]"
            )

    class EnhancedPipelineDAG:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Enhanced DAG functionality not available. Please install with: pip install cursus[all]"
            )


# Convenience function for quick pipeline creation
def create_pipeline_from_dag(dag, pipeline_name=None, **kwargs):
    """
    Create a SageMaker pipeline from a DAG specification.

    This is a convenience function that combines DAG compilation and pipeline creation
    in a single call with sensible defaults.

    Args:
        dag: PipelineDAG instance or DAG specification
        pipeline_name: Optional name for the pipeline
        **kwargs: Additional arguments passed to the compiler

    Returns:
        SageMaker Pipeline instance ready for execution

    Example:
        >>> dag = PipelineDAG()
        >>> # ... configure your DAG
        >>> pipeline = create_pipeline_from_dag(dag, "my-ml-pipeline")
        >>> pipeline.start()
    """
    return compile_dag_to_pipeline(dag, pipeline_name=pipeline_name, **kwargs)


# Public API
__all__ = [
    # Main API functions
    "compile_dag",
    "compile_dag_to_pipeline",
    "create_pipeline_from_dag",
    # Core classes
    "PipelineDAGCompiler",
    "PipelineDAG",
    "EnhancedPipelineDAG",
    "DynamicPipelineTemplate",
]

# Package metadata for introspection
__package_info__ = {
    "name": __title__,
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "license": "MIT",
    "python_requires": ">=3.8",
    "keywords": [
        "sagemaker",
        "pipeline",
        "dag",
        "machine-learning",
        "aws",
        "automation",
    ],
}

# Auto-install NVMe-aware security patch for MODSWorkflowHelper.
# Prevents VolumeKmsKeyId injection on NVMe GPU instances (ml.p4d, ml.g5, etc.).
# Ref: OfficeHour-1553. Remove once MODSWorkflowHelper ships its own NVMe gate.
try:
    from .core.utils.nvme_security import install_nvme_aware_security_patch

    install_nvme_aware_security_patch()
except Exception:
    pass
