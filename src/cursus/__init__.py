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

# Import version info with fallback for missing __version__.py (e.g., if gitignored)
try:
    from .__version__ import __version__, __title__, __description__, __author__
except ImportError:
    # Fallback: read directly from VERSION file or use defaults
    import os
    from pathlib import Path

    def _get_fallback_version():
        """Get version from VERSION file as fallback."""
        current_dir = Path(__file__).parent
        version_file_paths = [
            # Try project root (2 levels up from src/cursus/__init__.py)
            current_dir.parent.parent / "VERSION",
            # Try relative to current directory
            current_dir / "VERSION",
            # Try one level up
            current_dir.parent / "VERSION",
        ]

        for version_file in version_file_paths:
            try:
                if version_file.exists():
                    return version_file.read_text().strip()
            except (OSError, IOError):
                continue

        # Final fallback
        return "1.2.3"

    __version__ = _get_fallback_version()
    __title__ = "cursus"
    __description__ = "Automatic SageMaker Pipeline Generation from DAG Specifications"
    __author__ = "Tianpei Xie"

# Core API exports - main user interface
try:
    from .core.compiler import compile_dag_to_pipeline, PipelineDAGCompiler
    from .core.compiler import compile_dag_to_pipeline as compile_dag
    from .core.compiler import DynamicPipelineTemplate
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
    from .api.dag import PipelineDAG
except ImportError:

    class PipelineDAG:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DAG functionality not available. Please install with: pip install cursus[all]"
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
    # Version info
    "__version__",
    "__title__",
    "__description__",
    "__author__",
    # Main API functions
    "compile_dag",
    "compile_dag_to_pipeline",
    "create_pipeline_from_dag",
    # Core classes
    "PipelineDAGCompiler",
    "PipelineDAG",
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
