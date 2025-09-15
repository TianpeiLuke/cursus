"""
MODS Pipeline API

This module provides an API to convert regular pipelines into MODS-enhanced pipelines
by applying the MODSTemplate decorator. This eliminates the need for duplicate pipeline
definitions between pipeline_catalog/pipelines and pipeline_catalog/mods_pipelines.

The API extracts AUTHOR, PIPELINE_DESCRIPTION, and PIPELINE_VERSION from the base
configuration provided by the user, following the pattern established in mods_pipeline_adapter.py.

Example:
    ```python
    from cursus.pipeline_catalog.mods_api import create_mods_pipeline
    from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import XGBoostE2EComprehensivePipeline
    
    # Create MODS-enhanced pipeline from regular pipeline
    MODSXGBoostPipeline = create_mods_pipeline(
        XGBoostE2EComprehensivePipeline,
        author="lukexie",
        pipeline_description="XGBoost E2E Pipeline with MODS",
        pipeline_version="1.2.3"
    )
    
    # Or extract from config
    MODSXGBoostPipeline = create_mods_pipeline_from_config(
        XGBoostE2EComprehensivePipeline,
        config_path="path/to/config.json"
    )
    ```
"""

import json
import logging
from pathlib import Path
from typing import Type, Optional, Dict, Any, Union
from functools import wraps

# MODS template import
try:
    from mods.mods_template import MODSTemplate
    MODS_AVAILABLE = True
except ImportError:
    MODS_AVAILABLE = False
    # Create a dummy decorator for when MODS is not available
    def MODSTemplate(author: str, description: str, version: str):
        def decorator(cls):
            # Store MODS metadata as class attributes
            cls._mods_author = author
            cls._mods_description = description
            cls._mods_version = version
            return cls
        return decorator

from ..core.base.config_base import BasePipelineConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mods_pipeline(
    pipeline_class: Type,
    author: str,
    pipeline_description: str,
    pipeline_version: str
) -> Type:
    """
    Create a MODS-enhanced pipeline class from a regular pipeline class.
    
    This function applies the MODSTemplate decorator to a regular pipeline class,
    creating a MODS-enhanced version without code duplication.
    
    Args:
        pipeline_class: The regular pipeline class to enhance
        author: Author or owner of the pipeline
        pipeline_description: Description of the pipeline
        pipeline_version: Version string for the pipeline
        
    Returns:
        Type: MODS-enhanced pipeline class
        
    Example:
        ```python
        from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import XGBoostE2EComprehensivePipeline
        
        MODSXGBoostPipeline = create_mods_pipeline(
            XGBoostE2EComprehensivePipeline,
            author="lukexie",
            pipeline_description="XGBoost E2E Pipeline with MODS",
            pipeline_version="1.2.3"
        )
        
        # Use like any other pipeline
        pipeline_instance = MODSXGBoostPipeline(
            config_path="config.json",
            sagemaker_session=session,
            execution_role=role
        )
        pipeline = pipeline_instance.generate_pipeline()
        ```
    """
    if not MODS_AVAILABLE:
        logger.warning("MODS template not available, creating pipeline without MODS enhancement")
    
    # Apply MODSTemplate decorator
    @MODSTemplate(author=author, description=pipeline_description, version=pipeline_version)
    class MODSEnhancedPipeline(pipeline_class):
        """MODS-enhanced version of the pipeline class."""
        
        def __init__(self, *args, **kwargs):
            # Enable MODS by default for MODS-enhanced pipelines
            kwargs.setdefault('enable_mods', True)
            super().__init__(*args, **kwargs)
            
        @classmethod
        def get_mods_metadata(cls) -> Dict[str, str]:
            """Get MODS metadata for this pipeline."""
            return {
                'author': author,
                'description': pipeline_description,
                'version': pipeline_version
            }
    
    # Set a meaningful class name
    original_name = pipeline_class.__name__
    if original_name.endswith('Pipeline'):
        mods_name = original_name.replace('Pipeline', 'MODSPipeline')
    else:
        mods_name = f"MODS{original_name}"
    
    MODSEnhancedPipeline.__name__ = mods_name
    MODSEnhancedPipeline.__qualname__ = mods_name
    
    return MODSEnhancedPipeline


def create_mods_pipeline_from_config(
    pipeline_class: Type,
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    author: Optional[str] = None,
    pipeline_description: Optional[str] = None,
    pipeline_version: Optional[str] = None
) -> Type:
    """
    Create a MODS-enhanced pipeline class from a regular pipeline class,
    extracting MODS metadata from configuration.
    
    This function reads the configuration and extracts the required MODS fields
    (author, pipeline_description, pipeline_version) from the base configuration,
    following the pattern established in demo/demo_pipeline.ipynb where the base
    config is accessed using the 'Base' key.
    
    Args:
        pipeline_class: The regular pipeline class to enhance
        config_path: Path to configuration file (JSON)
        config_dict: Configuration dictionary (alternative to config_path)
        author: Override author from config
        pipeline_description: Override description from config
        pipeline_version: Override version from config
        
    Returns:
        Type: MODS-enhanced pipeline class
        
    Example:
        ```python
        from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import XGBoostE2EComprehensivePipeline
        
        # Extract MODS metadata from config file (with 'Base' key)
        MODSXGBoostPipeline = create_mods_pipeline_from_config(
            XGBoostE2EComprehensivePipeline,
            config_path="config.json"
        )
        
        # Or provide base config directly
        MODSXGBoostPipeline = create_mods_pipeline_from_config(
            XGBoostE2EComprehensivePipeline,
            config_dict={
                "Base": {
                    "author": "lukexie",
                    "service_name": "AtoZ",
                    "model_class": "xgboost",
                    "region": "NA",
                    "pipeline_version": "1.2.3"
                }
            }
        )
        ```
    """
    # Load configuration
    config_data = {}
    
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Configuration file {config_path} not found")
    
    if config_dict:
        config_data.update(config_dict)
    
    # Extract MODS metadata from config
    try:
        # Check if config has 'Base' key (following demo pattern)
        base_config_data = config_data.get('Base', config_data)
        
        # Create a temporary config instance to get derived fields
        temp_config = BasePipelineConfig(**base_config_data)
        
        # Extract fields, with overrides taking precedence
        final_author = author or temp_config.author
        final_description = pipeline_description or temp_config.pipeline_description
        final_version = pipeline_version or temp_config.pipeline_version
        
        logger.info(f"Extracted MODS metadata - Author: {final_author}, Version: {final_version}")
        
    except Exception as e:
        logger.error(f"Failed to extract MODS metadata from config: {e}")
        # Fallback to provided values or defaults
        final_author = author or "unknown"
        final_description = pipeline_description or "MODS-enhanced pipeline"
        final_version = pipeline_version or "1.0.0"
    
    return create_mods_pipeline(
        pipeline_class=pipeline_class,
        author=final_author,
        pipeline_description=final_description,
        pipeline_version=final_version
    )


def get_mods_pipeline_factory(pipeline_class: Type):
    """
    Create a factory function for generating MODS-enhanced pipelines.
    
    This is useful when you want to create multiple MODS variants of the same
    pipeline with different metadata.
    
    Args:
        pipeline_class: The regular pipeline class to enhance
        
    Returns:
        Callable: Factory function that creates MODS-enhanced pipelines
        
    Example:
        ```python
        from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import XGBoostE2EComprehensivePipeline
        
        # Create factory
        xgb_mods_factory = get_mods_pipeline_factory(XGBoostE2EComprehensivePipeline)
        
        # Create different MODS variants
        DevXGBoostPipeline = xgb_mods_factory(
            author="dev-team",
            pipeline_description="Development XGBoost Pipeline",
            pipeline_version="0.1.0"
        )
        
        ProdXGBoostPipeline = xgb_mods_factory(
            author="prod-team", 
            pipeline_description="Production XGBoost Pipeline",
            pipeline_version="2.0.0"
        )
        ```
    """
    def factory(author: str, pipeline_description: str, pipeline_version: str) -> Type:
        return create_mods_pipeline(
            pipeline_class=pipeline_class,
            author=author,
            pipeline_description=pipeline_description,
            pipeline_version=pipeline_version
        )
    
    return factory


# Convenience functions for common pipeline types

def create_mods_xgboost_e2e_comprehensive(
    author: str = None,
    pipeline_description: str = None,
    pipeline_version: str = None,
    config_path: str = None,
    config_dict: Dict[str, Any] = None
) -> Type:
    """Create MODS-enhanced XGBoost E2E Comprehensive Pipeline."""
    from .pipelines.xgb_e2e_comprehensive import XGBoostE2EComprehensivePipeline
    
    if config_path or config_dict:
        return create_mods_pipeline_from_config(
            XGBoostE2EComprehensivePipeline,
            config_path=config_path,
            config_dict=config_dict,
            author=author,
            pipeline_description=pipeline_description,
            pipeline_version=pipeline_version
        )
    else:
        return create_mods_pipeline(
            XGBoostE2EComprehensivePipeline,
            author=author,
            pipeline_description=pipeline_description,
            pipeline_version=pipeline_version
        )


def create_mods_pytorch_e2e_standard(
    author: str = None,
    pipeline_description: str = None,
    pipeline_version: str = None,
    config_path: str = None,
    config_dict: Dict[str, Any] = None
) -> Type:
    """Create MODS-enhanced PyTorch E2E Standard Pipeline."""
    from .pipelines.pytorch_e2e_standard import PyTorchE2EStandardPipeline
    
    if config_path or config_dict:
        return create_mods_pipeline_from_config(
            PyTorchE2EStandardPipeline,
            config_path=config_path,
            config_dict=config_dict,
            author=author,
            pipeline_description=pipeline_description,
            pipeline_version=pipeline_version
        )
    else:
        return create_mods_pipeline(
            PyTorchE2EStandardPipeline,
            author=author,
            pipeline_description=pipeline_description,
            pipeline_version=pipeline_version
        )


def create_mods_dummy_e2e_basic(
    author: str = None,
    pipeline_description: str = None,
    pipeline_version: str = None,
    config_path: str = None,
    config_dict: Dict[str, Any] = None
) -> Type:
    """Create MODS-enhanced Dummy E2E Basic Pipeline."""
    from .pipelines.dummy_e2e_basic import DummyE2EBasicPipeline
    
    if config_path or config_dict:
        return create_mods_pipeline_from_config(
            DummyE2EBasicPipeline,
            config_path=config_path,
            config_dict=config_dict,
            author=author,
            pipeline_description=pipeline_description,
            pipeline_version=pipeline_version
        )
    else:
        return create_mods_pipeline(
            DummyE2EBasicPipeline,
            author=author,
            pipeline_description=pipeline_description,
            pipeline_version=pipeline_version
        )


# Registry of available pipeline types for dynamic creation
PIPELINE_REGISTRY = {
    'xgb_e2e_comprehensive': 'pipelines.xgb_e2e_comprehensive.XGBoostE2EComprehensivePipeline',
    'pytorch_e2e_standard': 'pipelines.pytorch_e2e_standard.PyTorchE2EStandardPipeline',
    'pytorch_training_basic': 'pipelines.pytorch_training_basic.PyTorchTrainingBasicPipeline',
    'xgb_training_calibrated': 'pipelines.xgb_training_calibrated.XGBoostTrainingCalibratedPipeline',
    'xgb_training_evaluation': 'pipelines.xgb_training_evaluation.XGBoostTrainingEvaluationPipeline',
    'xgb_training_simple': 'pipelines.xgb_training_simple.XGBoostTrainingSimplePipeline',
    'dummy_e2e_basic': 'pipelines.dummy_e2e_basic.DummyE2EBasicPipeline',
}


def create_mods_pipeline_by_name(
    pipeline_name: str,
    author: str = None,
    pipeline_description: str = None,
    pipeline_version: str = None,
    config_path: str = None,
    config_dict: Dict[str, Any] = None
) -> Type:
    """
    Create a MODS-enhanced pipeline by name.
    
    Args:
        pipeline_name: Name of the pipeline (from PIPELINE_REGISTRY)
        author: Author or owner of the pipeline
        pipeline_description: Description of the pipeline
        pipeline_version: Version string for the pipeline
        config_path: Path to configuration file
        config_dict: Configuration dictionary
        
    Returns:
        Type: MODS-enhanced pipeline class
        
    Example:
        ```python
        # Create MODS XGBoost pipeline by name
        MODSXGBoostPipeline = create_mods_pipeline_by_name(
            'xgb_e2e_comprehensive',
            config_path='config.json'
        )
        ```
    """
    if pipeline_name not in PIPELINE_REGISTRY:
        available = ', '.join(PIPELINE_REGISTRY.keys())
        raise ValueError(f"Unknown pipeline name: {pipeline_name}. Available: {available}")
    
    # Import the pipeline class dynamically
    module_path = PIPELINE_REGISTRY[pipeline_name]
    module_name, class_name = module_path.rsplit('.', 1)
    
    # Import from the pipeline_catalog package
    full_module_name = f"cursus.pipeline_catalog.{module_name}"
    module = __import__(full_module_name, fromlist=[class_name])
    pipeline_class = getattr(module, class_name)
    
    # Create MODS-enhanced version
    if config_path or config_dict:
        return create_mods_pipeline_from_config(
            pipeline_class,
            config_path=config_path,
            config_dict=config_dict,
            author=author,
            pipeline_description=pipeline_description,
            pipeline_version=pipeline_version
        )
    else:
        return create_mods_pipeline(
            pipeline_class,
            author=author,
            pipeline_description=pipeline_description,
            pipeline_version=pipeline_version
        )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Create MODS-enhanced pipelines")
    parser.add_argument("--pipeline", required=True, help="Pipeline name")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--author", help="Pipeline author")
    parser.add_argument("--description", help="Pipeline description")
    parser.add_argument("--version", help="Pipeline version")
    
    args = parser.parse_args()
    
    try:
        # Create MODS pipeline
        MODSPipeline = create_mods_pipeline_by_name(
            pipeline_name=args.pipeline,
            config_path=args.config,
            author=args.author,
            pipeline_description=args.description,
            pipeline_version=args.version
        )
        
        print(f"Created MODS-enhanced pipeline: {MODSPipeline.__name__}")
        print(f"MODS metadata: {MODSPipeline.get_mods_metadata()}")
        
    except Exception as e:
        logger.error(f"Failed to create MODS pipeline: {e}")
        raise
