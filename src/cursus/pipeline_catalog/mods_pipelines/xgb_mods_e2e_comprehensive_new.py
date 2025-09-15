"""
MODS-Enhanced XGBoost E2E Comprehensive Pipeline

This file demonstrates the new approach using the MODS API to eliminate duplication.
Instead of duplicating the entire pipeline class, we simply import the regular pipeline
and apply the MODS decorator using the API.

This approach:
1. Eliminates code duplication between pipelines/ and mods_pipelines/ folders
2. Extracts MODS metadata from configuration automatically
3. Maintains the same interface and functionality
4. Reduces maintenance burden significantly

Example:
    ```python
    from cursus.pipeline_catalog.mods_pipelines.xgb_mods_e2e_comprehensive_new import XGBoostE2EComprehensiveMODSPipeline
    from sagemaker import Session
    from sagemaker.workflow.pipeline_context import PipelineSession
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Create MODS pipeline instance - same interface as before
    pipeline_instance = XGBoostE2EComprehensiveMODSPipeline(
        config_path="path/to/config.json",
        sagemaker_session=pipeline_session,
        execution_role=role
    )
    
    # Generate pipeline - MODS-enhanced automatically
    pipeline = pipeline_instance.generate_pipeline()
    
    # Execute the pipeline
    pipeline.upsert()
    execution = pipeline.start()
    ```
"""

import logging
from typing import Optional, Dict, Any

from sagemaker.workflow.pipeline_context import PipelineSession

# Import the MODS API
from ..mods_api import create_mods_pipeline_from_config

# Import the regular pipeline class
from ..pipelines.xgb_e2e_comprehensive import XGBoostE2EComprehensivePipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_xgboost_e2e_comprehensive_mods_pipeline(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    author: Optional[str] = None,
    pipeline_description: Optional[str] = None,
    pipeline_version: Optional[str] = None
):
    """
    Create a MODS-enhanced XGBoost E2E Comprehensive Pipeline.
    
    This function uses the MODS API to create a MODS-enhanced version of the
    regular XGBoost E2E Comprehensive Pipeline, extracting metadata from config.
    
    Args:
        config_path: Path to configuration file (JSON)
        config_dict: Configuration dictionary (alternative to config_path)
        author: Override author from config
        pipeline_description: Override description from config
        pipeline_version: Override version from config
        
    Returns:
        Type: MODS-enhanced pipeline class
        
    Example:
        ```python
        # Create from config file
        MODSPipeline = create_xgboost_e2e_comprehensive_mods_pipeline(
            config_path="config.json"
        )
        
        # Create from config dict
        MODSPipeline = create_xgboost_e2e_comprehensive_mods_pipeline(
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
        
        # Use the pipeline
        pipeline_instance = MODSPipeline(
            config_path="config.json",
            sagemaker_session=session,
            execution_role=role
        )
        pipeline = pipeline_instance.generate_pipeline()
        ```
    """
    return create_mods_pipeline_from_config(
        pipeline_class=XGBoostE2EComprehensivePipeline,
        config_path=config_path,
        config_dict=config_dict,
        author=author,
        pipeline_description=pipeline_description,
        pipeline_version=pipeline_version
    )


# Create the MODS-enhanced pipeline class for direct import
# This maintains backward compatibility with existing code
# Note: The MODS metadata will be extracted at runtime from the config
XGBoostE2EComprehensiveMODSPipeline = create_mods_pipeline_from_config(
    pipeline_class=XGBoostE2EComprehensivePipeline
)


if __name__ == "__main__":
    # Example usage
    import argparse
    from sagemaker import Session

    parser = argparse.ArgumentParser(description="Create MODS-enhanced XGBoost E2E Comprehensive Pipeline")
    parser.add_argument("--config-path", type=str, help="Path to the configuration file")
    parser.add_argument("--author", type=str, help="Pipeline author")
    parser.add_argument("--description", type=str, help="Pipeline description")
    parser.add_argument("--version", type=str, help="Pipeline version")
    
    args = parser.parse_args()
    
    try:
        # Create MODS pipeline
        MODSPipeline = create_xgboost_e2e_comprehensive_mods_pipeline(
            config_path=args.config_path,
            author=args.author,
            pipeline_description=args.description,
            pipeline_version=args.version
        )
        
        print(f"Created MODS-enhanced pipeline: {MODSPipeline.__name__}")
        print(f"MODS metadata: {MODSPipeline.get_mods_metadata()}")
        
        # Example of using the pipeline
        if args.config_path:
            sagemaker_session = Session()
            role = sagemaker_session.get_caller_identity_arn()
            
            pipeline_instance = MODSPipeline(
                config_path=args.config_path,
                sagemaker_session=sagemaker_session,
                execution_role=role
            )
            
            print(f"Pipeline instance created successfully")
            
    except Exception as e:
        logger.error(f"Failed to create MODS pipeline: {e}")
        raise
