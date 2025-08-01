"""
Example usage of the Pipeline API for converting DAGs to SageMaker pipelines.

This example demonstrates both simple and advanced usage patterns.
"""

import logging
from pathlib import Path

# Configure logging to see API operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_example():
    """
    Simple example: One-call conversion from DAG to pipeline.
    """
    print("=" * 60)
    print("SIMPLE EXAMPLE: One-call DAG to Pipeline conversion")
    print("=" * 60)
    
    from src.pipeline_dag.base_dag import PipelineDAG
    from src.pipeline_api import dag_to_pipeline_template
    
    # Create a simple DAG
    dag = PipelineDAG()
    dag.add_node("data_load")
    dag.add_node("preprocess")
    dag.add_node("train")
    dag.add_node("evaluate")
    
    # Add dependencies
    dag.add_edge("data_load", "preprocess")
    dag.add_edge("preprocess", "train")
    dag.add_edge("train", "evaluate")
    
    print(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    print(f"Nodes: {list(dag.nodes)}")
    print(f"Execution order: {dag.topological_sort()}")
    
    # Note: This would require actual config file and SageMaker session
    # For demonstration, we'll show the API call structure
    
    config_path = "pipeline_config/example_config.json"
    
    try:
        # This is how you would convert in practice:
        # pipeline = dag_to_pipeline_template(
        #     dag=dag,
        #     config_path=config_path,
        #     sagemaker_session=session,
        #     role="arn:aws:iam::123456789012:role/SageMakerRole",
        #     pipeline_name="example-pipeline"
        # )
        # 
        # pipeline.upsert()
        # execution = pipeline.start()
        
        print(f"✓ Would convert DAG using config: {config_path}")
        print("✓ Pipeline would be created and ready for deployment")
        
    except Exception as e:
        print(f"✗ Conversion would fail: {e}")
    
    print()


def advanced_example():
    """
    Advanced example: Full control with validation and reporting.
    """
    print("=" * 60)
    print("ADVANCED EXAMPLE: Full control with validation")
    print("=" * 60)
    
    from src.pipeline_dag.base_dag import PipelineDAG
    from src.pipeline_api import PipelineDAGConverter
    
    # Create a more complex DAG
    dag = PipelineDAG()
    
    # Data pipeline
    dag.add_node("cradle_data_load")
    dag.add_node("tabular_preprocess")
    
    # Training pipeline
    dag.add_node("xgb_training")
    dag.add_node("model_evaluation")
    dag.add_node("model_calibration")
    
    # Deployment pipeline
    dag.add_node("mims_packaging")
    dag.add_node("model_registration")
    
    # Add dependencies
    dag.add_edge("cradle_data_load", "tabular_preprocess")
    dag.add_edge("tabular_preprocess", "xgb_training")
    dag.add_edge("xgb_training", "model_evaluation")
    dag.add_edge("model_evaluation", "model_calibration")
    dag.add_edge("model_calibration", "mims_packaging")
    dag.add_edge("mims_packaging", "model_registration")
    
    print(f"Created complex DAG with {len(dag.nodes)} nodes")
    print(f"Nodes: {list(dag.nodes)}")
    
    # Create converter for advanced control
    config_path = "pipeline_config/advanced_config.json"
    
    try:
        converter = PipelineDAGConverter(
            config_path=config_path,
            # sagemaker_session=session,
            # role="arn:aws:iam::123456789012:role/SageMakerRole"
        )
        
        print(f"✓ Created converter with config: {config_path}")
        
        # Step 1: Validate DAG compatibility
        print("\n1. Validating DAG compatibility...")
        # validation_result = converter.validate_dag_compatibility(dag)
        # 
        # if validation_result.is_valid:
        #     print("✓ Validation passed!")
        #     print(validation_result.summary())
        # else:
        #     print("✗ Validation failed!")
        #     print(validation_result.detailed_report())
        #     return
        
        print("✓ Would validate DAG compatibility")
        
        # Step 2: Preview resolution
        print("\n2. Previewing node resolution...")
        # preview = converter.preview_resolution(dag)
        # print(preview.display())
        
        print("✓ Would show resolution preview")
        
        # Step 3: Convert with detailed reporting
        print("\n3. Converting DAG to pipeline...")
        # pipeline, report = converter.convert_with_report(
        #     dag=dag,
        #     pipeline_name="advanced-example-pipeline"
        # )
        # 
        # print("✓ Conversion successful!")
        # print(report.summary())
        # print("\nDetailed Report:")
        # print(report.detailed_report())
        
        print("✓ Would convert with detailed reporting")
        
        # Step 4: Get additional information
        print("\n4. Additional information...")
        supported_types = converter.get_supported_step_types()
        print(f"✓ Supported step types: {len(supported_types)}")
        print(f"  Examples: {supported_types[:5]}...")
        
        # config_validation = converter.validate_config_file()
        # print(f"✓ Config file validation: {config_validation}")
        
    except FileNotFoundError as e:
        print(f"✗ Config file not found: {e}")
        print("  This is expected in the example - you would need a real config file")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def error_handling_example():
    """
    Example of comprehensive error handling.
    """
    print("=" * 60)
    print("ERROR HANDLING EXAMPLE")
    print("=" * 60)
    
    from src.pipeline_dag.base_dag import PipelineDAG
    from src.pipeline_api import (
        dag_to_pipeline_template,
        ConfigurationError,
        RegistryError,
        ValidationError,
        PipelineAPIError
    )
    
    # Create DAG with potentially problematic nodes
    dag = PipelineDAG()
    dag.add_node("unknown_step_type")
    dag.add_node("ambiguous_name")
    dag.add_node("missing_config")
    
    config_path = "nonexistent_config.json"
    
    try:
        pipeline = dag_to_pipeline_template(
            dag=dag,
            config_path=config_path
        )
        print("✓ Conversion successful (unexpected)")
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        print("  → Check that config file exists and path is correct")
        
    except ConfigurationError as e:
        print(f"✗ Configuration error: {e}")
        print(f"  → Missing configs: {e.missing_configs}")
        print(f"  → Available configs: {e.available_configs}")
        print("  → Add missing configurations or rename DAG nodes")
        
    except RegistryError as e:
        print(f"✗ Registry error: {e}")
        print(f"  → Unresolvable types: {e.unresolvable_types}")
        print(f"  → Available builders: {e.available_builders}")
        print("  → Register custom builders or use supported config types")
        
    except ValidationError as e:
        print(f"✗ Validation error: {e}")
        for category, errors in e.validation_errors.items():
            print(f"  → {category}: {errors}")
        print("  → Fix validation issues before conversion")
        
    except PipelineAPIError as e:
        print(f"✗ General API error: {e}")
        print("  → Check logs for detailed error information")
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        print("  → This might indicate a bug - please report")
    
    print()


def customization_example():
    """
    Example of API customization and extension.
    """
    print("=" * 60)
    print("CUSTOMIZATION EXAMPLE")
    print("=" * 60)
    
    from src.pipeline_api import (
        PipelineDAGConverter,
        StepConfigResolver,
        register_global_builder,
        list_global_step_types
    )
    
    # Custom config resolver
    class CustomConfigResolver(StepConfigResolver):
        """Custom resolver with additional matching logic."""
        
        def _semantic_matching(self, node_name, configs):
            """Enhanced semantic matching."""
            matches = super()._semantic_matching(node_name, configs)
            
            # Add custom semantic rules
            if "etl" in node_name.lower():
                # ETL nodes should match preprocessing configs
                for config_name, config in configs.items():
                    if "preprocess" in type(config).__name__.lower():
                        matches.append((config, 0.8, 'custom_etl'))
            
            return matches
    
    # Custom step builder (placeholder)
    # class CustomStepBuilder(StepBuilderBase):
    #     """Custom step builder for special use cases."""
    #     pass
    
    print("✓ Defined custom config resolver")
    # print("✓ Defined custom step builder")
    
    # Register custom builder
    # register_global_builder("CustomStep", CustomStepBuilder)
    # print("✓ Registered custom step builder")
    
    # List supported types
    supported_types = list_global_step_types()
    print(f"✓ Available step types: {len(supported_types)}")
    
    # Create converter with custom resolver
    try:
        converter = PipelineDAGConverter(
            config_path="config.json",
            config_resolver=CustomConfigResolver(confidence_threshold=0.6)
        )
        print("✓ Created converter with custom resolver")
        
    except FileNotFoundError:
        print("✓ Would create converter with custom resolver")
    
    print()


def main():
    """
    Run all examples to demonstrate the Pipeline API.
    """
    print("Pipeline API Examples")
    print("=" * 60)
    print("This demonstrates the DAG-to-Template conversion API")
    print("Note: Examples show API structure - actual execution requires")
    print("      valid config files and SageMaker session")
    print()
    
    # Run examples
    simple_example()
    advanced_example()
    error_handling_example()
    customization_example()
    
    print("=" * 60)
    print("Examples completed!")
    print("See slipbox/pipeline_api/README.md for detailed documentation")
    print("=" * 60)


if __name__ == "__main__":
    main()
