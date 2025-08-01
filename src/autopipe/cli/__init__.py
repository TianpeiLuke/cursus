"""
AutoPipe Command Line Interface

Provides command-line tools for compiling DAGs to SageMaker pipelines,
validating pipeline configurations, and managing AutoPipe workflows.
"""

import sys
import json
import click
from pathlib import Path
from typing import Optional

from ..__version__ import __version__


@click.group()
@click.version_option(version=__version__, prog_name="autopipe")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def main(ctx, verbose):
    """
    AutoPipe: Automatic SageMaker Pipeline Generation
    
    Transform pipeline graphs into production-ready SageMaker pipelines automatically.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        click.echo(f"AutoPipe v{__version__}")


@main.command()
@click.argument('dag_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), 
              help='Output file path for the compiled pipeline')
@click.option('--name', '-n', help='Pipeline name')
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path),
              help='Configuration file path')
@click.option('--format', type=click.Choice(['json', 'yaml']), default='json',
              help='Output format')
@click.pass_context
def compile(ctx, dag_file, output, name, config, format):
    """
    Compile a DAG file to SageMaker pipeline.
    
    Takes a Python file containing a DAG definition and compiles it into
    a SageMaker pipeline that can be executed on AWS.
    
    Example:
        autopipe compile my_dag.py --name fraud-detection --output pipeline.json
    """
    try:
        from ..api.dag_compiler import compile_dag_to_pipeline
        from ..core.dag import PipelineDAG
        
        if ctx.obj['verbose']:
            click.echo(f"Compiling DAG from: {dag_file}")
            if config:
                click.echo(f"Using config: {config}")
        
        # Load the DAG from the Python file
        # This is a simplified implementation - in practice, you'd want more robust loading
        dag_module = _load_dag_from_file(dag_file)
        
        # Compile the DAG
        pipeline = compile_dag_to_pipeline(
            dag_module.dag if hasattr(dag_module, 'dag') else dag_module,
            pipeline_name=name,
            config_path=str(config) if config else None
        )
        
        # Output the result
        if output:
            with open(output, 'w') as f:
                if format == 'json':
                    json.dump(pipeline.definition(), f, indent=2)
                else:
                    # YAML output would require PyYAML
                    import yaml
                    yaml.dump(pipeline.definition(), f, default_flow_style=False)
            
            click.echo(f"Pipeline compiled successfully to: {output}")
        else:
            # Print to stdout
            if format == 'json':
                click.echo(json.dumps(pipeline.definition(), indent=2))
            else:
                import yaml
                click.echo(yaml.dump(pipeline.definition(), default_flow_style=False))
                
    except ImportError as e:
        click.echo(f"Error: Missing dependencies. Please install with: pip install autopipe[all]", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error compiling DAG: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('dag_file', type=click.Path(exists=True, path_type=Path))
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path),
              help='Configuration file path')
@click.pass_context
def validate(ctx, dag_file, config):
    """
    Validate a DAG file for compilation compatibility.
    
    Checks the DAG structure, dependencies, and configuration to ensure
    it can be successfully compiled to a SageMaker pipeline.
    
    Example:
        autopipe validate my_dag.py --config config.yaml
    """
    try:
        from ..api.validation import ValidationEngine
        
        if ctx.obj['verbose']:
            click.echo(f"Validating DAG: {dag_file}")
        
        # Load and validate the DAG
        dag_module = _load_dag_from_file(dag_file)
        validator = ValidationEngine()
        
        result = validator.validate_dag_compatibility(
            dag_module.dag if hasattr(dag_module, 'dag') else dag_module,
            config_path=str(config) if config else None
        )
        
        if result.is_valid:
            click.echo("✅ DAG validation passed!")
            if ctx.obj['verbose']:
                click.echo(result.summary())
        else:
            click.echo("❌ DAG validation failed!")
            click.echo(result.detailed_report())
            sys.exit(1)
            
    except ImportError as e:
        click.echo(f"Error: Missing dependencies. Please install with: pip install autopipe[all]", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error validating DAG: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('dag_file', type=click.Path(exists=True, path_type=Path))
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path),
              help='Configuration file path')
@click.pass_context
def preview(ctx, dag_file, config):
    """
    Preview the compilation results without generating the full pipeline.
    
    Shows what steps will be created, their dependencies, and configuration
    resolution without actually compiling the complete pipeline.
    
    Example:
        autopipe preview my_dag.py
    """
    try:
        from ..api.dag_compiler import PipelineDAGCompiler
        
        if ctx.obj['verbose']:
            click.echo(f"Previewing DAG compilation: {dag_file}")
        
        # Load the DAG
        dag_module = _load_dag_from_file(dag_file)
        
        # Create compiler and get preview
        compiler = PipelineDAGCompiler(
            config_path=str(config) if config else None
        )
        
        preview = compiler.preview_resolution(
            dag_module.dag if hasattr(dag_module, 'dag') else dag_module
        )
        
        click.echo("📋 Compilation Preview:")
        click.echo(preview.display())
        
    except ImportError as e:
        click.echo(f"Error: Missing dependencies. Please install with: pip install autopipe[all]", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error previewing DAG: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command('list-steps')
@click.pass_context
def list_steps(ctx):
    """
    List all available step types that can be used in DAGs.
    
    Shows the supported step types and their basic information.
    """
    try:
        from ..api.dag_compiler import PipelineDAGCompiler
        
        compiler = PipelineDAGCompiler()
        step_types = compiler.get_supported_step_types()
        
        click.echo("📚 Available Step Types:")
        for step_type in sorted(step_types):
            click.echo(f"  • {step_type}")
            
    except ImportError as e:
        click.echo(f"Error: Missing dependencies. Please install with: pip install autopipe[all]", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error listing step types: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--template', type=click.Choice(['xgboost', 'pytorch', 'basic']), 
              default='basic', help='Template type to generate')
@click.option('--name', '-n', required=True, help='Project name')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), 
              default=Path('.'), help='Output directory')
@click.pass_context
def init(ctx, template, name, output_dir):
    """
    Generate an example DAG project from a template.
    
    Creates a new project directory with example DAG files and configuration
    based on the selected template.
    
    Example:
        autopipe init --template xgboost --name fraud-detection
    """
    try:
        project_dir = output_dir / name
        project_dir.mkdir(exist_ok=True)
        
        if ctx.obj['verbose']:
            click.echo(f"Creating {template} project in: {project_dir}")
        
        # Create basic project structure
        (project_dir / "dags").mkdir(exist_ok=True)
        (project_dir / "config").mkdir(exist_ok=True)
        
        # Generate template files based on selection
        _generate_template_files(project_dir, template, name)
        
        click.echo(f"✅ Created {template} project: {project_dir}")
        click.echo(f"Next steps:")
        click.echo(f"  cd {name}")
        click.echo(f"  autopipe validate dags/main.py")
        click.echo(f"  autopipe compile dags/main.py --name {name}")
        
    except Exception as e:
        click.echo(f"Error creating project: {e}", err=True)
        sys.exit(1)


def _load_dag_from_file(dag_file: Path):
    """Load a DAG from a Python file."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("dag_module", dag_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load DAG from {dag_file}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module


def _generate_template_files(project_dir: Path, template: str, name: str):
    """Generate template files for a new project."""
    
    # Basic DAG template
    dag_content = f'''"""
Example DAG for {name} project.
"""

from autopipe.core.dag import PipelineDAG
from autopipe.steps.specs import *

# Create the DAG
dag = PipelineDAG(name="{name}")

# Add steps based on template
'''
    
    if template == 'xgboost':
        dag_content += '''
# Data loading step
dag.add_node("data_loading", CRADLE_DATA_LOADING_SPEC)

# Preprocessing step  
dag.add_node("preprocessing", TABULAR_PREPROCESSING_SPEC)

# XGBoost training step
dag.add_node("training", XGBOOST_TRAINING_SPEC)

# Model evaluation step
dag.add_node("evaluation", MODEL_EVALUATION_SPEC)

# Add dependencies
dag.add_edge("data_loading", "preprocessing")
dag.add_edge("preprocessing", "training") 
dag.add_edge("training", "evaluation")
'''
    elif template == 'pytorch':
        dag_content += '''
# Data loading step
dag.add_node("data_loading", CRADLE_DATA_LOADING_SPEC)

# Preprocessing step
dag.add_node("preprocessing", TEXT_PREPROCESSING_SPEC)

# PyTorch training step
dag.add_node("training", PYTORCH_TRAINING_SPEC)

# Model evaluation step
dag.add_node("evaluation", MODEL_EVALUATION_SPEC)

# Add dependencies
dag.add_edge("data_loading", "preprocessing")
dag.add_edge("preprocessing", "training")
dag.add_edge("training", "evaluation")
'''
    else:  # basic
        dag_content += '''
# Simple processing step
dag.add_node("processing", BASIC_PROCESSING_SPEC)

# Training step
dag.add_node("training", BASIC_TRAINING_SPEC)

# Add dependency
dag.add_edge("processing", "training")
'''
    
    # Write DAG file
    with open(project_dir / "dags" / "main.py", 'w') as f:
        f.write(dag_content)
    
    # Write basic config
    config_content = f'''# Configuration for {name} project
pipeline_name: "{name}"
region: "us-east-1"
role_arn: "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole"

# Step configurations
steps:
  data_loading:
    input_path: "s3://your-bucket/data/"
  
  training:
    instance_type: "ml.m5.large"
    max_runtime_in_seconds: 3600
'''
    
    with open(project_dir / "config" / "config.yaml", 'w') as f:
        f.write(config_content)
    
    # Write README
    readme_content = f'''# {name}

AutoPipe project generated from {template} template.

## Usage

1. Validate the DAG:
   ```bash
   autopipe validate dags/main.py --config config/config.yaml
   ```

2. Preview compilation:
   ```bash
   autopipe preview dags/main.py --config config/config.yaml
   ```

3. Compile to SageMaker pipeline:
   ```bash
   autopipe compile dags/main.py --config config/config.yaml --name {name}
   ```

## Configuration

Edit `config/config.yaml` to customize:
- AWS region and IAM role
- Step-specific configurations
- Input/output paths
- Instance types and resources

## DAG Structure

The DAG is defined in `dags/main.py`. Modify this file to:
- Add new steps
- Change step dependencies
- Customize step specifications
'''
    
    with open(project_dir / "README.md", 'w') as f:
        f.write(readme_content)


if __name__ == '__main__':
    main()
