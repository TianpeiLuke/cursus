"""
Workspace Templates - Phase 4 High-Level API Creation

This module provides workspace template management functionality,
allowing users to create standardized workspace structures.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import json
import yaml
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class TemplateType(Enum):
    """Available workspace template types."""
    BASIC = "basic"
    ML_PIPELINE = "ml_pipeline"
    DATA_PROCESSING = "data_processing"
    CUSTOM = "custom"


class WorkspaceTemplate(BaseModel):
    """Workspace template configuration."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )
    
    name: str = Field(..., min_length=1, description="Template name")
    type: TemplateType
    description: str = Field("", description="Template description")
    directories: List[str] = Field(default_factory=list, description="Directories to create")
    files: Dict[str, str] = Field(default_factory=dict, description="Files to create with content")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
    config_overrides: Dict[str, Any] = Field(default_factory=dict, description="Default configuration")
    created_at: Optional[str] = Field(None, description="Template creation timestamp")
    version: str = Field("1.0.0", description="Template version")


class TemplateManager:
    """Manages workspace templates."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize template manager.
        
        Args:
            templates_dir: Directory containing template definitions
        """
        self.templates_dir = templates_dir or Path(__file__).parent / "templates"
        self.templates_dir.mkdir(exist_ok=True)
        
        # Initialize built-in templates
        self._initialize_builtin_templates()
    
    def _initialize_builtin_templates(self):
        """Initialize built-in workspace templates."""
        
        # Basic template
        basic_template = WorkspaceTemplate(
            name="basic",
            type=TemplateType.BASIC,
            description="Basic workspace with standard directory structure",
            directories=[
                "builders",
                "configs", 
                "contracts",
                "specs",
                "scripts",
                "tests"
            ],
            files={
                "README.md": self._get_basic_readme(),
                ".gitignore": self._get_basic_gitignore(),
                "builders/__init__.py": "",
                "configs/__init__.py": "",
                "contracts/__init__.py": "",
                "specs/__init__.py": "",
                "scripts/__init__.py": "",
                "tests/__init__.py": "",
                "tests/test_example.py": self._get_basic_test_example()
            },
            created_at=datetime.now().isoformat()
        )
        
        # ML Pipeline template
        ml_template = WorkspaceTemplate(
            name="ml_pipeline",
            type=TemplateType.ML_PIPELINE,
            description="Machine learning pipeline workspace with ML-specific structure",
            directories=[
                "builders",
                "configs",
                "contracts", 
                "specs",
                "scripts",
                "data",
                "data/raw",
                "data/processed",
                "data/features",
                "models",
                "notebooks",
                "tests"
            ],
            files={
                "README.md": self._get_ml_readme(),
                ".gitignore": self._get_ml_gitignore(),
                "builders/__init__.py": "",
                "configs/__init__.py": "",
                "contracts/__init__.py": "",
                "specs/__init__.py": "",
                "scripts/__init__.py": "",
                "tests/__init__.py": "",
                "notebooks/exploration.ipynb": self._get_ml_notebook(),
                "data/README.md": self._get_data_readme(),
                "models/README.md": self._get_models_readme()
            },
            dependencies=[
                "pandas",
                "numpy", 
                "scikit-learn",
                "jupyter"
            ],
            created_at=datetime.now().isoformat()
        )
        
        # Data Processing template
        data_template = WorkspaceTemplate(
            name="data_processing",
            type=TemplateType.DATA_PROCESSING,
            description="Data processing workspace for ETL and data transformation pipelines",
            directories=[
                "builders",
                "configs",
                "contracts",
                "specs", 
                "scripts",
                "data",
                "data/raw",
                "data/processed",
                "data/output",
                "schemas",
                "tests"
            ],
            files={
                "README.md": self._get_data_processing_readme(),
                ".gitignore": self._get_basic_gitignore(),
                "builders/__init__.py": "",
                "configs/__init__.py": "",
                "contracts/__init__.py": "",
                "specs/__init__.py": "",
                "scripts/__init__.py": "",
                "tests/__init__.py": "",
                "schemas/README.md": self._get_schemas_readme(),
                "data/README.md": self._get_data_readme()
            },
            dependencies=[
                "pandas",
                "pydantic",
                "jsonschema"
            ],
            created_at=datetime.now().isoformat()
        )
        
        # Save built-in templates
        self._save_template(basic_template)
        self._save_template(ml_template)
        self._save_template(data_template)
    
    def get_template(self, name: str) -> Optional[WorkspaceTemplate]:
        """
        Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            WorkspaceTemplate if found, None otherwise
        """
        template_file = self.templates_dir / f"{name}.yaml"
        if not template_file.exists():
            return None
        
        try:
            with open(template_file, 'r') as f:
                template_data = yaml.safe_load(f)
            return WorkspaceTemplate(**template_data)
        except Exception:
            return None
    
    def list_templates(self) -> List[WorkspaceTemplate]:
        """
        List all available templates.
        
        Returns:
            List of available templates
        """
        templates = []
        
        for template_file in self.templates_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r') as f:
                    template_data = yaml.safe_load(f)
                templates.append(WorkspaceTemplate(**template_data))
            except Exception:
                continue
        
        return templates
    
    def create_template(self, template: WorkspaceTemplate) -> bool:
        """
        Create a new template.
        
        Args:
            template: Template to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            template.created_at = datetime.now().isoformat()
            self._save_template(template)
            return True
        except Exception:
            return False
    
    def apply_template(self, template_name: str, workspace_path: Path) -> bool:
        """
        Apply a template to a workspace.
        
        Args:
            template_name: Name of template to apply
            workspace_path: Path to workspace directory
            
        Returns:
            True if successful, False otherwise
        """
        template = self.get_template(template_name)
        if not template:
            return False
        
        try:
            # Create directories
            for directory in template.directories:
                dir_path = workspace_path / directory
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create files
            for file_path, content in template.files.items():
                full_path = workspace_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(full_path, 'w') as f:
                    f.write(content)
            
            return True
            
        except Exception:
            return False
    
    def _save_template(self, template: WorkspaceTemplate):
        """Save template to disk."""
        template_file = self.templates_dir / f"{template.name}.yaml"
        
        with open(template_file, 'w') as f:
            yaml.dump(template.model_dump(), f, default_flow_style=False)
    
    def _get_basic_readme(self) -> str:
        """Get basic README content."""
        return """# Developer Workspace

This is a basic developer workspace for the Cursus pipeline system.

## Directory Structure

- `builders/` - Step builder implementations
- `configs/` - Configuration classes
- `contracts/` - Step contracts
- `specs/` - Step specifications
- `scripts/` - Pipeline scripts
- `tests/` - Test files

## Getting Started

1. Implement your pipeline components in the appropriate directories
2. Use `cursus workspace validate --workspace-path .` to check your workspace
3. Use `cursus workspace discover components` to see your components
4. Use `cursus runtime test-script` to test your scripts

## Workspace Isolation

Remember: Everything in this workspace stays in this workspace.
Only shared code in `src/cursus/` is available to all workspaces.

## Testing

Run tests with:
```bash
python -m pytest tests/
```

## Development Workflow

1. Create your components following the established patterns
2. Write tests for your components
3. Validate workspace isolation
4. Test your pipeline scripts
5. Promote to staging when ready
"""
    
    def _get_basic_gitignore(self) -> str:
        """Get basic .gitignore content."""
        return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Data files (add specific patterns as needed)
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Model files
models/*.pkl
models/*.joblib
models/*.h5

# Jupyter
.ipynb_checkpoints/

# Testing
.pytest_cache/
.coverage
htmlcov/
"""
    
    def _get_basic_test_example(self) -> str:
        """Get basic test example."""
        return '''"""Example test file for workspace components."""

import pytest
from pathlib import Path


def test_workspace_structure():
    """Test that workspace has expected structure."""
    workspace_root = Path(__file__).parent.parent
    
    expected_dirs = [
        "builders",
        "configs", 
        "contracts",
        "specs",
        "scripts"
    ]
    
    for dir_name in expected_dirs:
        assert (workspace_root / dir_name).exists(), f"Missing directory: {dir_name}"
        assert (workspace_root / dir_name / "__init__.py").exists(), f"Missing __init__.py in {dir_name}"


def test_example_component():
    """Example test for a component."""
    # Add your component tests here
    assert True  # Placeholder
'''
    
    def _get_ml_readme(self) -> str:
        """Get ML pipeline README content."""
        return """# ML Pipeline Workspace

This workspace is configured for machine learning pipeline development.

## Directory Structure

- `builders/` - Step builder implementations
- `configs/` - Configuration classes
- `contracts/` - Step contracts
- `specs/` - Step specifications
- `scripts/` - Pipeline scripts
- `data/` - Data files (raw, processed, features)
- `models/` - Model artifacts and metadata
- `notebooks/` - Jupyter notebooks for experimentation
- `tests/` - Test files

## ML Pipeline Components

Common ML pipeline steps you might implement:
- Data preprocessing and cleaning
- Feature engineering and selection
- Model training and hyperparameter tuning
- Model evaluation and validation
- Model registration and deployment

## Getting Started

1. Start with data exploration in `notebooks/`
2. Implement data preprocessing components
3. Create feature engineering steps
4. Develop training and evaluation scripts
5. Test your pipeline with `cursus runtime test-pipeline`

## Data Management

- Place raw data in `data/raw/`
- Store processed data in `data/processed/`
- Save engineered features in `data/features/`
- Keep data files out of version control (see .gitignore)

## Model Management

- Save trained models in `models/`
- Include model metadata and performance metrics
- Use consistent naming conventions
- Document model versions and experiments

## Development Workflow

1. Explore data in notebooks
2. Implement reusable components
3. Create comprehensive tests
4. Validate model performance
5. Document experiments and results
"""
    
    def _get_ml_gitignore(self) -> str:
        """Get ML-specific .gitignore content."""
        return self._get_basic_gitignore() + """
# ML specific
*.pkl
*.joblib
*.h5
*.onnx
*.pb
checkpoints/
tensorboard_logs/
mlruns/
wandb/

# Data files
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
data/features/*
!data/features/.gitkeep

# Model artifacts
models/*.pkl
models/*.joblib
models/*.h5
models/*.onnx
models/checkpoints/
"""
    
    def _get_ml_notebook(self) -> str:
        """Get ML exploration notebook content."""
        return """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Exploration and Analysis\\n",
    "\\n",
    "This notebook is for initial data exploration and analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "# Set up plotting\\n",
    "plt.style.use('default')\\n",
    "sns.set_palette('husl')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load your data here\\n",
    "# df = pd.read_csv('../data/raw/your_data.csv')\\n",
    "# df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add your EDA code here"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
    
    def _get_data_processing_readme(self) -> str:
        """Get data processing README content."""
        return """# Data Processing Workspace

This workspace is configured for data processing pipeline development.

## Directory Structure

- `builders/` - Step builder implementations
- `configs/` - Configuration classes
- `contracts/` - Step contracts
- `specs/` - Step specifications
- `scripts/` - Pipeline scripts
- `data/` - Data files (raw, processed, output)
- `schemas/` - Data schemas and validation rules
- `tests/` - Test files

## Data Processing Components

Common data processing steps you might implement:
- Data ingestion from various sources
- Data cleaning and validation
- Data transformation and enrichment
- Data aggregation and summarization
- Data export to target systems

## Getting Started

1. Define your data schemas in `schemas/`
2. Place raw data in `data/raw/`
3. Implement data processing scripts
4. Create appropriate contracts and specs
5. Test your pipeline components

## Data Flow

1. **Ingestion**: Raw data → `data/raw/`
2. **Processing**: Raw data → `data/processed/`
3. **Output**: Processed data → `data/output/`

## Schema Management

- Define data schemas in `schemas/`
- Use schemas for validation and documentation
- Version your schemas appropriately
- Include examples and documentation

## Development Workflow

1. Define data contracts and schemas
2. Implement processing components
3. Create comprehensive tests
4. Validate data quality
5. Monitor pipeline performance
"""
    
    def _get_data_readme(self) -> str:
        """Get data directory README content."""
        return """# Data Directory

This directory contains data files for the workspace.

## Structure

- `raw/` - Raw, unprocessed data files
- `processed/` - Cleaned and processed data
- `output/` - Final output data ready for consumption

## Guidelines

- Keep raw data immutable
- Document data sources and lineage
- Use consistent naming conventions
- Include data dictionaries where appropriate
- Be mindful of data privacy and security

## File Naming

Use descriptive, consistent naming:
- `YYYY-MM-DD_source_description.ext`
- `dataset_version_subset.ext`
- Include metadata in filenames when helpful

## Data Formats

Prefer standard, interoperable formats:
- CSV for tabular data
- JSON for structured data
- Parquet for large datasets
- Document any custom formats
"""
    
    def _get_models_readme(self) -> str:
        """Get models directory README content."""
        return """# Models Directory

This directory contains trained models and related artifacts.

## Structure

- Store trained models with descriptive names
- Include model metadata and performance metrics
- Save preprocessing artifacts alongside models
- Document model versions and experiments

## Model Naming

Use consistent naming conventions:
- `model_type_version_date.ext`
- `experiment_name_run_id.ext`
- Include key hyperparameters in names

## Model Artifacts

For each model, consider saving:
- Trained model file (.pkl, .joblib, .h5, etc.)
- Model metadata (hyperparameters, training config)
- Performance metrics and evaluation results
- Feature importance or model explanations
- Preprocessing artifacts (scalers, encoders, etc.)

## Model Registry

Consider using a model registry for:
- Model versioning and lineage
- Performance tracking
- Deployment management
- Collaboration and sharing
"""
    
    def _get_schemas_readme(self) -> str:
        """Get schemas directory README content."""
        return """# Schemas Directory

This directory contains data schemas and validation rules.

## Purpose

- Define expected data structures
- Validate data quality and consistency
- Document data contracts
- Enable automated testing

## Schema Formats

Support multiple schema formats:
- JSON Schema for JSON data
- Pydantic models for Python validation
- SQL DDL for database schemas
- Custom validation rules

## Best Practices

- Version your schemas
- Include examples and documentation
- Test schema validation
- Keep schemas close to the data they describe
- Use descriptive names and comments
"""
