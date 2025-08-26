"""
Notebook Template Manager for Pipeline Runtime Testing

This module provides template management and generation capabilities for creating
standardized Jupyter notebooks for pipeline testing and analysis.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from jinja2 import Template, Environment, FileSystemLoader

try:
    from IPython.display import display, HTML
    import ipywidgets as widgets
    import nbformat
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
    # Create mock classes for graceful fallback
    class widgets:
        class Output: pass
        class VBox: pass
        class HBox: pass
    class nbformat: pass


class NotebookTemplate(BaseModel):
    """Notebook template configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    description: str
    category: str
    template_path: Optional[Path] = None
    template_content: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    required_imports: List[str] = Field(default_factory=list)
    cell_templates: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class NotebookTemplateManager:
    """
    Manages notebook templates for pipeline runtime testing
    
    Provides template creation, customization, and generation capabilities
    for standardized testing and analysis notebooks.
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """Initialize the template manager"""
        if not JUPYTER_AVAILABLE:
            print("Warning: Jupyter dependencies not available. Template features disabled.")
            return
        
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        self.templates: Dict[str, NotebookTemplate] = {}
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=True
        )
        
        # Initialize built-in templates
        self._initialize_builtin_templates()
    
    def _initialize_builtin_templates(self):
        """Initialize built-in notebook templates"""
        # Basic testing template
        self.register_template(NotebookTemplate(
            name="basic_testing",
            description="Basic pipeline testing notebook",
            category="testing",
            cell_templates=[
                {
                    "cell_type": "markdown",
                    "source": "# Pipeline Testing Notebook\n\nGenerated on: {{ timestamp }}\nPipeline: {{ pipeline_name }}"
                },
                {
                    "cell_type": "code",
                    "source": "# Import required libraries\n{% for import_stmt in required_imports %}\n{{ import_stmt }}\n{% endfor %}"
                },
                {
                    "cell_type": "code",
                    "source": "# Initialize testing environment\nfrom cursus.validation.runtime.jupyter import NotebookInterface\n\nnotebook = NotebookInterface()\nnotebook.display_welcome()"
                },
                {
                    "cell_type": "markdown",
                    "source": "## Load Pipeline Configuration"
                },
                {
                    "cell_type": "code",
                    "source": "# Load pipeline\npipeline_path = '{{ pipeline_path }}'\nnotebook.load_pipeline(pipeline_path)"
                },
                {
                    "cell_type": "markdown",
                    "source": "## Test Individual Steps"
                },
                {
                    "cell_type": "code",
                    "source": "# Test steps interactively\n# Use notebook.test_single_step('step_name') to test individual steps"
                }
            ],
            variables={
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_name": "{{ pipeline_name }}",
                "pipeline_path": "{{ pipeline_path }}"
            },
            required_imports=[
                "import pandas as pd",
                "import numpy as np",
                "from pathlib import Path",
                "import json"
            ]
        ))
        
        # Data analysis template
        self.register_template(NotebookTemplate(
            name="data_analysis",
            description="Data analysis and visualization notebook",
            category="analysis",
            cell_templates=[
                {
                    "cell_type": "markdown",
                    "source": "# Data Analysis Notebook\n\nPipeline: {{ pipeline_name }}\nGenerated: {{ timestamp }}"
                },
                {
                    "cell_type": "code",
                    "source": "# Import libraries for data analysis\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport plotly.express as px\nimport plotly.graph_objects as go\nfrom cursus.validation.runtime.jupyter import VisualizationReporter\n\n# Set up plotting\nplt.style.use('default')\nsns.set_palette('husl')\n%matplotlib inline"
                },
                {
                    "cell_type": "markdown",
                    "source": "## Load Test Results"
                },
                {
                    "cell_type": "code",
                    "source": "# Initialize visualization reporter\nreporter = VisualizationReporter()\n\n# Load test results\n# Add your test results here\n# reporter.add_test_result(...)"
                },
                {
                    "cell_type": "markdown",
                    "source": "## Performance Analysis"
                },
                {
                    "cell_type": "code",
                    "source": "# Create performance dashboard\ndashboard = reporter.create_interactive_dashboard()\ndisplay(dashboard)"
                },
                {
                    "cell_type": "markdown",
                    "source": "## Data Quality Assessment"
                },
                {
                    "cell_type": "code",
                    "source": "# Generate data quality report\nquality_report = reporter.create_data_quality_report()\nif quality_report:\n    display(quality_report)"
                }
            ],
            variables={
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_name": "{{ pipeline_name }}"
            }
        ))
        
        # Debugging template
        self.register_template(NotebookTemplate(
            name="debugging",
            description="Interactive debugging notebook",
            category="debugging",
            cell_templates=[
                {
                    "cell_type": "markdown",
                    "source": "# Pipeline Debugging Notebook\n\nPipeline: {{ pipeline_name }}\nGenerated: {{ timestamp }}"
                },
                {
                    "cell_type": "code",
                    "source": "# Import debugging tools\nfrom cursus.validation.runtime.jupyter import InteractiveDebugger\nfrom cursus.validation.runtime.core import PipelineScriptExecutor\n\n# Initialize debugger\ndebugger = InteractiveDebugger()\nsession_id = debugger.start_debug_session('{{ pipeline_name }}')"
                },
                {
                    "cell_type": "markdown",
                    "source": "## Debug Interface"
                },
                {
                    "cell_type": "code",
                    "source": "# Create interactive debug interface\ndebug_interface = debugger.create_debug_interface()\ndisplay(debug_interface)"
                },
                {
                    "cell_type": "markdown",
                    "source": "## Set Breakpoints"
                },
                {
                    "cell_type": "code",
                    "source": "# Set breakpoints for specific steps\n# debugger.set_breakpoint('step_name')\n# debugger.set_breakpoint('step_name', 'condition')"
                },
                {
                    "cell_type": "markdown",
                    "source": "## Execute Pipeline with Debugging"
                },
                {
                    "cell_type": "code",
                    "source": "# Execute pipeline with debugging enabled\n# Your pipeline execution code here"
                }
            ],
            variables={
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_name": "{{ pipeline_name }}"
            }
        ))
    
    def register_template(self, template: NotebookTemplate):
        """Register a new template"""
        self.templates[template.name] = template
        print(f"Template '{template.name}' registered successfully")
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates"""
        return [
            {
                'name': template.name,
                'description': template.description,
                'category': template.category,
                'created_at': template.created_at
            }
            for template in self.templates.values()
        ]
    
    def get_template(self, name: str) -> Optional[NotebookTemplate]:
        """Get a template by name"""
        return self.templates.get(name)
    
    def create_notebook_from_template(self, 
                                    template_name: str,
                                    variables: Optional[Dict[str, Any]] = None,
                                    output_path: Optional[Path] = None) -> Optional[str]:
        """Create a notebook from a template"""
        if not JUPYTER_AVAILABLE:
            print("Cannot create notebook: Jupyter dependencies not available")
            return None
        
        template = self.get_template(template_name)
        if not template:
            print(f"Template '{template_name}' not found")
            return None
        
        # Merge variables
        template_vars = template.variables.copy()
        if variables:
            template_vars.update(variables)
        
        # Create notebook
        nb = nbformat.v4.new_notebook()
        
        # Add metadata
        nb.metadata.update({
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.8.0'
            },
            'template_info': {
                'template_name': template_name,
                'generated_at': datetime.now().isoformat(),
                'variables': template_vars
            }
        })
        
        # Process cell templates
        for cell_template in template.cell_templates:
            cell_type = cell_template['cell_type']
            # Use the environment with autoescape enabled instead of direct Template instantiation
            source_template = self.jinja_env.from_string(cell_template['source'])
            source = source_template.render(**template_vars)
            
            if cell_type == 'code':
                cell = nbformat.v4.new_code_cell(source)
            elif cell_type == 'markdown':
                cell = nbformat.v4.new_markdown_cell(source)
            else:
                continue
            
            nb.cells.append(cell)
        
        # Save notebook if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            
            print(f"Notebook created: {output_path}")
            return str(output_path)
        else:
            # Return notebook as string
            return nbformat.writes(nb)
    
    def create_template_selector_widget(self) -> Optional[widgets.VBox]:
        """Create an interactive template selector widget"""
        if not JUPYTER_AVAILABLE:
            return None
        
        # Template selection
        template_dropdown = widgets.Dropdown(
            options=[(f"{t.name} - {t.description}", t.name) for t in self.templates.values()],
            description='Template:',
            disabled=False
        )
        
        # Variable inputs
        pipeline_name_text = widgets.Text(
            placeholder='Enter pipeline name',
            description='Pipeline:',
            value='my_pipeline'
        )
        
        pipeline_path_text = widgets.Text(
            placeholder='Enter pipeline path',
            description='Path:',
            value='./pipeline.yaml'
        )
        
        output_path_text = widgets.Text(
            placeholder='Enter output path',
            description='Output:',
            value='./generated_notebook.ipynb'
        )
        
        # Generate button
        generate_button = widgets.Button(
            description='Generate Notebook',
            button_style='primary'
        )
        
        # Output area
        output_area = widgets.Output()
        
        def generate_notebook(*args):
            """Generate notebook from selected template"""
            with output_area:
                output_area.clear_output()
                
                template_name = template_dropdown.value
                variables = {
                    'pipeline_name': pipeline_name_text.value,
                    'pipeline_path': pipeline_path_text.value,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                output_path = Path(output_path_text.value) if output_path_text.value else None
                
                try:
                    result = self.create_notebook_from_template(
                        template_name, variables, output_path
                    )
                    
                    if result:
                        if output_path:
                            print(f"✅ Notebook generated successfully: {output_path}")
                        else:
                            print("✅ Notebook generated successfully")
                            print("Notebook content:")
                            print(result[:500] + "..." if len(result) > 500 else result)
                    else:
                        print("❌ Failed to generate notebook")
                        
                except Exception as e:
                    print(f"❌ Error generating notebook: {e}")
        
        generate_button.on_click(generate_notebook)
        
        # Template preview
        preview_button = widgets.Button(
            description='Preview Template',
            button_style='info'
        )
        
        preview_output = widgets.Output()
        
        def preview_template(*args):
            """Preview selected template"""
            with preview_output:
                preview_output.clear_output()
                
                template_name = template_dropdown.value
                template = self.get_template(template_name)
                
                if template:
                    print(f"Template: {template.name}")
                    print(f"Description: {template.description}")
                    print(f"Category: {template.category}")
                    print(f"Cells: {len(template.cell_templates)}")
                    print(f"Required Imports: {len(template.required_imports)}")
                    
                    print("\nCell Structure:")
                    for i, cell in enumerate(template.cell_templates):
                        cell_type = cell['cell_type']
                        source_preview = cell['source'][:100] + "..." if len(cell['source']) > 100 else cell['source']
                        print(f"  {i+1}. {cell_type.title()}: {source_preview}")
                else:
                    print("Template not found")
        
        preview_button.on_click(preview_template)
        
        # Layout
        template_section = widgets.VBox([
            widgets.HTML("<h4>Template Selection</h4>"),
            template_dropdown,
            widgets.HBox([preview_button]),
            preview_output
        ])
        
        variables_section = widgets.VBox([
            widgets.HTML("<h4>Template Variables</h4>"),
            pipeline_name_text,
            pipeline_path_text,
            output_path_text
        ])
        
        generation_section = widgets.VBox([
            widgets.HTML("<h4>Generate Notebook</h4>"),
            generate_button,
            output_area
        ])
        
        main_widget = widgets.VBox([
            widgets.HTML("<h3>Notebook Template Generator</h3>"),
            template_section,
            variables_section,
            generation_section
        ])
        
        return main_widget
    
    def save_template_to_file(self, template_name: str, file_path: Path):
        """Save a template to a file"""
        template = self.get_template(template_name)
        if not template:
            print(f"Template '{template_name}' not found")
            return
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        template_data = template.model_dump()
        template_data['created_at'] = template_data['created_at'].isoformat()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2)
        
        print(f"Template saved to: {file_path}")
    
    def load_template_from_file(self, file_path: Path) -> Optional[NotebookTemplate]:
        """Load a template from a file"""
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Template file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            
            # Convert datetime string back to datetime object
            if 'created_at' in template_data:
                template_data['created_at'] = datetime.fromisoformat(template_data['created_at'])
            
            template = NotebookTemplate(**template_data)
            self.register_template(template)
            
            return template
            
        except Exception as e:
            print(f"Error loading template: {e}")
            return None
    
    def create_custom_template(self, 
                             name: str,
                             description: str,
                             category: str,
                             cell_definitions: List[Dict[str, Any]],
                             variables: Optional[Dict[str, Any]] = None,
                             required_imports: Optional[List[str]] = None) -> NotebookTemplate:
        """Create a custom template"""
        template = NotebookTemplate(
            name=name,
            description=description,
            category=category,
            cell_templates=cell_definitions,
            variables=variables or {},
            required_imports=required_imports or []
        )
        
        self.register_template(template)
        return template
    
    def export_templates(self, output_dir: Path):
        """Export all templates to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for template_name in self.templates:
            file_path = output_dir / f"{template_name}.json"
            self.save_template_to_file(template_name, file_path)
        
        print(f"All templates exported to: {output_dir}")
    
    def import_templates(self, template_dir: Path):
        """Import templates from a directory"""
        template_dir = Path(template_dir)
        if not template_dir.exists():
            print(f"Template directory not found: {template_dir}")
            return
        
        imported_count = 0
        for template_file in template_dir.glob("*.json"):
            template = self.load_template_from_file(template_file)
            if template:
                imported_count += 1
        
        print(f"Imported {imported_count} templates from: {template_dir}")
    
    def get_template_categories(self) -> List[str]:
        """Get all template categories"""
        categories = set()
        for template in self.templates.values():
            categories.add(template.category)
        return sorted(list(categories))
    
    def get_templates_by_category(self, category: str) -> List[NotebookTemplate]:
        """Get templates by category"""
        return [
            template for template in self.templates.values()
            if template.category == category
        ]
