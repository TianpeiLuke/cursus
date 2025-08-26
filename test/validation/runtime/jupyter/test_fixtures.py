"""
Test fixtures and mock data for Jupyter integration tests
"""

import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import pandas as pd
import json

# Test data fixtures based on real cursus contracts and specs
SAMPLE_PIPELINE_CONFIG = {
    "name": "xgb_training_with_eval_pipeline",
    "steps": {
        "currency_conversion": {
            "type": "processing",
            "entry_point": "currency_conversion.py",
            "expected_input_paths": {
                "data_input": "/opt/ml/processing/input/data"
            },
            "expected_output_paths": {
                "converted_data": "/opt/ml/processing/output"
            },
            "expected_arguments": {
                "job-type": "training",
                "mode": "per_split",
                "marketplace-id-col": "marketplace_id",
                "default-currency": "USD",
                "enable-conversion": "true",
                "n-workers": "50"
            },
            "required_env_vars": [
                "CURRENCY_CONVERSION_VARS",
                "CURRENCY_CONVERSION_DICT", 
                "MARKETPLACE_INFO",
                "LABEL_FIELD"
            ]
        },
        "xgboost_training": {
            "type": "training",
            "entry_point": "xgboost_training.py",
            "expected_input_paths": {
                "input_path": "/opt/ml/input/data",
                "hyperparameters_s3_uri": "/opt/ml/input/data/config/hyperparameters.json"
            },
            "expected_output_paths": {
                "model_output": "/opt/ml/model",
                "evaluation_output": "/opt/ml/output/data"
            },
            "framework_requirements": {
                "xgboost": "==1.7.6",
                "scikit-learn": ">=0.23.2,<1.0.0",
                "pandas": ">=1.2.0,<2.0.0"
            }
        },
        "xgboost_model_eval": {
            "type": "evaluation",
            "entry_point": "xgboost_model_evaluation.py",
            "expected_input_paths": {
                "model_input": "/opt/ml/processing/input/model",
                "processed_data": "/opt/ml/processing/input/eval_data"
            },
            "expected_output_paths": {
                "eval_output": "/opt/ml/processing/output/eval",
                "metrics_output": "/opt/ml/processing/output/metrics"
            },
            "required_env_vars": [
                "ID_FIELD",
                "LABEL_FIELD"
            ]
        }
    },
    "dependencies": {
        "xgboost_training": ["currency_conversion"],
        "xgboost_model_eval": ["xgboost_training"]
    }
}

SAMPLE_TEST_RESULTS = [
    {
        "script_name": "currency_conversion",
        "success": True,
        "execution_time": 1.5,
        "memory_usage": 100.0,
        "data_size": 1000,
        "timestamp": datetime.now(),
        "test_type": "synthetic",
        "recommendations": ["Consider optimizing data loading"]
    },
    {
        "script_name": "xgboost_training",
        "success": True,
        "execution_time": 45.2,
        "memory_usage": 512.0,
        "data_size": 5000,
        "timestamp": datetime.now() + timedelta(seconds=30),
        "test_type": "synthetic",
        "recommendations": ["Increase memory allocation for large datasets"]
    },
    {
        "script_name": "model_evaluation",
        "success": False,
        "execution_time": 2.1,
        "memory_usage": 150.0,
        "data_size": 1000,
        "timestamp": datetime.now() + timedelta(seconds=60),
        "test_type": "real",
        "error_message": "Missing evaluation data",
        "recommendations": ["Verify input data availability"]
    }
]

SAMPLE_DATA_QUALITY_METRICS = {
    "currency_conversion": {
        "completeness": 0.95,
        "validity": 0.98,
        "schema_compliance": 0.92,
        "overall_score": 0.95
    },
    "xgboost_training": {
        "completeness": 0.88,
        "validity": 0.94,
        "schema_compliance": 0.96,
        "overall_score": 0.93
    },
    "model_evaluation": {
        "completeness": 0.92,
        "validity": 0.89,
        "schema_compliance": 0.94,
        "overall_score": 0.92
    }
}

SAMPLE_PERFORMANCE_DATA = {
    "currency_conversion": {
        "avg_execution_time": 1.2,
        "max_memory_usage": 120.0,
        "throughput": 833.33,
        "error_rate": 0.0
    },
    "xgboost_training": {
        "avg_execution_time": 42.8,
        "max_memory_usage": 600.0,
        "throughput": 116.82,
        "error_rate": 0.05
    },
    "model_evaluation": {
        "avg_execution_time": 2.5,
        "max_memory_usage": 180.0,
        "throughput": 400.0,
        "error_rate": 0.1
    }
}


class MockJupyterEnvironment:
    """Mock Jupyter environment for testing"""
    
    def __init__(self):
        self.display_calls = []
        self.html_content = []
        self.widgets_created = []
    
    def mock_display(self, content):
        """Mock IPython display function"""
        self.display_calls.append(content)
    
    def mock_html(self, content):
        """Mock IPython HTML function"""
        self.html_content.append(content)
        return Mock(value=content)
    
    def mock_widgets(self):
        """Mock ipywidgets"""
        widget_mock = Mock()
        
        # Mock common widget types
        widget_mock.Button = Mock(return_value=Mock())
        widget_mock.Text = Mock(return_value=Mock())
        widget_mock.Textarea = Mock(return_value=Mock())
        widget_mock.Dropdown = Mock(return_value=Mock())
        widget_mock.IntText = Mock(return_value=Mock())
        widget_mock.IntSlider = Mock(return_value=Mock())
        widget_mock.Checkbox = Mock(return_value=Mock())
        widget_mock.Label = Mock(return_value=Mock())
        widget_mock.HTML = Mock(return_value=Mock())
        widget_mock.Output = Mock(return_value=Mock())
        widget_mock.VBox = Mock(return_value=Mock())
        widget_mock.HBox = Mock(return_value=Mock())
        widget_mock.Tab = Mock(return_value=Mock())
        widget_mock.Accordion = Mock(return_value=Mock())
        widget_mock.SelectMultiple = Mock(return_value=Mock())
        widget_mock.IntProgress = Mock(return_value=Mock())
        
        return widget_mock


class MockPlotlyEnvironment:
    """Mock Plotly environment for testing"""
    
    def __init__(self):
        self.figures_created = []
        self.charts_created = []
    
    def mock_go(self):
        """Mock plotly.graph_objects"""
        go_mock = Mock()
        
        # Mock Figure
        figure_mock = Mock()
        figure_mock.add_trace = Mock()
        figure_mock.update_layout = Mock()
        figure_mock.to_html = Mock(return_value="<div>Mock Chart</div>")
        go_mock.Figure = Mock(return_value=figure_mock)
        
        # Mock trace types
        go_mock.Scatter = Mock(return_value=Mock())
        go_mock.Bar = Mock(return_value=Mock())
        go_mock.Histogram = Mock(return_value=Mock())
        go_mock.Heatmap = Mock(return_value=Mock())
        
        self.figures_created.append(figure_mock)
        return go_mock
    
    def mock_px(self):
        """Mock plotly.express"""
        px_mock = Mock()
        
        figure_mock = Mock()
        figure_mock.update_layout = Mock()
        figure_mock.show = Mock()
        
        px_mock.histogram = Mock(return_value=figure_mock)
        px_mock.box = Mock(return_value=figure_mock)
        px_mock.scatter = Mock(return_value=figure_mock)
        px_mock.line = Mock(return_value=figure_mock)
        
        return px_mock
    
    def mock_make_subplots(self):
        """Mock plotly.subplots.make_subplots"""
        subplot_mock = Mock()
        subplot_mock.add_trace = Mock()
        subplot_mock.update_layout = Mock()
        subplot_mock.to_html = Mock(return_value="<div>Mock Subplot</div>")
        
        return Mock(return_value=subplot_mock)


class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def create_sample_dataframe(rows=100, columns=5):
        """Create a sample pandas DataFrame"""
        import numpy as np
        
        data = {}
        for i in range(columns):
            if i % 2 == 0:
                # Numeric columns
                data[f'numeric_col_{i}'] = np.random.randn(rows)
            else:
                # Categorical columns
                categories = ['A', 'B', 'C', 'D']
                data[f'category_col_{i}'] = np.random.choice(categories, rows)
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_test_execution_result(success=True, step_name="test_step"):
        """Create a test execution result"""
        result = {
            "script_name": step_name,
            "success": success,
            "execution_time": 1.5 if success else 0.8,
            "memory_usage": 100.0,
            "data_size": 1000,
            "timestamp": datetime.now(),
            "test_type": "synthetic"
        }
        
        if success:
            result["recommendations"] = ["Test recommendation"]
        else:
            result["error_message"] = "Test error occurred"
            result["recommendations"] = ["Fix the error"]
        
        return result
    
    @staticmethod
    def create_notebook_template_data():
        """Create sample notebook template data"""
        return {
            "name": "test_template",
            "description": "Test template for unit tests",
            "category": "testing",
            "variables": {
                "pipeline_name": "{{ pipeline_name }}",
                "timestamp": "{{ timestamp }}"
            },
            "required_imports": [
                "import pandas as pd",
                "import numpy as np"
            ],
            "cell_templates": [
                {
                    "cell_type": "markdown",
                    "source": "# Test Notebook\n\nPipeline: {{ pipeline_name }}\nGenerated: {{ timestamp }}"
                },
                {
                    "cell_type": "code",
                    "source": "# Import libraries\n{% for import_stmt in required_imports %}\n{{ import_stmt }}\n{% endfor %}"
                },
                {
                    "cell_type": "code",
                    "source": "print('Hello from test template!')"
                }
            ],
            "metadata": {
                "author": "test_user",
                "version": "1.0"
            },
            "created_at": datetime.now().isoformat()
        }


class TempWorkspaceManager:
    """Manage temporary workspaces for testing"""
    
    def __init__(self):
        self.temp_dirs = []
    
    def create_temp_workspace(self):
        """Create a temporary workspace directory"""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        workspace_path = Path(temp_dir) / "test_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        return workspace_path
    
    def create_sample_files(self, workspace_path):
        """Create sample files in workspace"""
        # Create sample pipeline config
        config_path = workspace_path / "pipeline_config.yaml"
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(SAMPLE_PIPELINE_CONFIG, f)
        
        # Create sample data files
        data_dir = workspace_path / "data"
        data_dir.mkdir(exist_ok=True)
        
        sample_df = TestDataGenerator.create_sample_dataframe()
        sample_df.to_csv(data_dir / "sample_data.csv", index=False)
        sample_df.to_parquet(data_dir / "sample_data.parquet", index=False)
        
        # Create sample template file
        template_dir = workspace_path / "templates"
        template_dir.mkdir(exist_ok=True)
        
        template_data = TestDataGenerator.create_notebook_template_data()
        with open(template_dir / "test_template.json", 'w') as f:
            json.dump(template_data, f, indent=2, default=str)
        
        return {
            "config_path": config_path,
            "data_dir": data_dir,
            "template_dir": template_dir
        }
    
    def cleanup(self):
        """Clean up all temporary directories"""
        for temp_dir in self.temp_dirs:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()


class MockExecutorEnvironment:
    """Mock execution environment for testing"""
    
    def __init__(self):
        self.execution_calls = []
        self.test_results = []
    
    def mock_pipeline_script_executor(self):
        """Mock PipelineScriptExecutor"""
        executor_mock = Mock()
        
        def mock_test_script_isolation(script_name, data_source="synthetic", **kwargs):
            # Record the call
            self.execution_calls.append({
                "script_name": script_name,
                "data_source": data_source,
                "kwargs": kwargs
            })
            
            # Return a mock result
            result_mock = Mock()
            result_data = TestDataGenerator.create_test_execution_result(
                success=True, 
                step_name=script_name
            )
            result_mock.model_dump.return_value = result_data
            
            self.test_results.append(result_data)
            return result_mock
        
        executor_mock.test_script_isolation = mock_test_script_isolation
        return executor_mock
    
    def mock_s3_data_downloader(self):
        """Mock S3DataDownloader"""
        downloader_mock = Mock()
        downloader_mock.download_execution_data = Mock(return_value=True)
        downloader_mock.list_available_executions = Mock(return_value=[
            "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test/execution/1",
            "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test/execution/2"
        ])
        return downloader_mock
    
    def mock_real_data_tester(self):
        """Mock RealDataTester"""
        tester_mock = Mock()
        tester_mock.test_with_real_data = Mock(return_value=Mock(
            success=True,
            execution_time=2.5,
            memory_usage=200.0
        ))
        return tester_mock


# Utility functions for test setup
def setup_mock_jupyter_environment():
    """Set up a complete mock Jupyter environment"""
    mock_env = MockJupyterEnvironment()
    
    patches = {
        'display': mock_env.mock_display,
        'HTML': mock_env.mock_html,
        'Markdown': mock_env.mock_html,
        'widgets': mock_env.mock_widgets()
    }
    
    return mock_env, patches


def setup_mock_plotly_environment():
    """Set up a complete mock Plotly environment"""
    mock_env = MockPlotlyEnvironment()
    
    patches = {
        'go': mock_env.mock_go(),
        'px': mock_env.mock_px(),
        'make_subplots': mock_env.mock_make_subplots()
    }
    
    return mock_env, patches


def create_test_session_data():
    """Create test session data for collaboration tests"""
    return {
        "session_id": "test_session_123",
        "user_id": "test_user",
        "pipeline_name": "test_pipeline",
        "workspace_path": Path("/tmp/test_workspace"),
        "bookmarks": [
            {
                "name": "important_cell",
                "cell_index": 5,
                "description": "Key analysis cell",
                "created_at": datetime.now().isoformat(),
                "user_id": "test_user"
            }
        ],
        "annotations": [
            {
                "cell_index": 3,
                "annotation": "This needs review",
                "type": "warning",
                "created_at": datetime.now().isoformat(),
                "user_id": "test_user"
            }
        ]
    }


def create_test_report_data():
    """Create test data for report generation"""
    return {
        "pipeline_name": "test_pipeline",
        "total_steps": 3,
        "total_execution_time": 48.8,
        "execution_date": datetime.now().strftime("%Y-%m-%d"),
        "total_tests": 3,
        "successful_tests": 2,
        "metrics": {
            "avg_response_time": 16.27,
            "throughput": 122.95,
            "error_rate": 0.33
        },
        "performance_metrics": {
            "avg_execution_time": 16.27,
            "memory_usage": 290.67,
            "throughput": 122.95
        },
        "errors": [
            {
                "type": "ValueError",
                "message": "Missing evaluation data",
                "step": "model_evaluation"
            }
        ],
        "data_quality": {
            "completeness": 0.92,
            "validity": 0.94,
            "consistency": 0.94
        }
    }


# Export commonly used fixtures
__all__ = [
    'SAMPLE_PIPELINE_CONFIG',
    'SAMPLE_TEST_RESULTS',
    'SAMPLE_DATA_QUALITY_METRICS',
    'SAMPLE_PERFORMANCE_DATA',
    'MockJupyterEnvironment',
    'MockPlotlyEnvironment',
    'TestDataGenerator',
    'TempWorkspaceManager',
    'MockExecutorEnvironment',
    'setup_mock_jupyter_environment',
    'setup_mock_plotly_environment',
    'create_test_session_data',
    'create_test_report_data'
]
