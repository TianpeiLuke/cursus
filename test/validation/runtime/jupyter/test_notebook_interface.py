"""
Unit tests for notebook_interface.py module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime

import pytest
import pandas as pd

# Import the module under test
from src.cursus.validation.runtime.jupyter.notebook_interface import (
    NotebookInterface,
    NotebookSession,
    JUPYTER_AVAILABLE
)
from src.cursus.workspace.core.registry import WorkspaceComponentRegistry


class TestNotebookSession(unittest.TestCase):
    """Test cases for NotebookSession model"""
    
    def test_notebook_session_creation(self):
        """Test NotebookSession creation with required fields"""
        session = NotebookSession(
            session_id="test_session",
            workspace_dir=Path("/tmp/test")
        )
        
        self.assertEqual(session.session_id, "test_session")
        self.assertEqual(session.workspace_dir, Path("/tmp/test"))
        self.assertIsNone(session.pipeline_name)
        self.assertIsNone(session.current_step)
        self.assertIsNone(session.test_results)
    
    def test_notebook_session_with_optional_fields(self):
        """Test NotebookSession creation with optional fields"""
        test_results = {"step1": {"success": True}}
        
        session = NotebookSession(
            session_id="test_session",
            workspace_dir=Path("/tmp/test"),
            pipeline_name="test_pipeline",
            current_step="step1",
            test_results=test_results
        )
        
        self.assertEqual(session.pipeline_name, "test_pipeline")
        self.assertEqual(session.current_step, "step1")
        self.assertEqual(session.test_results, test_results)


class TestNotebookInterface(unittest.TestCase):
    """Test cases for NotebookInterface class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = Path(self.temp_dir) / "test_workspace"
        self.interface = NotebookInterface(str(self.workspace_dir))
    
    def tearDown(self):
        """Clean up test fixtures"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test NotebookInterface initialization"""
        self.assertTrue(self.workspace_dir.exists())
        self.assertIsInstance(self.interface.session, NotebookSession)
        self.assertTrue(self.interface.session.session_id.startswith("session_"))
        self.assertEqual(self.interface.session.workspace_dir, self.workspace_dir)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_display_welcome(self, mock_html, mock_display):
        """Test display_welcome method"""
        self.interface.display_welcome()
        
        mock_html.assert_called_once()
        mock_display.assert_called_once()
        
        # Check that HTML content contains expected elements
        html_content = mock_html.call_args[0][0]
        self.assertIn("Pipeline Script Functionality Testing", html_content)
        self.assertIn(self.interface.session.session_id, html_content)
        self.assertIn(str(self.workspace_dir), html_content)
    
    def test_discover_pipeline_config_yaml(self):
        """Test pipeline config discovery for YAML files"""
        # Create a test YAML config file
        config_data = {"steps": ["step1", "step2"], "name": "test_pipeline"}
        config_path = self.workspace_dir / "test_pipeline.yaml"
        
        with patch('yaml.safe_load', return_value=config_data):
            with patch('builtins.open', unittest.mock.mock_open(read_data="test")):
                with patch.object(Path, 'exists', return_value=True):
                    result = self.interface._discover_pipeline_config("test_pipeline")
        
        self.assertEqual(result, config_data)
    
    def test_discover_pipeline_config_json(self):
        """Test pipeline config discovery for JSON files"""
        config_data = {"steps": ["step1", "step2"], "name": "test_pipeline"}
        config_path = self.workspace_dir / "test_pipeline.json"

        with patch('json.load', return_value=config_data):
            with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(config_data))):
                with patch.object(Path, 'exists', return_value=True):
                    result = self.interface._discover_pipeline_config("test_pipeline")

        self.assertEqual(result, config_data)
    
    def test_discover_pipeline_config_not_found(self):
        """Test pipeline config discovery when no config found"""
        with patch.object(Path, 'exists', return_value=False):
            result = self.interface._discover_pipeline_config("nonexistent_pipeline")
        
        self.assertIsNone(result)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.Markdown')
    def test_load_pipeline_success(self, mock_markdown, mock_display):
        """Test successful pipeline loading"""
        config_data = {"steps": ["step1", "step2"], "name": "test_pipeline"}
        
        with patch.object(self.interface, '_discover_pipeline_config', return_value=config_data):
            with patch.object(self.interface, '_display_pipeline_summary'):
                result = self.interface.load_pipeline("test_pipeline")
        
        self.assertEqual(result, config_data)
        self.assertEqual(self.interface.session.pipeline_name, "test_pipeline")
        mock_markdown.assert_called_once()
        mock_display.assert_called()
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_load_pipeline_failure(self, mock_html, mock_display):
        """Test pipeline loading failure"""
        with patch.object(self.interface, '_discover_pipeline_config', return_value=None):
            result = self.interface.load_pipeline("nonexistent_pipeline")
        
        self.assertIsNone(result)
        mock_html.assert_called_once()
        html_content = mock_html.call_args[0][0]
        self.assertIn("Failed to load pipeline configuration", html_content)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.PipelineScriptExecutor')
    def test_execute_step_test_success(self, mock_executor_class):
        """Test successful step execution"""
        # Mock the executor and its result
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor

        mock_result = Mock()
        mock_result.model_dump.return_value = {
            'script_name': 'test_step',
            'success': True,
            'execution_time': 1.5,
            'memory_usage': 100
        }
        mock_executor.test_script_isolation.return_value = mock_result

        # Mock the script discovery to succeed by patching the actual method that gets called
        with patch.object(self.interface, 'script_executor', mock_executor):
            result = self.interface._execute_step_test("test_step", "synthetic")

        self.assertTrue(result['success'])
        self.assertEqual(result['script_name'], 'test_step')
        self.assertIn('test_step', self.interface.session.test_results)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.PipelineScriptExecutor')
    def test_execute_step_test_failure(self, mock_executor_class):
        """Test step execution failure"""
        # Mock the executor to raise an exception
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor
        mock_executor.test_script_isolation.side_effect = Exception("Test error")

        # Mock the script discovery to succeed but execution to fail
        with patch.object(self.interface, 'script_executor', mock_executor):
            result = self.interface._execute_step_test("test_step", "synthetic")

        self.assertFalse(result['success'])
        self.assertEqual(result['error'], "Test error")
        self.assertIn('test_step', self.interface.session.test_results)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_display_step_result_success(self, mock_html, mock_display):
        """Test displaying successful step result"""
        result = {
            'script_name': 'test_step',
            'success': True,
            'execution_time': 1.5,
            'memory_usage': 100,
            'recommendations': ['Use more memory']
        }
        
        self.interface._display_step_result(result)
        
        mock_html.assert_called_once()
        mock_display.assert_called_once()
        
        html_content = mock_html.call_args[0][0]
        self.assertIn("✅ test_step", html_content)
        self.assertIn("Success", html_content)
        self.assertIn("1.50s", html_content)
        self.assertIn("100 MB", html_content)
        self.assertIn("Use more memory", html_content)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_display_step_result_failure(self, mock_html, mock_display):
        """Test displaying failed step result"""
        result = {
            'script_name': 'test_step',
            'success': False,
            'error_message': 'Test failed'
        }
        
        self.interface._display_step_result(result)
        
        mock_html.assert_called_once()
        mock_display.assert_called_once()
        
        html_content = mock_html.call_args[0][0]
        self.assertIn("❌ test_step", html_content)
        self.assertIn("Failed", html_content)
        self.assertIn("Test failed", html_content)
    
    def test_explore_data_with_dataframe(self):
        """Test data exploration with pandas DataFrame"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        with patch.object(self.interface, '_display_data_summary') as mock_display:
            result = self.interface.explore_data(df, interactive=False)
        
        mock_display.assert_called_once_with(df)
    
    def test_explore_data_with_csv_file(self):
        """Test data exploration with CSV file"""
        # Create a test CSV file
        csv_path = self.workspace_dir / "test.csv"
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        df.to_csv(csv_path, index=False)
        
        with patch.object(self.interface, '_display_data_summary') as mock_display:
            result = self.interface.explore_data(str(csv_path), interactive=False)
        
        mock_display.assert_called_once()
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_explore_data_invalid_file(self, mock_html, mock_display):
        """Test data exploration with invalid file"""
        result = self.interface.explore_data("nonexistent.csv", interactive=False)
        
        self.assertIsNone(result)
        mock_html.assert_called_once()
        html_content = mock_html.call_args[0][0]
        self.assertIn("Error loading data", html_content)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_display_data_summary(self, mock_html, mock_display):
        """Test data summary display"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        result = self.interface._display_data_summary(df)
        
        self.assertEqual(result.shape, df.shape)
        mock_html.assert_called_once()
        mock_display.assert_called()
        
        html_content = mock_html.call_args[0][0]
        self.assertIn("3 rows × 2 columns", html_content)
        self.assertIn("col1, col2", html_content)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_display_pipeline_summary(self, mock_html, mock_display):
        """Test pipeline summary display"""
        config = {
            'steps': ['step1', 'step2', 'step3']
        }
        
        self.interface._display_pipeline_summary(config)
        
        mock_html.assert_called_once()
        mock_display.assert_called_once()
        
        html_content = mock_html.call_args[0][0]
        self.assertIn("3", html_content)  # Number of steps
        self.assertIn("step1", html_content)
        self.assertIn("step2", html_content)
        self.assertIn("step3", html_content)
    
    def test_display_pipeline_summary_with_dict_steps(self):
        """Test pipeline summary display with dictionary steps"""
        config = {
            'steps': {
                'step1': {'type': 'processing'},
                'step2': {'type': 'training'}
            }
        }
        
        with patch('src.cursus.validation.runtime.jupyter.notebook_interface.display'):
            with patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML') as mock_html:
                self.interface._display_pipeline_summary(config)
        
        html_content = mock_html.call_args[0][0]
        self.assertIn("2", html_content)  # Number of steps
        self.assertIn("step1", html_content)
        self.assertIn("step2", html_content)
    
    def test_get_session_info(self):
        """Test getting session information"""
        info = self.interface.get_session_info()
        
        expected_keys = [
            'session_id', 'workspace_dir', 'pipeline_name', 
            'current_step', 'test_results_count', 'jupyter_available'
        ]
        
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['session_id'], self.interface.session.session_id)
        self.assertEqual(info['workspace_dir'], str(self.interface.session.workspace_dir))
        self.assertEqual(info['jupyter_available'], JUPYTER_AVAILABLE)


@unittest.skipIf(not JUPYTER_AVAILABLE, "Jupyter dependencies not available")
class TestNotebookInterfaceWithJupyter(unittest.TestCase):
    """Test cases that require Jupyter dependencies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = Path(self.temp_dir) / "test_workspace"
        self.interface = NotebookInterface(str(self.workspace_dir))
    
    def tearDown(self):
        """Clean up test fixtures"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.widgets')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    def test_create_interactive_step_tester(self, mock_display, mock_widgets):
        """Test creating interactive step tester widget"""
        # Mock widgets
        mock_dropdown = Mock()
        mock_textarea = Mock()
        mock_button = Mock()
        mock_output = Mock()
        mock_vbox = Mock()
        
        mock_widgets.Dropdown.return_value = mock_dropdown
        mock_widgets.Textarea.return_value = mock_textarea
        mock_widgets.Button.return_value = mock_button
        mock_widgets.Output.return_value = mock_output
        mock_widgets.VBox.return_value = mock_vbox
        
        result = self.interface._create_interactive_step_tester("test_step", "synthetic")
        
        self.assertIsNotNone(result)
        mock_widgets.Dropdown.assert_called()
        mock_widgets.Textarea.assert_called()
        mock_widgets.Button.assert_called()
        mock_widgets.Output.assert_called()
        mock_widgets.VBox.assert_called()
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.widgets')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    def test_create_interactive_data_explorer(self, mock_display, mock_widgets):
        """Test creating interactive data explorer widget"""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        
        # Mock widgets
        mock_dropdown = Mock()
        mock_button = Mock()
        mock_output = Mock()
        mock_vbox = Mock()
        mock_hbox = Mock()
        
        mock_widgets.Dropdown.return_value = mock_dropdown
        mock_widgets.Button.return_value = mock_button
        mock_widgets.Output.return_value = mock_output
        mock_widgets.VBox.return_value = mock_vbox
        mock_widgets.HBox.return_value = mock_hbox

        # Mock the interactive function to avoid widget validation
        with patch('src.cursus.validation.runtime.jupyter.notebook_interface.interactive') as mock_interactive:
            mock_interactive.return_value = Mock()
            result = self.interface._create_interactive_data_explorer(df)
        
        self.assertIsNotNone(result)
        mock_widgets.Dropdown.assert_called()
        # Button is not called in this specific method, only dropdowns
        mock_widgets.Output.assert_called()


class TestNotebookInterfaceWorkspaceFeatures(unittest.TestCase):
    """Test cases for workspace-aware features"""
    
    def setUp(self):
        """Set up test fixtures with workspace context"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = Path(self.temp_dir) / "test_workspace"
        self.workspace_root = str(Path(self.temp_dir) / "workspace_root")
        
        # Create workspace root directory
        Path(self.workspace_root).mkdir(parents=True, exist_ok=True)
        
        self.interface = NotebookInterface(
            str(self.workspace_dir), 
            workspace_root=self.workspace_root
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization_with_workspace_root(self):
        """Test NotebookInterface initialization with workspace root"""
        self.assertEqual(self.interface.session.workspace_root, self.workspace_root)
        self.assertIsNotNone(self.interface.workspace_registry)
        self.assertIsInstance(self.interface.workspace_registry, WorkspaceComponentRegistry)
    
    def test_initialization_without_workspace_root(self):
        """Test NotebookInterface initialization without workspace root"""
        interface = NotebookInterface(str(self.workspace_dir))
        self.assertIsNone(interface.session.workspace_root)
        self.assertIsNone(interface.workspace_registry)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_display_welcome_with_workspace_info(self, mock_html, mock_display):
        """Test display_welcome method with workspace information"""
        self.interface.session.selected_developer = "test_developer"
        self.interface.display_welcome()
        
        mock_html.assert_called_once()
        mock_display.assert_called_once()
        
        html_content = mock_html.call_args[0][0]
        self.assertIn("Workspace Root:", html_content)
        self.assertIn(self.workspace_root, html_content)
        self.assertIn("Selected Developer:", html_content)
        self.assertIn("test_developer", html_content)
    
    def test_get_session_info_with_workspace_context(self):
        """Test getting session information with workspace context"""
        self.interface.session.selected_developer = "test_developer"
        info = self.interface.get_session_info()
        
        expected_keys = [
            'session_id', 'workspace_dir', 'workspace_root', 'selected_developer',
            'pipeline_name', 'current_step', 'test_results_count', 'jupyter_available'
        ]
        
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['workspace_root'], self.workspace_root)
        self.assertEqual(info['selected_developer'], "test_developer")
    
    def test_execute_step_test_with_developer_id(self):
        """Test step execution with developer ID"""
        with patch.object(self.interface, 'script_executor') as mock_executor:
            mock_result = Mock()
            mock_result.model_dump.return_value = {
                'script_name': 'test_step',
                'success': True,
                'execution_time': 1.5
            }
            mock_executor.test_script_isolation.return_value = mock_result
            
            result = self.interface._execute_step_test(
                "test_step", "synthetic", developer_id="test_developer"
            )
            
            # Verify developer_id was passed to script executor
            mock_executor.test_script_isolation.assert_called_once_with(
                "test_step", "synthetic", developer_id="test_developer"
            )
            
            # Verify workspace context in result
            self.assertEqual(result['developer_id'], "test_developer")
            self.assertIn('workspace_context', result)
            self.assertEqual(result['workspace_context']['workspace_root'], self.workspace_root)
            self.assertEqual(result['workspace_context']['selected_developer'], "test_developer")
    
    def test_execute_step_test_with_workspace_context_on_error(self):
        """Test step execution error includes workspace context"""
        with patch.object(self.interface, 'script_executor') as mock_executor:
            mock_executor.test_script_isolation.side_effect = Exception("Test error")
            
            result = self.interface._execute_step_test(
                "test_step", "synthetic", developer_id="test_developer"
            )
            
            self.assertFalse(result['success'])
            self.assertEqual(result['error'], "Test error")
            self.assertEqual(result['developer_id'], "test_developer")
            self.assertIn('workspace_context', result)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_select_workspace_developer_without_registry(self, mock_html, mock_display):
        """Test developer selection without workspace registry"""
        interface = NotebookInterface(str(self.workspace_dir))  # No workspace root
        result = interface.select_workspace_developer()
        
        self.assertIsNone(result)
        mock_html.assert_called_once()
        html_content = mock_html.call_args[0][0]
        self.assertIn("No workspace context available", html_content)
    
    def test_set_developer(self):
        """Test setting developer for session"""
        with patch('src.cursus.validation.runtime.jupyter.notebook_interface.display'):
            with patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML'):
                result = self.interface._set_developer("test_developer")
        
        self.assertEqual(result, "test_developer")
        self.assertEqual(self.interface.session.selected_developer, "test_developer")
    
    def test_set_developer_none(self):
        """Test setting developer to None"""
        with patch('src.cursus.validation.runtime.jupyter.notebook_interface.display'):
            with patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML'):
                result = self.interface._set_developer(None)
        
        self.assertIsNone(result)
        self.assertIsNone(self.interface.session.selected_developer)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_display_developer_info(self, mock_html, mock_display):
        """Test displaying developer information"""
        mock_components = {
            'scripts': {'script1': {'developer_id': 'test_dev'}},
            'builders': {'builder1': {'developer_id': 'test_dev'}},
            'configs': {},
            'contracts': {},
            'specs': {}
        }
        
        with patch.object(self.interface.workspace_registry, 'discover_components', 
                         return_value=mock_components):
            self.interface._display_developer_info("test_developer")
        
        mock_html.assert_called_once()
        mock_display.assert_called_once()
        
        html_content = mock_html.call_args[0][0]
        self.assertIn("Developer: test_developer", html_content)
        self.assertIn("Scripts:", html_content)
        self.assertIn("Builders:", html_content)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_display_developer_info_error(self, mock_html, mock_display):
        """Test displaying developer information with error"""
        with patch.object(self.interface.workspace_registry, 'discover_components', 
                         side_effect=Exception("Discovery error")):
            self.interface._display_developer_info("test_developer")
        
        mock_html.assert_called_once()
        html_content = mock_html.call_args[0][0]
        self.assertIn("Error getting developer info", html_content)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_list_workspace_components_without_registry(self, mock_html, mock_display):
        """Test listing workspace components without registry"""
        interface = NotebookInterface(str(self.workspace_dir))  # No workspace root
        result = interface.list_workspace_components()
        
        self.assertIsNone(result)
        mock_html.assert_called_once()
        html_content = mock_html.call_args[0][0]
        self.assertIn("No workspace context available", html_content)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_list_workspace_components_success(self, mock_html, mock_display):
        """Test successful workspace components listing"""
        mock_components = {
            'summary': {
                'total_components': 5,
                'developers': ['dev1', 'dev2'],
                'step_types': ['processing', 'training']
            },
            'scripts': {
                'script1': {
                    'developer_id': 'dev1',
                    'step_name': 'process_data',
                    'step_type': 'processing'
                }
            },
            'builders': {
                'builder1': {
                    'developer_id': 'dev2',
                    'class_name': 'TrainingBuilder'
                }
            },
            'configs': {},
            'contracts': {},
            'specs': {}
        }
        
        with patch.object(self.interface.workspace_registry, 'discover_components', 
                         return_value=mock_components):
            with patch('pandas.DataFrame') as mock_df:
                mock_df.return_value = Mock()
                result = self.interface.list_workspace_components()
        
        self.assertEqual(result, mock_components)
        mock_html.assert_called()
        mock_display.assert_called()
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_list_workspace_components_with_developer_id(self, mock_html, mock_display):
        """Test listing workspace components for specific developer"""
        mock_components = {
            'summary': {
                'total_components': 2,
                'developers': ['test_dev'],
                'step_types': ['processing']
            },
            'scripts': {
                'script1': {
                    'developer_id': 'test_dev',
                    'step_name': 'process_data'
                }
            },
            'builders': {},
            'configs': {},
            'contracts': {},
            'specs': {}
        }
        
        with patch.object(self.interface.workspace_registry, 'discover_components', 
                         return_value=mock_components) as mock_discover:
            result = self.interface.list_workspace_components("test_dev")
        
        mock_discover.assert_called_once_with("test_dev")
        self.assertEqual(result, mock_components)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_list_workspace_components_error(self, mock_html, mock_display):
        """Test listing workspace components with error"""
        with patch.object(self.interface.workspace_registry, 'discover_components', 
                         side_effect=Exception("Discovery error")):
            result = self.interface.list_workspace_components()
        
        self.assertIsNone(result)
        mock_html.assert_called_once()
        html_content = mock_html.call_args[0][0]
        self.assertIn("Error listing components", html_content)


@unittest.skipIf(not JUPYTER_AVAILABLE, "Jupyter dependencies not available")
class TestNotebookInterfaceWorkspaceWidgets(unittest.TestCase):
    """Test cases for workspace-aware interactive widgets"""
    
    def setUp(self):
        """Set up test fixtures with workspace context"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = Path(self.temp_dir) / "test_workspace"
        self.workspace_root = str(Path(self.temp_dir) / "workspace_root")
        
        Path(self.workspace_root).mkdir(parents=True, exist_ok=True)
        
        self.interface = NotebookInterface(
            str(self.workspace_dir), 
            workspace_root=self.workspace_root
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.widgets')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    def test_create_interactive_step_tester_with_developer_options(self, mock_display, mock_widgets):
        """Test creating interactive step tester with developer options"""
        # Mock workspace registry discovery
        mock_components = {
            'summary': {
                'developers': ['dev1', 'dev2']
            }
        }
        
        with patch.object(self.interface.workspace_registry, 'discover_components', 
                         return_value=mock_components):
            
            # Mock widgets
            mock_dropdown = Mock()
            mock_textarea = Mock()
            mock_button = Mock()
            mock_output = Mock()
            mock_vbox = Mock()
            
            mock_widgets.Dropdown.return_value = mock_dropdown
            mock_widgets.Textarea.return_value = mock_textarea
            mock_widgets.Button.return_value = mock_button
            mock_widgets.Output.return_value = mock_output
            mock_widgets.VBox.return_value = mock_vbox
            
            result = self.interface._create_interactive_step_tester("test_step", "synthetic")
            
            # Verify developer dropdown was created with discovered developers
            dropdown_calls = mock_widgets.Dropdown.call_args_list
            developer_dropdown_call = None
            for call in dropdown_calls:
                if 'Developer:' in str(call):
                    developer_dropdown_call = call
                    break
            
            self.assertIsNotNone(developer_dropdown_call)
            self.assertIsNotNone(result)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.widgets')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    def test_create_developer_selector_success(self, mock_display, mock_widgets):
        """Test creating developer selector widget successfully"""
        # Mock workspace registry discovery
        mock_components = {
            'summary': {
                'developers': ['dev1', 'dev2', 'dev3']
            }
        }
        
        with patch.object(self.interface.workspace_registry, 'discover_components', 
                         return_value=mock_components):
            
            # Mock widgets
            mock_dropdown = Mock()
            mock_button = Mock()
            mock_output = Mock()
            mock_vbox = Mock()
            
            mock_widgets.Dropdown.return_value = mock_dropdown
            mock_widgets.Button.return_value = mock_button
            mock_widgets.Output.return_value = mock_output
            mock_widgets.VBox.return_value = mock_vbox
            
            result = self.interface._create_developer_selector()
            
            self.assertIsNotNone(result)
            mock_widgets.Dropdown.assert_called_once()
            mock_widgets.Button.assert_called_once()
            mock_widgets.VBox.assert_called_once()
            mock_display.assert_called_once()
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.widgets')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_create_developer_selector_no_developers(self, mock_html, mock_display, mock_widgets):
        """Test creating developer selector when no developers found"""
        # Mock workspace registry discovery with no developers
        mock_components = {
            'summary': {
                'developers': []
            }
        }
        
        with patch.object(self.interface.workspace_registry, 'discover_components', 
                         return_value=mock_components):
            result = self.interface._create_developer_selector()
            
            self.assertIsNone(result)
            mock_html.assert_called_once()
            html_content = mock_html.call_args[0][0]
            self.assertIn("No developers found", html_content)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.widgets')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.display')
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML')
    def test_create_developer_selector_error(self, mock_html, mock_display, mock_widgets):
        """Test creating developer selector with discovery error"""
        with patch.object(self.interface.workspace_registry, 'discover_components', 
                         side_effect=Exception("Discovery error")):
            result = self.interface._create_developer_selector()
            
            self.assertIsNone(result)
            mock_html.assert_called_once()
            html_content = mock_html.call_args[0][0]
            self.assertIn("Error creating developer selector", html_content)
    
    def test_select_workspace_developer_non_interactive(self):
        """Test non-interactive developer selection"""
        result = self.interface.select_workspace_developer("test_dev", interactive=False)
        
        self.assertEqual(result, "test_dev")
        self.assertEqual(self.interface.session.selected_developer, "test_dev")


class TestNotebookInterfaceEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = Path(self.temp_dir) / "test_workspace"
        self.interface = NotebookInterface(str(self.workspace_dir))
    
    def tearDown(self):
        """Clean up test fixtures"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_load_pipeline_with_invalid_json(self):
        """Test loading pipeline with invalid JSON"""
        config_path = self.workspace_dir / "invalid.json"
        config_path.write_text("invalid json content")
        
        with patch('src.cursus.validation.runtime.jupyter.notebook_interface.display'):
            with patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML'):
                result = self.interface.load_pipeline("test", str(config_path))
        
        self.assertIsNone(result)
    
    def test_load_pipeline_with_invalid_yaml(self):
        """Test loading pipeline with invalid YAML"""
        config_path = self.workspace_dir / "invalid.yaml"
        config_path.write_text("invalid: yaml: content:")
        
        with patch('yaml.safe_load', side_effect=Exception("Invalid YAML")):
            with patch('src.cursus.validation.runtime.jupyter.notebook_interface.display'):
                with patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML'):
                    result = self.interface.load_pipeline("test", str(config_path))
        
        self.assertIsNone(result)
    
    def test_execute_step_test_with_empty_params(self):
        """Test step execution with empty parameters"""
        with patch.object(self.interface, 'script_executor') as mock_executor:
            mock_result = Mock()
            mock_result.model_dump.return_value = {'success': True}
            mock_executor.test_script_isolation.return_value = mock_result
            
            result = self.interface._execute_step_test("test_step", "synthetic", {})
            
            self.assertTrue(result['success'])
    
    def test_display_pipeline_summary_with_many_steps(self):
        """Test pipeline summary display with many steps"""
        steps = [f"step_{i}" for i in range(15)]  # More than 10 steps
        config = {'steps': steps}
        
        with patch('src.cursus.validation.runtime.jupyter.notebook_interface.display'):
            with patch('src.cursus.validation.runtime.jupyter.notebook_interface.HTML') as mock_html:
                self.interface._display_pipeline_summary(config)
        
        html_content = mock_html.call_args[0][0]
        self.assertIn("15", html_content)  # Total number of steps
        self.assertIn("... and 5 more", html_content)  # Truncation message


if __name__ == '__main__':
    unittest.main()
