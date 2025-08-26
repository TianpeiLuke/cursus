"""
Unit tests for debugger.py module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

import pytest

# Import the module under test
from src.cursus.validation.runtime.jupyter.debugger import (
    InteractiveDebugger,
    DebugSession,
    BreakpointManager,
    JUPYTER_AVAILABLE
)


class TestDebugSession(unittest.TestCase):
    """Test cases for DebugSession model"""
    
    def test_debug_session_creation(self):
        """Test DebugSession creation with required fields"""
        session = DebugSession(
            session_id="debug_123",
            pipeline_name="test_pipeline"
        )
        
        self.assertEqual(session.session_id, "debug_123")
        self.assertEqual(session.pipeline_name, "test_pipeline")
        self.assertIsNone(session.current_step)
        self.assertEqual(session.breakpoints, [])
        self.assertEqual(session.variables, {})
        self.assertEqual(session.call_stack, [])
        self.assertEqual(session.execution_history, [])
        self.assertIsInstance(session.created_at, datetime)
    
    def test_debug_session_with_optional_fields(self):
        """Test DebugSession creation with optional fields"""
        breakpoints = ["step1", "step2"]
        variables = {"var1": "value1"}
        call_stack = [{"function": "test_func", "line": 10}]
        execution_history = [{"step": "step1", "result": "success"}]
        
        session = DebugSession(
            session_id="debug_123",
            pipeline_name="test_pipeline",
            current_step="step1",
            breakpoints=breakpoints,
            variables=variables,
            call_stack=call_stack,
            execution_history=execution_history
        )
        
        self.assertEqual(session.current_step, "step1")
        self.assertEqual(session.breakpoints, breakpoints)
        self.assertEqual(session.variables, variables)
        self.assertEqual(session.call_stack, call_stack)
        self.assertEqual(session.execution_history, execution_history)


class TestBreakpointManager(unittest.TestCase):
    """Test cases for BreakpointManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bp_manager = BreakpointManager()
    
    def test_initialization(self):
        """Test BreakpointManager initialization"""
        self.assertIsInstance(self.bp_manager.breakpoints, dict)
        self.assertIsInstance(self.bp_manager.active_breakpoints, list)
        self.assertEqual(len(self.bp_manager.breakpoints), 0)
        self.assertEqual(len(self.bp_manager.active_breakpoints), 0)
    
    def test_add_breakpoint_simple(self):
        """Test adding a simple breakpoint"""
        self.bp_manager.add_breakpoint("step1")
        
        self.assertIn("step1", self.bp_manager.breakpoints)
        self.assertIn("step1", self.bp_manager.active_breakpoints)
        
        bp = self.bp_manager.breakpoints["step1"]
        self.assertIsNone(bp["condition"])
        self.assertEqual(bp["hit_count"], 0)
        self.assertTrue(bp["enabled"])
        self.assertIsInstance(bp["created_at"], datetime)
    
    def test_add_breakpoint_with_condition(self):
        """Test adding a breakpoint with condition"""
        condition = "data_size > 1000"
        self.bp_manager.add_breakpoint("step1", condition)
        
        bp = self.bp_manager.breakpoints["step1"]
        self.assertEqual(bp["condition"], condition)
    
    def test_add_duplicate_breakpoint(self):
        """Test adding duplicate breakpoint (should update existing)"""
        self.bp_manager.add_breakpoint("step1")
        original_created_at = self.bp_manager.breakpoints["step1"]["created_at"]
        
        # Add same breakpoint again
        self.bp_manager.add_breakpoint("step1", "new_condition")
        
        # Should have updated the existing breakpoint
        self.assertEqual(len(self.bp_manager.breakpoints), 1)
        self.assertEqual(len(self.bp_manager.active_breakpoints), 1)
        self.assertEqual(self.bp_manager.breakpoints["step1"]["condition"], "new_condition")
    
    def test_remove_breakpoint(self):
        """Test removing a breakpoint"""
        self.bp_manager.add_breakpoint("step1")
        self.bp_manager.add_breakpoint("step2")
        
        self.bp_manager.remove_breakpoint("step1")
        
        self.assertNotIn("step1", self.bp_manager.breakpoints)
        self.assertNotIn("step1", self.bp_manager.active_breakpoints)
        self.assertIn("step2", self.bp_manager.breakpoints)
        self.assertIn("step2", self.bp_manager.active_breakpoints)
    
    def test_remove_nonexistent_breakpoint(self):
        """Test removing a breakpoint that doesn't exist"""
        # Should not raise an exception
        self.bp_manager.remove_breakpoint("nonexistent")
        self.assertEqual(len(self.bp_manager.breakpoints), 0)
    
    def test_should_break_no_breakpoint(self):
        """Test should_break when no breakpoint exists"""
        result = self.bp_manager.should_break("step1", {})
        self.assertFalse(result)
    
    def test_should_break_simple_breakpoint(self):
        """Test should_break with simple breakpoint"""
        self.bp_manager.add_breakpoint("step1")
        
        result = self.bp_manager.should_break("step1", {})
        
        self.assertTrue(result)
        self.assertEqual(self.bp_manager.breakpoints["step1"]["hit_count"], 1)
    
    def test_should_break_disabled_breakpoint(self):
        """Test should_break with disabled breakpoint"""
        self.bp_manager.add_breakpoint("step1")
        self.bp_manager.breakpoints["step1"]["enabled"] = False

        result = self.bp_manager.should_break("step1", {})

        self.assertFalse(result)
        self.assertEqual(self.bp_manager.breakpoints["step1"]["hit_count"], 0)  # Not incremented when disabled
    
    def test_should_break_with_condition_true(self):
        """Test should_break with condition that evaluates to True"""
        self.bp_manager.add_breakpoint("step1", "data_size > 100")
        context = {"data_size": 200}
        
        result = self.bp_manager.should_break("step1", context)
        
        self.assertTrue(result)
    
    def test_should_break_with_condition_false(self):
        """Test should_break with condition that evaluates to False"""
        self.bp_manager.add_breakpoint("step1", "data_size > 100")
        context = {"data_size": 50}
        
        result = self.bp_manager.should_break("step1", context)
        
        self.assertFalse(result)
    
    def test_should_break_with_invalid_condition(self):
        """Test should_break with invalid condition (should break anyway)"""
        self.bp_manager.add_breakpoint("step1", "invalid_variable > 100")
        context = {"data_size": 50}
        
        result = self.bp_manager.should_break("step1", context)
        
        self.assertTrue(result)  # Should break when condition evaluation fails
    
    def test_list_breakpoints(self):
        """Test listing all breakpoints"""
        self.bp_manager.add_breakpoint("step1")
        self.bp_manager.add_breakpoint("step2", "condition")
        
        breakpoints = self.bp_manager.list_breakpoints()
        
        self.assertEqual(len(breakpoints), 2)
        
        step1_bp = next(bp for bp in breakpoints if bp["step_name"] == "step1")
        step2_bp = next(bp for bp in breakpoints if bp["step_name"] == "step2")
        
        self.assertIsNone(step1_bp["condition"])
        self.assertEqual(step2_bp["condition"], "condition")
        self.assertEqual(step1_bp["hit_count"], 0)
        self.assertEqual(step2_bp["hit_count"], 0)
        self.assertTrue(step1_bp["enabled"])
        self.assertTrue(step2_bp["enabled"])


class TestInteractiveDebugger(unittest.TestCase):
    """Test cases for InteractiveDebugger class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.debugger = InteractiveDebugger()
    
    def test_initialization(self):
        """Test InteractiveDebugger initialization"""
        self.assertIsNone(self.debugger.session)
        self.assertIsInstance(self.debugger.breakpoint_manager, BreakpointManager)
        self.assertFalse(self.debugger.execution_paused)
        self.assertIsInstance(self.debugger.current_context, dict)
    
    def test_initialization_without_jupyter(self):
        """Test initialization when Jupyter is not available"""
        with patch('src.cursus.validation.runtime.jupyter.debugger.JUPYTER_AVAILABLE', False):
            debugger = InteractiveDebugger()
            self.assertIsNotNone(debugger)
    
    def test_start_debug_session(self):
        """Test starting a debug session"""
        session_id = self.debugger.start_debug_session("test_pipeline")
        
        self.assertIsNotNone(self.debugger.session)
        self.assertEqual(self.debugger.session.pipeline_name, "test_pipeline")
        self.assertEqual(self.debugger.session.session_id, session_id)
        self.assertTrue(session_id.startswith("debug_"))
    
    def test_start_debug_session_with_custom_id(self):
        """Test starting a debug session with custom ID"""
        custom_id = "custom_debug_session"
        session_id = self.debugger.start_debug_session("test_pipeline", custom_id)
        
        self.assertEqual(session_id, custom_id)
        self.assertEqual(self.debugger.session.session_id, custom_id)
    
    def test_set_breakpoint(self):
        """Test setting a breakpoint"""
        with patch('builtins.print') as mock_print:
            self.debugger.set_breakpoint("step1")
        
        self.assertIn("step1", self.debugger.breakpoint_manager.active_breakpoints)
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("Breakpoint set for step: step1" in call for call in print_calls))
    
    def test_set_breakpoint_with_condition(self):
        """Test setting a breakpoint with condition"""
        condition = "data_size > 100"
        
        with patch('builtins.print') as mock_print:
            self.debugger.set_breakpoint("step1", condition)
        
        bp = self.debugger.breakpoint_manager.breakpoints["step1"]
        self.assertEqual(bp["condition"], condition)
        
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("Condition: data_size > 100" in call for call in print_calls))
    
    def test_remove_breakpoint(self):
        """Test removing a breakpoint"""
        self.debugger.set_breakpoint("step1")
        
        with patch('builtins.print') as mock_print:
            self.debugger.remove_breakpoint("step1")
        
        self.assertNotIn("step1", self.debugger.breakpoint_manager.active_breakpoints)
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("Breakpoint removed for step: step1" in call for call in print_calls))
    
    def test_list_breakpoints_empty(self):
        """Test listing breakpoints when none exist"""
        with patch('builtins.print') as mock_print:
            self.debugger.list_breakpoints()
        
        mock_print.assert_called_with("No breakpoints set")
    
    def test_list_breakpoints_with_breakpoints(self):
        """Test listing breakpoints when they exist"""
        self.debugger.set_breakpoint("step1")
        self.debugger.set_breakpoint("step2", "condition")
        
        with patch('builtins.print') as mock_print:
            self.debugger.list_breakpoints()
        
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        output_text = ' '.join(print_calls)
        
        self.assertIn("Active Breakpoints:", output_text)
        self.assertIn("step1", output_text)
        self.assertIn("step2", output_text)
        self.assertIn("condition", output_text)
    
    def test_inspect_variable_exists(self):
        """Test inspecting a variable that exists"""
        self.debugger.current_context = {"test_var": [1, 2, 3]}
        
        with patch('builtins.print') as mock_print:
            result = self.debugger.inspect_variable("test_var")
        
        self.assertEqual(result, [1, 2, 3])
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        output_text = ' '.join(print_calls)
        
        self.assertIn("Variable: test_var", output_text)
        self.assertIn("Type: list", output_text)
        self.assertIn("Length: 3", output_text)
    
    def test_inspect_variable_not_exists(self):
        """Test inspecting a variable that doesn't exist"""
        self.debugger.current_context = {"other_var": "value"}
        
        with patch('builtins.print') as mock_print:
            result = self.debugger.inspect_variable("nonexistent_var")
        
        self.assertIsNone(result)
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("not found in current context" in call for call in print_calls))
    
    def test_inspect_variable_with_attributes(self):
        """Test inspecting a variable with attributes"""
        class TestObject:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = "value2"
        
        test_obj = TestObject()
        self.debugger.current_context = {"test_obj": test_obj}
        
        with patch('builtins.print') as mock_print:
            result = self.debugger.inspect_variable("test_obj")
        
        self.assertEqual(result, test_obj)
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        output_text = ' '.join(print_calls)
        
        self.assertIn("Attributes:", output_text)
    
    def test_analyze_error(self):
        """Test error analysis"""
        error = KeyError("missing_key")
        context = {"available_key": "value"}
        
        result = self.debugger.analyze_error(error, context)
        
        self.assertEqual(result["error_type"], "KeyError")
        self.assertEqual(result["error_message"], "'missing_key'")
        self.assertIn("traceback", result)
        self.assertIn("context_variables", result)
        self.assertIn("suggestions", result)
        self.assertEqual(result["context_variables"], ["available_key"])
        
        # Check for KeyError-specific suggestion
        suggestions_text = ' '.join(result["suggestions"])
        self.assertIn("not found", suggestions_text)
    
    def test_analyze_error_attribute_error(self):
        """Test error analysis for AttributeError"""
        error = AttributeError("'str' object has no attribute 'nonexistent'")
        context = {}
        
        result = self.debugger.analyze_error(error, context)
        
        self.assertEqual(result["error_type"], "AttributeError")
        suggestions_text = ' '.join(result["suggestions"])
        self.assertIn("attribute or method", suggestions_text)
    
    def test_analyze_error_type_error(self):
        """Test error analysis for TypeError"""
        error = TypeError("unsupported operand type(s)")
        context = {}
        
        result = self.debugger.analyze_error(error, context)
        
        self.assertEqual(result["error_type"], "TypeError")
        suggestions_text = ' '.join(result["suggestions"])
        self.assertIn("argument types", suggestions_text)
    
    def test_analyze_error_value_error(self):
        """Test error analysis for ValueError"""
        error = ValueError("invalid literal")
        context = {}
        
        result = self.debugger.analyze_error(error, context)
        
        self.assertEqual(result["error_type"], "ValueError")
        suggestions_text = ' '.join(result["suggestions"])
        self.assertIn("input values", suggestions_text)
    
    def test_should_break_at_step(self):
        """Test should_break_at_step method"""
        self.debugger.start_debug_session("test_pipeline")
        self.debugger.set_breakpoint("step1")
        
        context = {"var1": "value1"}
        result = self.debugger.should_break_at_step("step1", context)
        
        self.assertTrue(result)
        self.assertEqual(self.debugger.current_context, context)
        self.assertEqual(self.debugger.session.current_step, "step1")
    
    def test_should_break_at_step_no_breakpoint(self):
        """Test should_break_at_step when no breakpoint exists"""
        self.debugger.start_debug_session("test_pipeline")
        
        context = {"var1": "value1"}
        result = self.debugger.should_break_at_step("step1", context)
        
        self.assertFalse(result)
        self.assertEqual(self.debugger.current_context, context)
        self.assertEqual(self.debugger.session.current_step, "step1")
    
    def test_wait_for_user_input(self):
        """Test wait_for_user_input method"""
        self.debugger.start_debug_session("test_pipeline")
        self.debugger.session.current_step = "step1"
        
        with patch('builtins.print') as mock_print:
            self.debugger.wait_for_user_input()
        
        self.assertTrue(self.debugger.execution_paused)
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        output_text = ' '.join(print_calls)
        
        self.assertIn("Execution paused at step: step1", output_text)
    
    def test_wait_for_user_input_without_jupyter(self):
        """Test wait_for_user_input when Jupyter is not available"""
        with patch('src.cursus.validation.runtime.jupyter.debugger.JUPYTER_AVAILABLE', False):
            debugger = InteractiveDebugger()
            debugger.start_debug_session("test_pipeline")
            
            # Should not raise an exception
            debugger.wait_for_user_input()


@unittest.skipIf(not JUPYTER_AVAILABLE, "Jupyter dependencies not available")
class TestInteractiveDebuggerWithJupyter(unittest.TestCase):
    """Test cases that require Jupyter dependencies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.debugger = InteractiveDebugger()
    
    @patch('src.cursus.validation.runtime.jupyter.debugger.widgets')
    @patch('src.cursus.validation.runtime.jupyter.debugger.display')
    def test_create_debug_interface(self, mock_display, mock_widgets):
        """Test creating debug interface widget"""
        # Mock widgets
        mock_button = Mock()
        mock_dropdown = Mock()
        mock_text = Mock()
        mock_label = Mock()
        mock_output = Mock()
        mock_vbox = Mock()
        mock_hbox = Mock()
        
        mock_widgets.Button.return_value = mock_button
        mock_widgets.Dropdown.return_value = mock_dropdown
        mock_widgets.Text.return_value = mock_text
        mock_widgets.Label.return_value = mock_label
        mock_widgets.Output.return_value = mock_output
        mock_widgets.VBox.return_value = mock_vbox
        mock_widgets.HBox.return_value = mock_hbox
        
        # Make mock_output support context manager protocol
        mock_output.__enter__ = Mock(return_value=mock_output)
        mock_output.__exit__ = Mock(return_value=None)

        result = self.debugger.create_debug_interface()
        
        self.assertIsNotNone(result)
        mock_widgets.Button.assert_called()
        mock_widgets.Text.assert_called()
        mock_widgets.Label.assert_called()
        mock_widgets.Output.assert_called()
        mock_widgets.VBox.assert_called()
        mock_widgets.HBox.assert_called()
    
    @patch('src.cursus.validation.runtime.jupyter.debugger.widgets')
    @patch('src.cursus.validation.runtime.jupyter.debugger.display')
    def test_create_error_analysis_widget(self, mock_display, mock_widgets):
        """Test creating error analysis widget"""
        error = ValueError("Test error")
        context = {"var1": "value1"}
        
        # Mock widgets
        mock_html = Mock()
        mock_textarea = Mock()
        mock_dropdown = Mock()
        mock_button = Mock()
        mock_output = Mock()
        mock_vbox = Mock()
        mock_hbox = Mock()
        
        mock_widgets.HTML.return_value = mock_html
        mock_widgets.Textarea.return_value = mock_textarea
        mock_widgets.Dropdown.return_value = mock_dropdown
        mock_widgets.Button.return_value = mock_button
        mock_widgets.Output.return_value = mock_output
        mock_widgets.VBox.return_value = mock_vbox
        mock_widgets.HBox.return_value = mock_hbox
        
        result = self.debugger.create_error_analysis_widget(error, context)
        
        self.assertIsNotNone(result)
        mock_widgets.HTML.assert_called()
        mock_widgets.Textarea.assert_called()
        mock_widgets.Dropdown.assert_called()
        mock_widgets.Button.assert_called()
        mock_widgets.Output.assert_called()
        mock_widgets.VBox.assert_called()
        mock_widgets.HBox.assert_called()
    
    @patch('src.cursus.validation.runtime.jupyter.debugger.get_ipython')
    def test_register_magic_commands(self, mock_get_ipython):
        """Test registering IPython magic commands"""
        mock_ip = Mock()
        mock_get_ipython.return_value = mock_ip
        
        debugger = InteractiveDebugger()
        
        mock_ip.register_magic_function.assert_called()
        # Should register multiple magic functions
        self.assertGreater(mock_ip.register_magic_function.call_count, 0)
    
    @patch('src.cursus.validation.runtime.jupyter.debugger.get_ipython')
    def test_register_magic_commands_no_ipython(self, mock_get_ipython):
        """Test registering magic commands when IPython is not available"""
        mock_get_ipython.return_value = None
        
        # Should not raise an exception
        debugger = InteractiveDebugger()
        self.assertIsNotNone(debugger)
    
    @patch('src.cursus.validation.runtime.jupyter.debugger.get_ipython')
    def test_register_magic_commands_exception(self, mock_get_ipython):
        """Test registering magic commands when an exception occurs"""
        mock_get_ipython.side_effect = Exception("IPython error")
        
        with patch('builtins.print') as mock_print:
            debugger = InteractiveDebugger()
        
        # Should handle the exception gracefully
        self.assertIsNotNone(debugger)
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("Could not register magic commands" in call for call in print_calls))


class TestInteractiveDebuggerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.debugger = InteractiveDebugger()
    
    def test_magic_commands_without_session(self):
        """Test magic commands when no debug session is active"""
        with patch('builtins.print') as mock_print:
            self.debugger._debug_step_magic("")
        
        # Should not raise an exception
        mock_print.assert_called()
    
    def test_inspect_variable_empty_context(self):
        """Test inspecting variable with empty context"""
        self.debugger.current_context = {}
        
        with patch('builtins.print') as mock_print:
            result = self.debugger.inspect_variable("any_var")
        
        self.assertIsNone(result)
        mock_print.assert_called()
    
    def test_should_break_at_step_no_session(self):
        """Test should_break_at_step when no session exists"""
        context = {"var1": "value1"}
        result = self.debugger.should_break_at_step("step1", context)
        
        # Should still update context even without session
        self.assertEqual(self.debugger.current_context, context)
        # Return value depends on breakpoint manager
        self.assertIsInstance(result, bool)
    
    def test_wait_for_user_input_no_session(self):
        """Test wait_for_user_input when no session exists"""
        with patch('builtins.print') as mock_print:
            self.debugger.wait_for_user_input()
        
        self.assertTrue(self.debugger.execution_paused)
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        output_text = ' '.join(print_calls)
        self.assertIn("Unknown", output_text)


if __name__ == '__main__':
    unittest.main()
