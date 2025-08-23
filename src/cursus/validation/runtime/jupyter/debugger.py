"""
Interactive Debugger for Pipeline Runtime Testing

This module provides advanced debugging capabilities for pipeline testing,
including step-by-step execution, variable inspection, and error analysis.
"""

from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import traceback
import inspect
import sys
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

try:
    from IPython.display import display, HTML, Code
    import ipywidgets as widgets
    from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
    from IPython import get_ipython
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
    # Create mock classes for graceful fallback
    class widgets:
        class Output: pass
        class VBox: pass
        class HBox: pass
        class Button: pass
        class Textarea: pass
        class Dropdown: pass


class DebugSession(BaseModel):
    """Debug session state management"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    session_id: str
    pipeline_name: str
    current_step: Optional[str] = None
    breakpoints: List[str] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)
    call_stack: List[Dict[str, Any]] = Field(default_factory=list)
    execution_history: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class BreakpointManager:
    """Manages breakpoints for debugging"""
    
    def __init__(self):
        self.breakpoints: Dict[str, Dict[str, Any]] = {}
        self.active_breakpoints: List[str] = []
    
    def add_breakpoint(self, step_name: str, condition: Optional[str] = None):
        """Add a breakpoint for a specific step"""
        self.breakpoints[step_name] = {
            'condition': condition,
            'hit_count': 0,
            'enabled': True,
            'created_at': datetime.now()
        }
        if step_name not in self.active_breakpoints:
            self.active_breakpoints.append(step_name)
    
    def remove_breakpoint(self, step_name: str):
        """Remove a breakpoint"""
        if step_name in self.breakpoints:
            del self.breakpoints[step_name]
        if step_name in self.active_breakpoints:
            self.active_breakpoints.remove(step_name)
    
    def should_break(self, step_name: str, context: Dict[str, Any]) -> bool:
        """Check if execution should break at this step"""
        if step_name not in self.active_breakpoints:
            return False
        
        bp = self.breakpoints.get(step_name)
        if not bp or not bp['enabled']:
            return False
        
        # Increment hit count
        bp['hit_count'] += 1
        
        # Check condition if specified
        if bp['condition']:
            try:
                # Safely evaluate condition in context
                return eval(bp['condition'], {"__builtins__": {}}, context)
            except Exception:
                # If condition evaluation fails, break anyway
                return True
        
        return True
    
    def list_breakpoints(self) -> List[Dict[str, Any]]:
        """List all breakpoints with their status"""
        result = []
        for step_name, bp in self.breakpoints.items():
            result.append({
                'step_name': step_name,
                'condition': bp['condition'],
                'hit_count': bp['hit_count'],
                'enabled': bp['enabled'],
                'created_at': bp['created_at']
            })
        return result


class InteractiveDebugger:
    """
    Interactive debugger for pipeline runtime testing
    
    Provides step-by-step execution, variable inspection, breakpoints,
    and comprehensive error analysis capabilities.
    """
    
    def __init__(self):
        """Initialize the interactive debugger"""
        if not JUPYTER_AVAILABLE:
            print("Warning: Jupyter dependencies not available. Debugging features disabled.")
            return
        
        self.session: Optional[DebugSession] = None
        self.breakpoint_manager = BreakpointManager()
        self.execution_paused = False
        self.current_context: Dict[str, Any] = {}
        self.output_widget = widgets.Output()
        
        # Register custom magic commands if in IPython
        self._register_magic_commands()
    
    def _register_magic_commands(self):
        """Register custom IPython magic commands for debugging"""
        if not JUPYTER_AVAILABLE:
            return
        
        try:
            ip = get_ipython()
            if ip:
                # Register line magics
                ip.register_magic_function(self._debug_step_magic, 'line', 'debug_step')
                ip.register_magic_function(self._set_breakpoint_magic, 'line', 'breakpoint')
                ip.register_magic_function(self._inspect_var_magic, 'line', 'inspect')
        except Exception as e:
            print(f"Could not register magic commands: {e}")
    
    def start_debug_session(self, pipeline_name: str, session_id: Optional[str] = None) -> str:
        """Start a new debug session"""
        if not session_id:
            session_id = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session = DebugSession(
            session_id=session_id,
            pipeline_name=pipeline_name
        )
        
        print(f"Debug session started: {session_id}")
        print(f"Pipeline: {pipeline_name}")
        print("Use debug commands or the interactive interface to control execution")
        
        return session_id
    
    def create_debug_interface(self) -> Optional[widgets.VBox]:
        """Create interactive debugging interface"""
        if not JUPYTER_AVAILABLE:
            return None
        
        # Control buttons
        step_button = widgets.Button(
            description='Step',
            button_style='primary',
            tooltip='Execute next step'
        )
        
        continue_button = widgets.Button(
            description='Continue',
            button_style='success',
            tooltip='Continue execution'
        )
        
        pause_button = widgets.Button(
            description='Pause',
            button_style='warning',
            tooltip='Pause execution'
        )
        
        stop_button = widgets.Button(
            description='Stop',
            button_style='danger',
            tooltip='Stop debugging'
        )
        
        # Variable inspection
        var_dropdown = widgets.Dropdown(
            options=[],
            description='Variable:',
            disabled=False
        )
        
        inspect_button = widgets.Button(
            description='Inspect',
            button_style='info'
        )
        
        # Breakpoint management
        bp_step_text = widgets.Text(
            placeholder='Step name',
            description='Step:'
        )
        
        bp_condition_text = widgets.Text(
            placeholder='Condition (optional)',
            description='Condition:'
        )
        
        add_bp_button = widgets.Button(
            description='Add Breakpoint',
            button_style='primary'
        )
        
        remove_bp_button = widgets.Button(
            description='Remove',
            button_style='danger'
        )
        
        # Status display
        status_label = widgets.Label(value="Ready")
        current_step_label = widgets.Label(value="Current Step: None")
        
        # Output areas
        variable_output = widgets.Output()
        execution_output = widgets.Output()
        breakpoint_output = widgets.Output()
        
        def step_execution(*args):
            """Execute single step"""
            with execution_output:
                execution_output.clear_output()
                print("Stepping to next execution point...")
                self.execution_paused = False
                status_label.value = "Stepping..."
        
        def continue_execution(*args):
            """Continue execution"""
            with execution_output:
                execution_output.clear_output()
                print("Continuing execution...")
                self.execution_paused = False
                status_label.value = "Running..."
        
        def pause_execution(*args):
            """Pause execution"""
            with execution_output:
                execution_output.clear_output()
                print("Execution paused")
                self.execution_paused = True
                status_label.value = "Paused"
        
        def stop_debugging(*args):
            """Stop debugging session"""
            with execution_output:
                execution_output.clear_output()
                print("Debugging session stopped")
                self.session = None
                status_label.value = "Stopped"
        
        def inspect_variable(*args):
            """Inspect selected variable"""
            with variable_output:
                variable_output.clear_output()
                var_name = var_dropdown.value
                if var_name and var_name in self.current_context:
                    var_value = self.current_context[var_name]
                    print(f"Variable: {var_name}")
                    print(f"Type: {type(var_value).__name__}")
                    print(f"Value: {repr(var_value)}")
                    
                    # Additional inspection for complex objects
                    if hasattr(var_value, '__dict__'):
                        print(f"Attributes: {list(var_value.__dict__.keys())}")
                    
                    if hasattr(var_value, '__len__'):
                        try:
                            print(f"Length: {len(var_value)}")
                        except:
                            pass
                else:
                    print(f"Variable '{var_name}' not found in current context")
        
        def add_breakpoint(*args):
            """Add a new breakpoint"""
            with breakpoint_output:
                step_name = bp_step_text.value.strip()
                condition = bp_condition_text.value.strip() or None
                
                if step_name:
                    self.breakpoint_manager.add_breakpoint(step_name, condition)
                    print(f"Breakpoint added for step: {step_name}")
                    if condition:
                        print(f"Condition: {condition}")
                    
                    # Clear inputs
                    bp_step_text.value = ""
                    bp_condition_text.value = ""
                    
                    # Update breakpoint list
                    self._update_breakpoint_display(breakpoint_output)
                else:
                    print("Please enter a step name")
        
        def remove_breakpoint(*args):
            """Remove a breakpoint"""
            with breakpoint_output:
                step_name = bp_step_text.value.strip()
                if step_name:
                    self.breakpoint_manager.remove_breakpoint(step_name)
                    print(f"Breakpoint removed for step: {step_name}")
                    bp_step_text.value = ""
                    self._update_breakpoint_display(breakpoint_output)
                else:
                    print("Please enter a step name to remove")
        
        # Set up event handlers
        step_button.on_click(step_execution)
        continue_button.on_click(continue_execution)
        pause_button.on_click(pause_execution)
        stop_button.on_click(stop_debugging)
        inspect_button.on_click(inspect_variable)
        add_bp_button.on_click(add_breakpoint)
        remove_bp_button.on_click(remove_breakpoint)
        
        # Create layout
        control_buttons = widgets.HBox([step_button, continue_button, pause_button, stop_button])
        status_info = widgets.VBox([status_label, current_step_label])
        
        variable_section = widgets.VBox([
            widgets.HTML("<h4>Variable Inspection</h4>"),
            widgets.HBox([var_dropdown, inspect_button]),
            variable_output
        ])
        
        breakpoint_section = widgets.VBox([
            widgets.HTML("<h4>Breakpoint Management</h4>"),
            widgets.HBox([bp_step_text, bp_condition_text]),
            widgets.HBox([add_bp_button, remove_bp_button]),
            breakpoint_output
        ])
        
        execution_section = widgets.VBox([
            widgets.HTML("<h4>Execution Output</h4>"),
            execution_output
        ])
        
        # Main layout
        debug_interface = widgets.VBox([
            widgets.HTML("<h3>Interactive Debugger</h3>"),
            status_info,
            control_buttons,
            variable_section,
            breakpoint_section,
            execution_section
        ])
        
        # Initialize displays
        self._update_variable_dropdown(var_dropdown)
        self._update_breakpoint_display(breakpoint_output)
        
        return debug_interface
    
    def _update_variable_dropdown(self, dropdown: widgets.Dropdown):
        """Update variable dropdown with current context"""
        if self.current_context:
            dropdown.options = list(self.current_context.keys())
        else:
            dropdown.options = []
    
    def _update_breakpoint_display(self, output: widgets.Output):
        """Update breakpoint display"""
        with output:
            output.clear_output()
            breakpoints = self.breakpoint_manager.list_breakpoints()
            if breakpoints:
                print("Active Breakpoints:")
                for bp in breakpoints:
                    status = "✓" if bp['enabled'] else "✗"
                    condition_str = f" (condition: {bp['condition']})" if bp['condition'] else ""
                    print(f"  {status} {bp['step_name']} - hits: {bp['hit_count']}{condition_str}")
            else:
                print("No breakpoints set")
    
    def set_breakpoint(self, step_name: str, condition: Optional[str] = None):
        """Set a breakpoint for a specific step"""
        self.breakpoint_manager.add_breakpoint(step_name, condition)
        print(f"Breakpoint set for step: {step_name}")
        if condition:
            print(f"Condition: {condition}")
    
    def remove_breakpoint(self, step_name: str):
        """Remove a breakpoint"""
        self.breakpoint_manager.remove_breakpoint(step_name)
        print(f"Breakpoint removed for step: {step_name}")
    
    def list_breakpoints(self):
        """List all active breakpoints"""
        breakpoints = self.breakpoint_manager.list_breakpoints()
        if breakpoints:
            print("Active Breakpoints:")
            for bp in breakpoints:
                status = "enabled" if bp['enabled'] else "disabled"
                condition_str = f" (condition: {bp['condition']})" if bp['condition'] else ""
                print(f"  {bp['step_name']} - {status}, hits: {bp['hit_count']}{condition_str}")
        else:
            print("No breakpoints set")
    
    def inspect_variable(self, var_name: str) -> Any:
        """Inspect a variable in the current context"""
        if var_name in self.current_context:
            var_value = self.current_context[var_name]
            print(f"Variable: {var_name}")
            print(f"Type: {type(var_value).__name__}")
            print(f"Value: {repr(var_value)}")
            
            # Additional inspection for complex objects
            if hasattr(var_value, '__dict__'):
                print(f"Attributes: {list(var_value.__dict__.keys())}")
            
            if hasattr(var_value, '__len__'):
                try:
                    print(f"Length: {len(var_value)}")
                except:
                    pass
            
            return var_value
        else:
            print(f"Variable '{var_name}' not found in current context")
            return None
    
    def analyze_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze an error and provide debugging information"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context_variables': list(context.keys()),
            'suggestions': []
        }
        
        # Add specific suggestions based on error type
        if isinstance(error, KeyError):
            error_info['suggestions'].append(
                f"Key '{error}' not found. Available keys: {list(context.keys())}"
            )
        elif isinstance(error, AttributeError):
            error_info['suggestions'].append(
                "Check if the object has the expected attribute or method"
            )
        elif isinstance(error, TypeError):
            error_info['suggestions'].append(
                "Check argument types and function signatures"
            )
        elif isinstance(error, ValueError):
            error_info['suggestions'].append(
                "Check input values and their validity"
            )
        
        return error_info
    
    def create_error_analysis_widget(self, error: Exception, context: Dict[str, Any]) -> Optional[widgets.VBox]:
        """Create an interactive error analysis widget"""
        if not JUPYTER_AVAILABLE:
            return None
        
        error_info = self.analyze_error(error, context)
        
        # Error details
        error_html = f"""
        <div style="border: 2px solid #d32f2f; padding: 10px; margin: 10px 0; background-color: #ffebee;">
            <h4 style="color: #d32f2f; margin-top: 0;">Error Analysis</h4>
            <p><strong>Type:</strong> {error_info['error_type']}</p>
            <p><strong>Message:</strong> {error_info['error_message']}</p>
        </div>
        """
        
        error_display = widgets.HTML(error_html)
        
        # Traceback display
        traceback_text = widgets.Textarea(
            value=error_info['traceback'],
            description='Traceback:',
            layout=widgets.Layout(width='100%', height='200px'),
            disabled=True
        )
        
        # Suggestions
        suggestions_html = "<h4>Debugging Suggestions:</h4><ul>"
        for suggestion in error_info['suggestions']:
            suggestions_html += f"<li>{suggestion}</li>"
        suggestions_html += "</ul>"
        
        suggestions_display = widgets.HTML(suggestions_html)
        
        # Context variables
        context_dropdown = widgets.Dropdown(
            options=list(context.keys()),
            description='Inspect:',
            disabled=False
        )
        
        context_output = widgets.Output()
        
        def inspect_context_var(*args):
            """Inspect a context variable"""
            with context_output:
                context_output.clear_output()
                var_name = context_dropdown.value
                if var_name and var_name in context:
                    var_value = context[var_name]
                    print(f"Variable: {var_name}")
                    print(f"Type: {type(var_value).__name__}")
                    print(f"Value: {repr(var_value)}")
        
        inspect_button = widgets.Button(
            description='Inspect Variable',
            button_style='info'
        )
        inspect_button.on_click(inspect_context_var)
        
        # Layout
        error_widget = widgets.VBox([
            error_display,
            traceback_text,
            suggestions_display,
            widgets.HTML("<h4>Context Variables:</h4>"),
            widgets.HBox([context_dropdown, inspect_button]),
            context_output
        ])
        
        return error_widget
    
    def _debug_step_magic(self, line):
        """Magic command for stepping through execution"""
        print("Stepping to next execution point...")
        self.execution_paused = False
    
    def _set_breakpoint_magic(self, line):
        """Magic command for setting breakpoints"""
        parts = line.strip().split(' ', 1)
        step_name = parts[0]
        condition = parts[1] if len(parts) > 1 else None
        self.set_breakpoint(step_name, condition)
    
    def _inspect_var_magic(self, line):
        """Magic command for inspecting variables"""
        var_name = line.strip()
        self.inspect_variable(var_name)
    
    def should_break_at_step(self, step_name: str, context: Dict[str, Any]) -> bool:
        """Check if execution should break at this step"""
        self.current_context = context
        
        if self.session:
            self.session.current_step = step_name
        
        return self.breakpoint_manager.should_break(step_name, context)
    
    def wait_for_user_input(self):
        """Wait for user input when execution is paused"""
        if not JUPYTER_AVAILABLE:
            return
        
        print(f"Execution paused at step: {self.session.current_step if self.session else 'Unknown'}")
        print("Use the debug interface or magic commands to continue")
        
        # In a real implementation, this would integrate with the Jupyter kernel
        # to properly pause execution and wait for user commands
        self.execution_paused = True
