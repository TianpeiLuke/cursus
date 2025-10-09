---
tags:
  - design
  - ui
  - wizard
  - nested
  - cradle_ui
  - config_ui
keywords:
  - nested wizard
  - multi-step wizard
  - cradle data loading
  - navigation control
  - state management
topics:
  - nested wizard pattern design
  - cradle ui integration
  - multi-step navigation control
language: python, javascript
date of note: 2025-10-08
---

# Nested Wizard Pattern Design

## Problem Statement

CradleDataLoadConfig requires a 4-step inner wizard within the main 7-step pipeline configuration wizard. This creates a "wizard within a wizard" scenario that needs clean navigation control and state management.

**Challenge**: When user reaches the CradleDataLoadConfig step, they need to complete a 4-step inner wizard before the main wizard's "Next" button should be enabled.

## Design Solution: Simple State-Based Approach

### Core Concept

Treat the nested wizard as a **stateful component** with three simple states:
1. **COLLAPSED**: Show "Configure" button, Next disabled
2. **ACTIVE**: Inner wizard running, Next disabled  
3. **COMPLETED**: Configuration done, Next enabled

### User Experience Flow

```
Step 3: CradleDataLoadConfig - COLLAPSED State
┌─────────────────────────────────────────────────────────────┐
│ 🎯 Configuration Workflow - Step 3 of 7                    │
│ 📋 CradleDataLoadingConfig                                 │
│                                                             │
│ [🎛️ Configure Cradle Data Loading (4 steps)]              │
│                                                             │
│ [← Previous] [Next → DISABLED] [Complete Workflow]        │
└─────────────────────────────────────────────────────────────┘

↓ Click Configure ↓

Step 3: CradleDataLoadConfig - ACTIVE State  
┌─────────────────────────────────────────────────────────────┐
│ 🎯 Configuration Workflow - Step 3 of 7                    │
│ 📋 CradleDataLoadingConfig                                 │
│                                                             │
│ ┌─ Cradle Wizard: ●●○○ (2/4) ─────────────────────────────┐ │
│ │ 2️⃣ Transform Specification                             │ │
│ │ [Cradle UI embedded here]                               │ │
│ │ [← Back] [Continue →]                                   │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ [← Previous DISABLED] [Next → DISABLED] [Complete DISABLED]│
└─────────────────────────────────────────────────────────────┘

↓ Complete 4 steps ↓

Step 3: CradleDataLoadConfig - COMPLETED State
┌─────────────────────────────────────────────────────────────┐
│ 🎯 Configuration Workflow - Step 3 of 7                    │
│ 📋 CradleDataLoadingConfig                                 │
│                                                             │
│ ✅ Cradle Configuration Complete                           │
│ Data Sources: 2 configured, Transform: SQL, Output: PARQUET │
│                                                             │
│ [← Previous] [Next → ENABLED] [Complete Workflow]         │
└─────────────────────────────────────────────────────────────┘
```

## Technical Implementation

### Simple State Management

```python
class CradleStepState:
    COLLAPSED = "collapsed"
    ACTIVE = "active" 
    COMPLETED = "completed"

class CradleStepWidget:
    def __init__(self, ...):
        self.state = CradleStepState.COLLAPSED
        self.config_result = None
        self.navigation_callback = None  # Callback to parent wizard
    
    def configure_clicked(self):
        """User clicks Configure button."""
        self.state = CradleStepState.ACTIVE
        self._notify_parent('disable_navigation')
        self._show_inner_wizard()
    
    def inner_wizard_complete(self, config):
        """Inner wizard finished."""
        self.config_result = config
        self.state = CradleStepState.COMPLETED
        self._notify_parent('enable_navigation')
        self._show_completion_summary()
    
    def _notify_parent(self, action):
        if self.navigation_callback:
            self.navigation_callback(action)
```

### Parent Wizard Integration

```python
class MultiStepWizard:
    def _create_step_widget(self, step):
        widget = UniversalConfigWidget(step)
        
        # If specialized widget, set up navigation callback
        if step.get('is_specialized'):
            widget.set_navigation_callback(self._handle_navigation_control)
        
        return widget
    
    def _handle_navigation_control(self, action):
        """Simple navigation control."""
        if action == 'disable_navigation':
            self.next_button.disabled = True
            self.prev_button.disabled = True
        elif action == 'enable_navigation':
            self.next_button.disabled = False
            self.prev_button.disabled = False
```

### CradleConfigWidget Integration

```python
class CradleConfigWidget:
    def __init__(self, ..., embedded_mode=False):
        self.embedded_mode = embedded_mode
        self.completion_callback = None
    
    def set_completion_callback(self, callback):
        """Set callback for when 4-step wizard completes."""
        self.completion_callback = callback
    
    def _on_finish_clicked(self):
        """Handle finish - different behavior for embedded mode."""
        if self.embedded_mode:
            # Create config object and notify parent
            config = self._create_config_instance()
            if self.completion_callback:
                self.completion_callback(config)
        else:
            # Original behavior - save JSON file
            self._save_json_file()
```

## Key Design Principles

### 1. **Simple State Machine**
- Only 3 states: COLLAPSED → ACTIVE → COMPLETED
- Clear transitions with obvious triggers
- No complex state combinations

### 2. **Callback-Based Communication**
- Parent wizard provides navigation callback to child
- Child widget calls back when state changes
- Minimal coupling between components

### 3. **Preserve Existing UI**
- CradleConfigWidget keeps existing 4-step wizard
- Just add `embedded_mode` flag to change save behavior
- No changes to inner wizard navigation

### 4. **Clear Visual Hierarchy**
- Outer wizard progress always visible
- Inner wizard progress shown when active
- Completion state shows summary, not details

## Implementation Priority

### Phase 1: Core Pattern (Week 1)
1. Add state management to specialized widgets
2. Implement navigation callback system
3. Update CradleConfigWidget with embedded_mode
4. Test basic state transitions

### Phase 2: Polish (Week 2)  
1. Enhance visual states (collapsed/active/completed)
2. Add progress indicators for inner wizard
3. Improve completion summary display
4. Add error handling for incomplete states

## Benefits of This Approach

- **Simple**: Only 3 states, clear transitions
- **Reusable**: Pattern works for any nested wizard
- **Non-invasive**: Minimal changes to existing code
- **Maintainable**: Clear separation of concerns
- **Extensible**: Easy to add more nested wizards later

## Anti-Patterns Avoided

- ❌ Complex state machines with many states
- ❌ Tight coupling between parent and child wizards  
- ❌ Merging navigation systems
- ❌ Over-engineering with event systems
- ❌ Breaking existing cradle UI functionality

This design keeps it simple while solving the core UX challenge of nested wizard navigation control.
