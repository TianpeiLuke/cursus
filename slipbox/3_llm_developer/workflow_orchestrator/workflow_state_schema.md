---
tags:
  - llm_developer
  - workflow_orchestrator
  - state_management
  - schema_definition
keywords:
  - workflow state
  - state schema
  - data structure
  - state management
  - orchestration
topics:
  - workflow state management
  - data schemas
  - orchestration patterns
language: json
date of note: 2025-09-05
---

# Workflow State Schema Documentation

## Overview

This document defines the comprehensive state schema used by the Workflow Orchestrator to track and manage the execution of the 7-step agentic ML pipeline development workflow. The state schema ensures consistent state management, enables workflow recovery, and provides the foundation for intelligent orchestration decisions.

## Core State Schema

### Primary Workflow State

```json
{
  "workflow_id": "string",
  "workflow_name": "string",
  "created_at": "iso_datetime",
  "updated_at": "iso_datetime",
  "status": "initialized|running|paused|completed|failed|cancelled",
  "current_phase": "phase_1|phase_2",
  "current_step": "integer (1-7)",
  "last_completed_step": "integer (0-7)",
  "total_execution_time": "integer (seconds)",
  "step_history": [
    {
      "step_number": "integer",
      "agent": "planner|validator|programmer",
      "template": "string",
      "status": "completed|failed|in_progress|skipped",
      "started_at": "iso_datetime",
      "completed_at": "iso_datetime",
      "execution_time": "integer (seconds)",
      "input_context": "object",
      "output": "object",
      "validation_scores": {
        "alignment_score": "float (0-10)",
        "standardization_score": "float (0-10)",
        "compatibility_score": "float (0-10)",
        "overall_score": "float (0-10)"
      },
      "error_info": {
        "error_type": "string",
        "error_message": "string",
        "error_details": "object",
        "recovery_attempts": "integer"
      }
    }
  ],
  "convergence_status": {
    "phase_1_converged": "boolean",
    "phase_2_converged": "boolean",
    "convergence_criteria": {
      "alignment_threshold": "float (default: 9.0)",
      "standardization_threshold": "float (default: 8.0)",
      "compatibility_threshold": "float (default: 8.0)"
    },
    "convergence_history": [
      {
        "phase": "phase_1|phase_2",
        "attempt": "integer",
        "scores": "object",
        "converged": "boolean",
        "timestamp": "iso_datetime"
      }
    ]
  },
  "human_approvals": {
    "plan_approved": "boolean",
    "plan_approval_timestamp": "iso_datetime",
    "plan_approver": "string",
    "code_approved": "boolean", 
    "code_approval_timestamp": "iso_datetime",
    "code_approver": "string",
    "final_approved": "boolean",
    "final_approval_timestamp": "iso_datetime",
    "final_approver": "string",
    "approval_history": [
      {
        "approval_type": "plan|code|final",
        "status": "approved|rejected|changes_requested",
        "approver": "string",
        "timestamp": "iso_datetime",
        "comments": "string",
        "change_requests": ["array of strings"]
      }
    ]
  },
  "workspace_context": {
    "developer_type": "shared|isolated",
    "workspace_path": "string",
    "project_context": {
      "project_name": "string",
      "project_type": "string",
      "target_environment": "string"
    },
    "file_locations": {
      "builders_path": "string",
      "configs_path": "string", 
      "specs_path": "string",
      "contracts_path": "string",
      "scripts_path": "string"
    }
  },
  "requirements": {
    "initial_requirements": "string",
    "step_type": "string",
    "documentation_locations": ["array of strings"],
    "special_requirements": ["array of strings"],
    "constraints": ["array of strings"]
  },
  "artifacts": {
    "implementation_plan": "object",
    "generated_code": {
      "builder": "string",
      "config": "string",
      "spec": "string", 
      "contract": "string",
      "script": "string"
    },
    "validation_reports": ["array of objects"],
    "revision_history": ["array of objects"]
  },
  "metrics": {
    "total_iterations": "integer",
    "phase_1_iterations": "integer",
    "phase_2_iterations": "integer",
    "validation_failures": "integer",
    "recovery_actions": "integer",
    "human_interventions": "integer"
  }
}
```

## Detailed Schema Components

### Step History Entry

```json
{
  "step_number": 1,
  "agent": "planner",
  "template": "step1_initial_planner_prompt_template.md",
  "status": "completed",
  "started_at": "2025-09-05T23:00:00Z",
  "completed_at": "2025-09-05T23:05:30Z",
  "execution_time": 330,
  "input_context": {
    "requirements": "Create a new tabular preprocessing step",
    "documentation_locations": [
      "slipbox/0_developer_guide/",
      "src/cursus/steps/"
    ],
    "workspace_context": {
      "developer_type": "shared",
      "workspace_path": "src/cursus/steps/"
    }
  },
  "output": {
    "implementation_plan": {
      "step_overview": {
        "purpose": "Tabular data preprocessing with feature engineering",
        "inputs": ["raw_tabular_data"],
        "outputs": ["preprocessed_data", "feature_metadata"],
        "position_in_pipeline": "early_preprocessing"
      },
      "components_to_create": [
        {
          "type": "script_contract",
          "file_path": "src/cursus/steps/contracts/tabular_preprocessing_contract.py",
          "description": "Define input/output paths and environment variables"
        },
        {
          "type": "step_specification", 
          "file_path": "src/cursus/steps/specs/tabular_preprocessing_spec.py",
          "description": "Define dependencies and outputs with logical names"
        }
      ],
      "integration_strategy": {
        "upstream_steps": ["data_loading", "data_validation"],
        "downstream_steps": ["feature_selection", "model_training"]
      }
    }
  },
  "validation_scores": {
    "alignment_score": 8.5,
    "standardization_score": 8.2,
    "compatibility_score": 8.8,
    "overall_score": 8.5
  },
  "error_info": null
}
```

### Convergence Status

```json
{
  "phase_1_converged": true,
  "phase_2_converged": false,
  "convergence_criteria": {
    "alignment_threshold": 9.0,
    "standardization_threshold": 8.0,
    "compatibility_threshold": 8.0
  },
  "convergence_history": [
    {
      "phase": "phase_1",
      "attempt": 1,
      "scores": {
        "alignment_score": 8.5,
        "standardization_score": 7.8,
        "compatibility_score": 8.2
      },
      "converged": false,
      "timestamp": "2025-09-05T23:10:00Z"
    },
    {
      "phase": "phase_1", 
      "attempt": 2,
      "scores": {
        "alignment_score": 9.2,
        "standardization_score": 8.5,
        "compatibility_score": 8.8
      },
      "converged": true,
      "timestamp": "2025-09-05T23:15:30Z"
    }
  ]
}
```

### Human Approval Entry

```json
{
  "approval_type": "plan",
  "status": "approved",
  "approver": "senior_developer_001",
  "timestamp": "2025-09-05T23:20:00Z",
  "comments": "Plan looks comprehensive and well-structured. Approved for implementation.",
  "change_requests": []
}
```

### Workspace Context

```json
{
  "developer_type": "isolated",
  "workspace_path": "development/projects/project_alpha/src/cursus_dev/",
  "project_context": {
    "project_name": "project_alpha",
    "project_type": "ml_pipeline_extension",
    "target_environment": "development"
  },
  "file_locations": {
    "builders_path": "development/projects/project_alpha/src/cursus_dev/steps/builders/",
    "configs_path": "development/projects/project_alpha/src/cursus_dev/steps/configs/",
    "specs_path": "development/projects/project_alpha/src/cursus_dev/steps/specs/",
    "contracts_path": "development/projects/project_alpha/src/cursus_dev/steps/contracts/",
    "scripts_path": "development/projects/project_alpha/src/cursus_dev/steps/scripts/"
  }
}
```

## State Transitions

### Phase Transitions

```json
{
  "phase_transition": {
    "from_phase": "phase_1",
    "to_phase": "phase_2", 
    "trigger": "convergence_achieved",
    "timestamp": "2025-09-05T23:25:00Z",
    "validation_scores": {
      "alignment_score": 9.2,
      "standardization_score": 8.5,
      "compatibility_score": 8.8
    },
    "human_approval_required": true,
    "human_approval_status": "pending"
  }
}
```

### Step Transitions

```json
{
  "step_transition": {
    "from_step": 2,
    "to_step": 3,
    "trigger": "validation_failed",
    "timestamp": "2025-09-05T23:12:00Z",
    "reason": "Alignment score below threshold (8.5 < 9.0)",
    "next_action": "invoke_planner_for_revision"
  }
}
```

## Error Handling Schema

### Error Information

```json
{
  "error_info": {
    "error_type": "agent_timeout",
    "error_message": "Validator agent timed out after 300 seconds",
    "error_details": {
      "agent": "validator",
      "step": 5,
      "template": "step5a_two_level_validation_agent_prompt_template.md",
      "timeout_duration": 300,
      "partial_output": "object"
    },
    "recovery_attempts": 2,
    "recovery_strategies": [
      {
        "strategy": "retry_with_extended_timeout",
        "attempted_at": "2025-09-05T23:30:00Z",
        "result": "failed"
      },
      {
        "strategy": "retry_with_different_agent_instance",
        "attempted_at": "2025-09-05T23:35:00Z", 
        "result": "success"
      }
    ]
  }
}
```

## State Validation Rules

### Required Fields

```json
{
  "required_fields": [
    "workflow_id",
    "workflow_name", 
    "created_at",
    "status",
    "current_phase",
    "current_step",
    "workspace_context",
    "requirements"
  ],
  "conditional_requirements": {
    "if_status_running": ["step_history"],
    "if_phase_2": ["convergence_status.phase_1_converged"],
    "if_step_completed": ["step_history[].output"]
  }
}
```

### State Consistency Rules

```json
{
  "consistency_rules": [
    {
      "rule": "current_step_matches_last_history_entry",
      "description": "current_step should match the step_number of the last entry in step_history"
    },
    {
      "rule": "phase_step_alignment",
      "description": "Steps 1-3 must be in phase_1, steps 4-7 must be in phase_2"
    },
    {
      "rule": "convergence_before_transition",
      "description": "phase_1_converged must be true before transitioning to phase_2"
    },
    {
      "rule": "sequential_step_execution",
      "description": "Steps must be executed in sequential order (1,2,3,4,5,6,7)"
    }
  ]
}
```

## State Persistence

### Storage Format

```json
{
  "storage_metadata": {
    "schema_version": "1.0.0",
    "storage_format": "json",
    "compression": "gzip",
    "encryption": "aes-256",
    "backup_frequency": "every_step_completion",
    "retention_policy": "30_days"
  },
  "file_structure": {
    "primary_state": "workflow_states/{workflow_id}/state.json",
    "step_artifacts": "workflow_states/{workflow_id}/artifacts/step_{step_number}/",
    "backups": "workflow_states/{workflow_id}/backups/",
    "logs": "workflow_states/{workflow_id}/logs/"
  }
}
```

### State Recovery

```json
{
  "recovery_metadata": {
    "last_checkpoint": "iso_datetime",
    "recovery_point": "step_number",
    "state_integrity": "verified|corrupted|partial",
    "recovery_actions": [
      {
        "action": "restore_from_backup",
        "backup_timestamp": "iso_datetime",
        "success": "boolean"
      }
    ]
  }
}
```

## Usage Examples

### State Query Examples

```python
# Get current workflow status
def get_workflow_status(workflow_id: str) -> dict:
    state = load_workflow_state(workflow_id)
    return {
        "workflow_id": state["workflow_id"],
        "status": state["status"],
        "current_phase": state["current_phase"],
        "current_step": state["current_step"],
        "progress_percentage": calculate_progress(state)
    }

# Check convergence status
def check_convergence(workflow_id: str, phase: str) -> bool:
    state = load_workflow_state(workflow_id)
    if phase == "phase_1":
        return state["convergence_status"]["phase_1_converged"]
    elif phase == "phase_2":
        return state["convergence_status"]["phase_2_converged"]
    return False

# Get validation scores for latest step
def get_latest_validation_scores(workflow_id: str) -> dict:
    state = load_workflow_state(workflow_id)
    if state["step_history"]:
        latest_step = state["step_history"][-1]
        return latest_step.get("validation_scores", {})
    return {}
```

### State Update Examples

```python
# Update step completion
def complete_step(workflow_id: str, step_number: int, output: dict, scores: dict):
    state = load_workflow_state(workflow_id)
    
    # Update step history
    step_entry = {
        "step_number": step_number,
        "status": "completed",
        "completed_at": datetime.utcnow().isoformat(),
        "output": output,
        "validation_scores": scores
    }
    
    # Find and update the step in history
    for i, step in enumerate(state["step_history"]):
        if step["step_number"] == step_number:
            state["step_history"][i].update(step_entry)
            break
    
    # Update current step
    state["current_step"] = step_number
    state["last_completed_step"] = step_number
    state["updated_at"] = datetime.utcnow().isoformat()
    
    save_workflow_state(workflow_id, state)

# Record phase transition
def transition_phase(workflow_id: str, from_phase: str, to_phase: str, scores: dict):
    state = load_workflow_state(workflow_id)
    
    # Update phase
    state["current_phase"] = to_phase
    
    # Update convergence status
    if from_phase == "phase_1":
        state["convergence_status"]["phase_1_converged"] = True
    
    # Record transition in history
    transition_record = {
        "from_phase": from_phase,
        "to_phase": to_phase,
        "timestamp": datetime.utcnow().isoformat(),
        "validation_scores": scores
    }
    
    if "phase_transitions" not in state:
        state["phase_transitions"] = []
    state["phase_transitions"].append(transition_record)
    
    save_workflow_state(workflow_id, state)
```

## Integration Points

### Orchestrator Integration

The Workflow Orchestrator uses this state schema to:

1. **Make Decisions**: Analyze current state to determine next actions
2. **Track Progress**: Monitor workflow progression and convergence
3. **Handle Errors**: Record and recover from failures
4. **Coordinate Agents**: Provide context for agent invocations
5. **Manage Human Interactions**: Track approval status and requirements

### Agent Integration

Individual agents (Planner, Validator, Programmer) interact with the state to:

1. **Receive Context**: Get relevant workflow context for their tasks
2. **Report Results**: Update state with their outputs and validation scores
3. **Handle Errors**: Report failures and error conditions
4. **Track Metrics**: Contribute to workflow performance metrics

## Conclusion

This comprehensive state schema provides the foundation for robust workflow orchestration, enabling the Workflow Orchestrator to make intelligent decisions, handle errors gracefully, and maintain complete visibility into workflow execution. The schema supports both shared and isolated development workflows while maintaining consistency and enabling recovery from failures.
