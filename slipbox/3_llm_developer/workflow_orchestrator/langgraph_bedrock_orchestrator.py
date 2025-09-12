"""
LangGraph-based Workflow Orchestrator with AWS Bedrock

This implementation uses LangGraph for state management and workflow orchestration
while calling AWS Bedrock (Claude 4.0) for intelligent decision-making.
"""

import json
import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    raise ImportError("Boto3 package required. Install with: pip install boto3")

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.memory import MemorySaver

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    raise ImportError("LangGraph package required. Install with: pip install langgraph")

logger = logging.getLogger(__name__)


class WorkflowPhase(Enum):
    """Workflow phases for the 7-step agentic process"""

    PHASE_1_PLANNING = "phase_1_planning"
    PHASE_2_IMPLEMENTATION = "phase_2_implementation"
    TRANSITION = "transition"
    COMPLETE = "complete"


class AgentType(Enum):
    """Available agent types in the system"""

    PLANNER = "planner"
    VALIDATOR = "validator"
    PROGRAMMER = "programmer"
    WORKFLOW_ORCHESTRATOR = "workflow_orchestrator"


class WorkflowState(TypedDict):
    """LangGraph state for workflow orchestration"""

    current_step: int
    current_phase: str
    active_agents: List[str]
    completed_steps: List[int]
    validation_scores: Dict[str, float]
    context_data: Dict[str, Any]
    convergence_status: Dict[str, float]
    error_log: List[str]
    decision_history: List[Dict[str, Any]]
    usage_stats: Dict[str, Any]
    messages: Annotated[List[Dict[str, Any]], add_messages]
    next_action: Optional[str]
    workflow_completed: bool


@dataclass
class BedrockResponse:
    """Response from AWS Bedrock"""

    content: str
    usage: Dict[str, int]
    model: str
    latency: float
    success: bool
    error: Optional[str] = None


class LangGraphBedrockOrchestrator:
    """
    LangGraph-based Workflow Orchestrator using AWS Bedrock with Claude 4.0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the LangGraph Bedrock orchestrator"""
        self.config = self._load_config(config)
        self.bedrock_client = self._initialize_bedrock_client()
        self.prompt_templates = self._load_prompt_templates()

        # Initialize LangGraph
        self.graph = self._build_workflow_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)

    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load orchestrator configuration"""
        default_config = {
            "aws_region": "us-east-1",
            "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",  # Claude 4.0 equivalent
            "temperature": 0.3,
            "max_tokens": 4000,
            "convergence_thresholds": {
                "alignment": 9.0,
                "standardization": 8.0,
                "compatibility": 8.0,
            },
            "max_iterations": 50,
            "confidence_threshold": 0.7,
            "templates_path": "slipbox/3_llm_developer/developer_prompt_templates/",
            "orchestrator_template": "slipbox/3_llm_developer/workflow_orchestrator/workflow_orchestrator_prompt_template.md",
        }

        if config:
            default_config.update(config)

        return default_config

    def _initialize_bedrock_client(self):
        """Initialize AWS Bedrock client"""
        try:
            session = boto3.Session(region_name=self.config["aws_region"])
            return session.client("bedrock-runtime")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load all available prompt templates"""
        templates = {}

        # Load orchestrator template
        from pathlib import Path

        orchestrator_path = Path(self.config["orchestrator_template"])
        if orchestrator_path.exists():
            with open(orchestrator_path, "r") as f:
                templates["workflow_orchestrator"] = f.read()

        # Load other templates from templates directory
        templates_dir = Path(self.config["templates_path"])
        if templates_dir.exists():
            for template_file in templates_dir.glob("*.md"):
                template_name = template_file.stem
                with open(template_file, "r") as f:
                    templates[template_name] = f.read()

        return templates

    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph"""

        # Create the state graph
        workflow = StateGraph(WorkflowState)

        # Add nodes for each workflow component
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("programmer", self._programmer_node)
        workflow.add_node("phase_transition", self._phase_transition_node)
        workflow.add_node("completion_check", self._completion_check_node)

        # Set entry point
        workflow.set_entry_point("orchestrator")

        # Add conditional edges based on orchestrator decisions
        workflow.add_conditional_edges(
            "orchestrator",
            self._route_decision,
            {
                "planner": "planner",
                "validator": "validator",
                "programmer": "programmer",
                "phase_transition": "phase_transition",
                "complete": END,
            },
        )

        # Add edges back to orchestrator after agent execution
        workflow.add_edge("planner", "completion_check")
        workflow.add_edge("validator", "completion_check")
        workflow.add_edge("programmer", "completion_check")
        workflow.add_edge("phase_transition", "completion_check")

        # Add conditional edges from completion check
        workflow.add_conditional_edges(
            "completion_check",
            self._check_completion,
            {"continue": "orchestrator", "complete": END},
        )

        return workflow

    async def _invoke_claude(self, prompt: str) -> BedrockResponse:
        """Invoke Claude 4.0 via AWS Bedrock"""
        start_time = time.time()

        try:
            # Prepare Claude request body
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config["max_tokens"],
                "temperature": self.config["temperature"],
                "system": "You are a Workflow Orchestrator AI that makes intelligent decisions about workflow progression, agent selection, and automation strategies. Always respond with valid JSON.",
                "messages": [{"role": "user", "content": prompt}],
            }

            # Make the request
            response = self.bedrock_client.invoke_model(
                modelId=self.config["model_id"],
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

            # Parse response
            response_body = json.loads(response["body"].read())
            content = response_body["content"][0]["text"]

            latency = time.time() - start_time

            # Track usage
            usage = {
                "prompt_tokens": response_body["usage"]["input_tokens"],
                "completion_tokens": response_body["usage"]["output_tokens"],
                "total_tokens": response_body["usage"]["input_tokens"]
                + response_body["usage"]["output_tokens"],
            }

            return BedrockResponse(
                content=content,
                usage=usage,
                model=self.config["model_id"],
                latency=latency,
                success=True,
            )

        except ClientError as e:
            latency = time.time() - start_time
            error_msg = f"Bedrock API error: {e.response['Error']['Message']}"
            logger.error(error_msg)

            return BedrockResponse(
                content="",
                usage={},
                model=self.config["model_id"],
                latency=latency,
                success=False,
                error=error_msg,
            )
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)

            return BedrockResponse(
                content="",
                usage={},
                model=self.config["model_id"],
                latency=latency,
                success=False,
                error=error_msg,
            )

    def _track_usage(
        self, state: WorkflowState, usage: Dict[str, int], latency: float
    ) -> None:
        """Track usage statistics in state"""
        if "usage_stats" not in state:
            state["usage_stats"] = {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "average_latency": 0.0,
            }

        stats = state["usage_stats"]
        stats["total_requests"] += 1
        stats["total_tokens"] += usage.get("total_tokens", 0)

        # Update average latency
        total_latency = (
            stats["average_latency"] * (stats["total_requests"] - 1) + latency
        )
        stats["average_latency"] = total_latency / stats["total_requests"]

        # Estimate cost (Claude 3.5 Sonnet pricing on Bedrock)
        cost_per_input_token = 0.000003  # $3 per 1M input tokens
        cost_per_output_token = 0.000015  # $15 per 1M output tokens

        request_cost = (
            usage.get("prompt_tokens", 0) * cost_per_input_token
            + usage.get("completion_tokens", 0) * cost_per_output_token
        )
        stats["total_cost"] += request_cost

    async def _orchestrator_node(self, state: WorkflowState) -> WorkflowState:
        """Orchestrator node - makes decisions using Claude 4.0"""
        logger.info(
            f"Orchestrator node: Step {state['current_step']}, Phase {state['current_phase']}"
        )

        # Prepare orchestrator prompt
        prompt = self._prepare_orchestrator_prompt(state)

        # Enhance prompt with JSON schema
        enhanced_prompt = self._enhance_prompt_with_schema(prompt)

        # Invoke Claude
        response = await self._invoke_claude(enhanced_prompt)

        # Track usage
        self._track_usage(state, response.usage, response.latency)

        if not response.success:
            logger.error(f"Claude invocation failed: {response.error}")
            state["error_log"].append(f"Claude invocation failed: {response.error}")
            # Use fallback decision
            decision = self._create_fallback_decision(state)
        else:
            # Parse Claude's decision
            try:
                decision_data = json.loads(response.content)
                if self._validate_decision_structure(decision_data):
                    decision = decision_data
                    logger.info(
                        f"Claude decision: {decision['action_type']} -> {decision['target_agent']}"
                    )
                else:
                    logger.warning(
                        "Invalid decision structure from Claude, using fallback"
                    )
                    decision = self._create_fallback_decision(state)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Claude response as JSON: {e}")
                decision = self._create_fallback_decision(state)

        # Add decision to history
        state["decision_history"].append(decision)

        # Set next action for routing
        state["next_action"] = decision["target_agent"]

        # Add message to conversation
        state["messages"].append(
            {
                "role": "orchestrator",
                "content": f"Decision: {decision['action_type']} -> {decision['target_agent']}",
                "reasoning": decision["reasoning"],
                "confidence": decision["confidence"],
            }
        )

        return state

    def _prepare_orchestrator_prompt(self, state: WorkflowState) -> str:
        """Prepare the orchestrator prompt with current state"""
        base_template = self.prompt_templates.get("workflow_orchestrator", "")

        # Create context from state
        context = {
            "current_step": state["current_step"],
            "current_phase": state["current_phase"],
            "completed_steps": state["completed_steps"],
            "convergence_status": state["convergence_status"],
            "context_data": state["context_data"],
            "decision_history": state["decision_history"][-5:],  # Last 5 decisions
            "available_templates": list(self.prompt_templates.keys()),
            "convergence_thresholds": self.config["convergence_thresholds"],
        }

        # Format template with context
        try:
            formatted_prompt = base_template.format(**context)
            return formatted_prompt
        except KeyError as e:
            logger.warning(f"Template formatting error: {e}, using base template")
            return base_template + f"\n\nContext: {json.dumps(context, indent=2)}"

    def _enhance_prompt_with_schema(self, base_prompt: str) -> str:
        """Enhance prompt with JSON schema requirements"""
        schema_requirement = """

CRITICAL: You must respond with a valid JSON object that follows this exact structure:

{
    "action_type": "string - type of action to take",
    "target_agent": "string - planner|validator|programmer|workflow_orchestrator", 
    "prompt_template": "string - name of prompt template to use",
    "parameters": {
        "key": "value - parameters for the action"
    },
    "reasoning": "string - explanation of why this decision was made",
    "confidence": "float - confidence score between 0.0 and 1.0",
    "next_step": "integer or null - next workflow step number"
}

Do not include any text before or after the JSON object. Only return valid JSON.
"""

        return base_prompt + schema_requirement

    def _validate_decision_structure(self, decision: Dict[str, Any]) -> bool:
        """Validate that the decision has the required structure"""
        required_fields = [
            "action_type",
            "target_agent",
            "prompt_template",
            "parameters",
            "reasoning",
            "confidence",
        ]

        # Check required fields exist
        if not all(field in decision for field in required_fields):
            return False

        # Validate target_agent value
        valid_agents = ["planner", "validator", "programmer", "workflow_orchestrator"]
        if decision["target_agent"] not in valid_agents:
            return False

        # Validate confidence is a number between 0 and 1
        try:
            confidence = float(decision["confidence"])
            if not (0.0 <= confidence <= 1.0):
                return False
        except (ValueError, TypeError):
            return False

        return True

    def _create_fallback_decision(self, state: WorkflowState) -> Dict[str, Any]:
        """Create a fallback decision when Claude fails"""
        current_step = state["current_step"]
        current_phase = state["current_phase"]

        # Simple rule-based fallback logic
        if current_phase == "phase_1_planning":
            if current_step == 1:
                action_type = "initiate_planning"
                target_agent = "planner"
                template = "step1_specification_analysis_prompt_template"
            elif current_step == 2:
                action_type = "validate_specifications"
                target_agent = "validator"
                template = "step2_contract_validation_prompt_template"
            elif current_step == 3:
                action_type = "validate_dependencies"
                target_agent = "validator"
                template = "step3_dependency_validation_prompt_template"
            else:
                action_type = "transition_to_implementation"
                target_agent = "workflow_orchestrator"
                template = "workflow_orchestrator"
        else:
            # Phase 2 implementation
            if current_step == 4:
                action_type = "implement_builders"
                target_agent = "programmer"
                template = "step4_builder_implementation_prompt_template"
            elif current_step == 5:
                action_type = "implement_configurations"
                target_agent = "programmer"
                template = "step5_configuration_implementation_prompt_template"
            elif current_step == 6:
                action_type = "validate_implementation"
                target_agent = "validator"
                template = "step6_implementation_validation_prompt_template"
            elif current_step == 7:
                action_type = "final_integration"
                target_agent = "programmer"
                template = "step7_integration_prompt_template"
            else:
                action_type = "analyze_current_state"
                target_agent = "workflow_orchestrator"
                template = "workflow_orchestrator"

        return {
            "action_type": action_type,
            "target_agent": target_agent,
            "prompt_template": template,
            "parameters": {"fallback": True},
            "reasoning": "Claude decision failed, using rule-based fallback",
            "confidence": 0.5,
            "next_step": current_step + 1 if current_step < 7 else None,
        }

    async def _planner_node(self, state: WorkflowState) -> WorkflowState:
        """Planner agent node"""
        logger.info("Executing Planner agent")

        # Simulate planner execution
        result = {
            "agent": "planner",
            "action": "planning_completed",
            "output": "Planning analysis completed successfully",
            "validation_score": 8.5,
        }

        # Add to messages
        state["messages"].append(
            {
                "role": "planner",
                "content": f"Planner completed: {result['output']}",
                "validation_score": result["validation_score"],
            }
        )

        return state

    async def _validator_node(self, state: WorkflowState) -> WorkflowState:
        """Validator agent node"""
        logger.info("Executing Validator agent")

        # Simulate validator execution
        validation_results = {
            "alignment": 9.2,
            "standardization": 8.8,
            "compatibility": 8.5,
        }

        # Update convergence status
        state["convergence_status"].update(validation_results)

        # Add to messages
        state["messages"].append(
            {
                "role": "validator",
                "content": "Validation completed",
                "validation_results": validation_results,
                "issues_found": [],
                "recommendations": ["Continue to next step"],
            }
        )

        return state

    async def _programmer_node(self, state: WorkflowState) -> WorkflowState:
        """Programmer agent node"""
        logger.info("Executing Programmer agent")

        # Simulate programmer execution
        result = {
            "agent": "programmer",
            "files_modified": [f"src/cursus/step_{state['current_step']}.py"],
            "tests_created": [f"test/step_{state['current_step']}_test.py"],
            "implementation_score": 8.7,
        }

        # Add to messages
        state["messages"].append(
            {
                "role": "programmer",
                "content": f"Implementation completed for step {state['current_step']}",
                "files_modified": result["files_modified"],
                "implementation_score": result["implementation_score"],
            }
        )

        return state

    async def _phase_transition_node(self, state: WorkflowState) -> WorkflowState:
        """Phase transition node"""
        logger.info("Executing phase transition")

        if state["current_phase"] == "phase_1_planning":
            state["current_phase"] = "phase_2_implementation"
            state["current_step"] = 4
            logger.info("Transitioned to Phase 2: Implementation")

        # Add to messages
        state["messages"].append(
            {
                "role": "orchestrator",
                "content": f"Phase transition completed: {state['current_phase']}",
            }
        )

        return state

    async def _completion_check_node(self, state: WorkflowState) -> WorkflowState:
        """Check if workflow step is completed and update state"""

        # Mark current step as completed
        if state["current_step"] not in state["completed_steps"]:
            state["completed_steps"].append(state["current_step"])

        # Advance to next step if not at end
        if state["current_step"] < 7:
            state["current_step"] += 1
        else:
            state["workflow_completed"] = True

        return state

    def _route_decision(self, state: WorkflowState) -> str:
        """Route based on orchestrator decision"""
        next_action = state.get("next_action", "complete")

        if next_action == "workflow_orchestrator":
            return "phase_transition"
        elif state["workflow_completed"]:
            return "complete"
        else:
            return next_action

    def _check_completion(self, state: WorkflowState) -> str:
        """Check if workflow is completed"""
        if state["workflow_completed"] or state["current_step"] > 7:
            return "complete"
        else:
            return "continue"

    def _initialize_state(self, initial_context: Dict[str, Any]) -> WorkflowState:
        """Initialize the workflow state"""
        return WorkflowState(
            current_step=1,
            current_phase="phase_1_planning",
            active_agents=[],
            completed_steps=[],
            validation_scores={},
            context_data=initial_context,
            convergence_status={
                "alignment": 0.0,
                "standardization": 0.0,
                "compatibility": 0.0,
            },
            error_log=[],
            decision_history=[],
            usage_stats={
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "average_latency": 0.0,
            },
            messages=[],
            next_action=None,
            workflow_completed=False,
        )

    async def run_workflow(
        self, initial_context: Dict[str, Any], thread_id: str = "default"
    ) -> Dict[str, Any]:
        """Run the complete workflow using LangGraph"""
        logger.info("Starting LangGraph workflow with Claude 4.0")

        # Initialize state
        initial_state = self._initialize_state(initial_context)

        # Configure thread
        config = {"configurable": {"thread_id": thread_id}}

        # Run the workflow
        final_state = None
        step_count = 0

        async for state in self.app.astream(initial_state, config):
            step_count += 1
            logger.info(f"Workflow step {step_count}: {list(state.keys())}")
            final_state = state

            # Safety check to prevent infinite loops
            if step_count > self.config["max_iterations"]:
                logger.warning("Maximum iterations reached, stopping workflow")
                break

        # Extract final state from the last node
        if final_state:
            final_workflow_state = list(final_state.values())[0]
        else:
            final_workflow_state = initial_state

        return {
            "workflow_completed": final_workflow_state.get("workflow_completed", False),
            "final_state": final_workflow_state,
            "total_steps": step_count,
            "usage_summary": final_workflow_state.get("usage_stats", {}),
            "messages": final_workflow_state.get("messages", []),
        }

    async def run_single_step(
        self, context: Dict[str, Any], thread_id: str = "default"
    ) -> Dict[str, Any]:
        """Run a single workflow step"""
        logger.info("Running single workflow step with LangGraph")

        # Get current state or initialize
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # Get current state from memory
            current_state = await self.app.aget_state(config)
            if current_state.values:
                state = current_state.values
            else:
                state = self._initialize_state(context)
        except:
            # Initialize if no previous state
            state = self._initialize_state(context)

        # Run one step
        result = await self.app.ainvoke(state, config)

        return {
            "success": True,
            "state": result,
            "usage_stats": result.get("usage_stats", {}),
        }

    def get_workflow_status(self, thread_id: str = "default") -> Dict[str, Any]:
        """Get current workflow status"""
        config = {"configurable": {"thread_id": thread_id}}

        try:
            current_state = self.app.get_state(config)
            if current_state.values:
                state = current_state.values
                return {
                    "current_step": state.get("current_step", 1),
                    "current_phase": state.get("current_phase", "phase_1_planning"),
                    "progress_percentage": (len(state.get("completed_steps", [])) / 7)
                    * 100,
                    "convergence_status": state.get("convergence_status", {}),
                    "workflow_completed": state.get("workflow_completed", False),
                    "usage_stats": state.get("usage_stats", {}),
                    "recent_messages": state.get("messages", [])[-3:],
                }
        except:
            pass

        return {
            "current_step": 1,
            "current_phase": "phase_1_planning",
            "progress_percentage": 0,
            "convergence_status": {},
            "workflow_completed": False,
            "usage_stats": {},
            "recent_messages": [],
        }


# Example usage
async def example_langgraph_orchestration():
    """Example of using LangGraph orchestration with Claude 4.0"""

    # Configuration
    config = {
        "aws_region": "us-east-1",
        "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",  # Claude 4.0 equivalent
        "temperature": 0.3,
        "max_tokens": 4000,
        "max_iterations": 20,
    }

    # Initialize orchestrator
    orchestrator = LangGraphBedrockOrchestrator(config)

    # Example context
    context = {
        "task_type": "new_pipeline_step",
        "step_name": "data_preprocessing",
        "requirements": ["Handle missing values", "Feature scaling", "Data validation"],
        "target_framework": "scikit-learn",
    }

    # Run complete workflow
    result = await orchestrator.run_workflow(context, thread_id="example_workflow")

    print("LangGraph Workflow Result:")
    print(
        json.dumps(
            {
                "completed": result["workflow_completed"],
                "total_steps": result["total_steps"],
                "usage": result["usage_summary"],
            },
            indent=2,
        )
    )

    # Get final status
    status = orchestrator.get_workflow_status("example_workflow")
    print("\nFinal Status:")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    # Run example
    asyncio.run(example_langgraph_orchestration())
