"""
AWS Bedrock Workflow Orchestrator

Simplified LLM-based Workflow Orchestrator using only AWS Bedrock with Claude 4.0
for intelligent decision-making in the 7-step agentic workflow process.
"""

import json
import logging
import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    raise ImportError("Boto3 package required. Install with: pip install boto3")

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


@dataclass
class WorkflowState:
    """Current state of the workflow execution"""

    current_step: int
    current_phase: WorkflowPhase
    active_agents: List[AgentType]
    completed_steps: List[int]
    validation_scores: Dict[str, float]
    context_data: Dict[str, Any]
    next_actions: List[str]
    convergence_status: Dict[str, float]
    error_log: List[str]
    metadata: Dict[str, Any]


@dataclass
class OrchestrationDecision:
    """Decision made by the orchestrator"""

    action_type: str
    target_agent: AgentType
    prompt_template: str
    parameters: Dict[str, Any]
    reasoning: str
    confidence: float
    next_step: Optional[int]


@dataclass
class BedrockResponse:
    """Response from AWS Bedrock"""

    content: str
    usage: Dict[str, int]
    model: str
    latency: float
    success: bool
    error: Optional[str] = None


class BedrockOrchestrator:
    """
    AWS Bedrock-based Workflow Orchestrator using Claude 4.0
    for intelligent workflow decision-making
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Bedrock orchestrator"""
        self.config = self._load_config(config)
        self.workflow_state = self._initialize_workflow_state()
        self.prompt_templates = self._load_prompt_templates()
        self.decision_history: List[OrchestrationDecision] = []
        self.bedrock_client = self._initialize_bedrock_client()
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_latency": 0.0,
        }

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

    def _initialize_workflow_state(self) -> WorkflowState:
        """Initialize the workflow state"""
        return WorkflowState(
            current_step=1,
            current_phase=WorkflowPhase.PHASE_1_PLANNING,
            active_agents=[],
            completed_steps=[],
            validation_scores={},
            context_data={},
            next_actions=[],
            convergence_status={
                "alignment": 0.0,
                "standardization": 0.0,
                "compatibility": 0.0,
            },
            error_log=[],
            metadata={"iteration_count": 0, "start_time": time.time()},
        )

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

    async def invoke_claude(self, prompt: str) -> BedrockResponse:
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

            self._track_usage(usage, latency)

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

    def _track_usage(self, usage: Dict[str, int], latency: float) -> None:
        """Track usage statistics"""
        self.usage_stats["total_requests"] += 1
        self.usage_stats["total_tokens"] += usage.get("total_tokens", 0)

        # Update average latency
        total_latency = (
            self.usage_stats["average_latency"]
            * (self.usage_stats["total_requests"] - 1)
            + latency
        )
        self.usage_stats["average_latency"] = (
            total_latency / self.usage_stats["total_requests"]
        )

        # Estimate cost (Claude 3.5 Sonnet pricing on Bedrock)
        cost_per_input_token = 0.000003  # $3 per 1M input tokens
        cost_per_output_token = 0.000015  # $15 per 1M output tokens

        request_cost = (
            usage.get("prompt_tokens", 0) * cost_per_input_token
            + usage.get("completion_tokens", 0) * cost_per_output_token
        )
        self.usage_stats["total_cost"] += request_cost

    async def make_orchestration_decision(
        self, context: Dict[str, Any]
    ) -> OrchestrationDecision:
        """Make an orchestration decision using Claude 4.0"""

        # Prepare the orchestrator prompt
        orchestrator_prompt = self._prepare_orchestrator_prompt(context)

        # Enhance prompt with JSON schema requirement
        enhanced_prompt = self._enhance_prompt_with_schema(orchestrator_prompt)

        # Invoke Claude
        response = await self.invoke_claude(enhanced_prompt)

        if not response.success:
            logger.error(f"Claude invocation failed: {response.error}")
            return self._create_fallback_decision(context)

        # Parse JSON response
        try:
            decision_data = json.loads(response.content)

            # Validate decision structure
            if self._validate_decision_structure(decision_data):
                decision = OrchestrationDecision(
                    action_type=decision_data["action_type"],
                    target_agent=AgentType(decision_data["target_agent"]),
                    prompt_template=decision_data["prompt_template"],
                    parameters=decision_data["parameters"],
                    reasoning=decision_data["reasoning"],
                    confidence=decision_data["confidence"],
                    next_step=decision_data.get("next_step"),
                )

                self.decision_history.append(decision)
                logger.info(
                    f"Claude orchestration decision: {decision.action_type} -> {decision.target_agent.value}"
                )
                return decision
            else:
                logger.warning("Invalid decision structure from Claude, using fallback")
                return self._create_fallback_decision(context)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.debug(f"Raw response: {response.content}")
            return self._create_fallback_decision(context)

    def _prepare_orchestrator_prompt(self, context: Dict[str, Any]) -> str:
        """Prepare the orchestrator prompt with current context"""
        base_template = self.prompt_templates.get("workflow_orchestrator", "")

        # Inject current workflow state and context
        prompt_context = {
            "current_state": asdict(self.workflow_state),
            "context": context,
            "available_templates": list(self.prompt_templates.keys()),
            "convergence_thresholds": self.config["convergence_thresholds"],
            "decision_history": [
                asdict(d) for d in self.decision_history[-5:]
            ],  # Last 5 decisions
        }

        # Format the template with context
        try:
            formatted_prompt = base_template.format(**prompt_context)
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

    def _create_fallback_decision(
        self, context: Dict[str, Any]
    ) -> OrchestrationDecision:
        """Create a fallback decision when Claude fails"""
        current_step = self.workflow_state.current_step
        current_phase = self.workflow_state.current_phase

        # Simple rule-based fallback logic
        if current_phase == WorkflowPhase.PHASE_1_PLANNING:
            if current_step == 1:
                action_type = "initiate_planning"
                target_agent = AgentType.PLANNER
                template = "step1_specification_analysis_prompt_template"
            elif current_step == 2:
                action_type = "validate_specifications"
                target_agent = AgentType.VALIDATOR
                template = "step2_contract_validation_prompt_template"
            elif current_step == 3:
                action_type = "validate_dependencies"
                target_agent = AgentType.VALIDATOR
                template = "step3_dependency_validation_prompt_template"
            else:
                action_type = "transition_to_implementation"
                target_agent = AgentType.WORKFLOW_ORCHESTRATOR
                template = "workflow_orchestrator"
        else:
            # Phase 2 implementation
            if current_step == 4:
                action_type = "implement_builders"
                target_agent = AgentType.PROGRAMMER
                template = "step4_builder_implementation_prompt_template"
            elif current_step == 5:
                action_type = "implement_configurations"
                target_agent = AgentType.PROGRAMMER
                template = "step5_configuration_implementation_prompt_template"
            elif current_step == 6:
                action_type = "validate_implementation"
                target_agent = AgentType.VALIDATOR
                template = "step6_implementation_validation_prompt_template"
            elif current_step == 7:
                action_type = "final_integration"
                target_agent = AgentType.PROGRAMMER
                template = "step7_integration_prompt_template"
            else:
                action_type = "analyze_current_state"
                target_agent = AgentType.WORKFLOW_ORCHESTRATOR
                template = "workflow_orchestrator"

        decision = OrchestrationDecision(
            action_type=action_type,
            target_agent=target_agent,
            prompt_template=template,
            parameters={"fallback": True},
            reasoning="Claude decision failed, using rule-based fallback",
            confidence=0.5,
            next_step=current_step + 1 if current_step < 7 else None,
        )

        self.decision_history.append(decision)
        return decision

    def _check_phase_transition_criteria(self) -> bool:
        """Check if criteria are met for phase transition"""
        convergence = self.workflow_state.convergence_status
        thresholds = self.config["convergence_thresholds"]

        return (
            convergence.get("alignment", 0) >= thresholds["alignment"]
            and convergence.get("standardization", 0) >= thresholds["standardization"]
            and convergence.get("compatibility", 0) >= thresholds["compatibility"]
        )

    async def execute_workflow_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step with Claude orchestration"""
        try:
            # Make orchestration decision using Claude
            decision = await self.make_orchestration_decision(context)

            # Execute the decision (simulated for now)
            result = self._execute_decision(decision, context)

            # Update workflow state
            self._update_workflow_state(decision, result)

            return {
                "success": True,
                "decision": asdict(decision),
                "result": result,
                "workflow_state": asdict(self.workflow_state),
                "usage_stats": self.usage_stats,
            }

        except Exception as e:
            error_msg = f"Error executing workflow step: {str(e)}"
            logger.error(error_msg)
            self.workflow_state.error_log.append(error_msg)

            return {
                "success": False,
                "error": error_msg,
                "workflow_state": asdict(self.workflow_state),
            }

    def _execute_decision(
        self, decision: OrchestrationDecision, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the orchestration decision (simulated)"""
        logger.info(
            f"Executing {decision.action_type} with {decision.target_agent.value}"
        )

        # Simulate agent execution based on decision
        if decision.target_agent == AgentType.PLANNER:
            return {
                "agent": "planner",
                "action": decision.action_type,
                "template_used": decision.prompt_template,
                "output": f"Planning completed for {decision.action_type}",
                "validation_score": 8.5,
            }
        elif decision.target_agent == AgentType.VALIDATOR:
            return {
                "agent": "validator",
                "action": decision.action_type,
                "template_used": decision.prompt_template,
                "validation_results": {
                    "alignment": 9.2,
                    "standardization": 8.8,
                    "compatibility": 8.5,
                },
                "issues_found": [],
                "recommendations": ["Continue to next step"],
            }
        elif decision.target_agent == AgentType.PROGRAMMER:
            return {
                "agent": "programmer",
                "action": decision.action_type,
                "template_used": decision.prompt_template,
                "files_modified": [f"src/cursus/{decision.action_type}.py"],
                "tests_created": [f"test/{decision.action_type}_test.py"],
                "implementation_score": 8.7,
            }
        else:
            return {"status": "orchestrator_decision", "action": decision.action_type}

    def _update_workflow_state(
        self, decision: OrchestrationDecision, result: Dict[str, Any]
    ) -> None:
        """Update the workflow state based on execution results"""
        # Update completed steps
        if (
            decision.next_step
            and self.workflow_state.current_step
            not in self.workflow_state.completed_steps
        ):
            self.workflow_state.completed_steps.append(self.workflow_state.current_step)

        # Update current step
        if decision.next_step:
            self.workflow_state.current_step = decision.next_step

        # Update validation scores
        if "validation_results" in result:
            self.workflow_state.convergence_status.update(result["validation_results"])

        # Update phase if transition criteria met
        if decision.action_type == "transition_to_implementation":
            self.workflow_state.current_phase = WorkflowPhase.PHASE_2_IMPLEMENTATION

        # Update active agents
        if decision.target_agent not in self.workflow_state.active_agents:
            self.workflow_state.active_agents.append(decision.target_agent)

        # Update metadata
        self.workflow_state.metadata["iteration_count"] += 1

    async def run_complete_workflow(
        self, initial_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the complete 7-step workflow with Claude orchestration"""
        logger.info("Starting complete workflow automation with Claude 4.0")

        workflow_results = []
        context = initial_context.copy()

        while (
            self.workflow_state.current_step <= 7
            and self.workflow_state.metadata["iteration_count"]
            < self.config["max_iterations"]
        ):

            # Execute current step
            step_result = await self.execute_workflow_step(context)
            workflow_results.append(step_result)

            # Update context with results
            context.update(step_result.get("result", {}))

            # Check for completion or errors
            if not step_result["success"]:
                logger.error("Workflow step failed, stopping automation")
                break

            if self.workflow_state.current_step > 7:
                logger.info("Workflow completed successfully")
                break

        return {
            "workflow_completed": self.workflow_state.current_step > 7,
            "final_state": asdict(self.workflow_state),
            "step_results": workflow_results,
            "total_iterations": self.workflow_state.metadata["iteration_count"],
            "usage_summary": self.usage_stats,
        }

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and progress"""
        return {
            "current_step": self.workflow_state.current_step,
            "current_phase": self.workflow_state.current_phase.value,
            "progress_percentage": (len(self.workflow_state.completed_steps) / 7) * 100,
            "convergence_status": self.workflow_state.convergence_status,
            "active_agents": [
                agent.value for agent in self.workflow_state.active_agents
            ],
            "recent_decisions": [asdict(d) for d in self.decision_history[-3:]],
            "error_count": len(self.workflow_state.error_log),
            "usage_stats": self.usage_stats,
        }


# Example usage
async def example_bedrock_orchestration():
    """Example of using Bedrock orchestration with Claude 4.0"""

    # Configuration for Bedrock orchestrator
    config = {
        "aws_region": "us-east-1",
        "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",  # Claude 4.0 equivalent
        "temperature": 0.3,
        "max_tokens": 4000,
        "convergence_thresholds": {
            "alignment": 9.0,
            "standardization": 8.0,
            "compatibility": 8.0,
        },
        "max_iterations": 20,
    }

    # Initialize Bedrock orchestrator
    orchestrator = BedrockOrchestrator(config)

    # Example context
    context = {
        "task_type": "new_pipeline_step",
        "step_name": "data_preprocessing",
        "requirements": ["Handle missing values", "Feature scaling", "Data validation"],
        "target_framework": "scikit-learn",
    }

    # Execute single step with Claude
    result = await orchestrator.execute_workflow_step(context)
    print("Claude Decision Result:", json.dumps(result, indent=2, default=str))

    # Get workflow status
    status = orchestrator.get_workflow_status()
    print("Workflow Status:", json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    # Run example
    asyncio.run(example_bedrock_orchestration())
