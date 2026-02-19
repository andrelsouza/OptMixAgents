"""
Workflow execution engine for OptMix.

Parses YAML workflow definitions and executes them step-by-step,
routing each step to the appropriate agent, managing shared state,
and enforcing quality gates between phases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from optmix.agents.schema import AgentDefinition
from optmix.core.executor import AgentExecutor, AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of a single workflow step."""

    step_id: str
    agent_name: str
    status: str  # "completed", "failed", "skipped"
    response: AgentResponse | None = None
    error: str | None = None


@dataclass
class WorkflowResult:
    """Result of a complete workflow execution."""

    workflow_name: str
    status: str  # "completed", "failed", "halted"
    steps_completed: list[StepResult] = field(default_factory=list)
    steps_failed: list[StepResult] = field(default_factory=list)
    steps_skipped: list[StepResult] = field(default_factory=list)
    total_usage: dict[str, int] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "completed"

    @property
    def total_steps(self) -> int:
        return len(self.steps_completed) + len(self.steps_failed) + len(self.steps_skipped)

    def summary(self) -> str:
        """Human-readable summary of workflow execution."""
        lines = [
            f"Workflow: {self.workflow_name}",
            f"Status: {self.status}",
            f"Steps: {len(self.steps_completed)} completed, "
            f"{len(self.steps_failed)} failed, "
            f"{len(self.steps_skipped)} skipped",
        ]
        if self.total_usage:
            lines.append(
                f"Tokens: {self.total_usage.get('input_tokens', 0):,} in, "
                f"{self.total_usage.get('output_tokens', 0):,} out"
            )
        return "\n".join(lines)


@dataclass
class WorkflowStep:
    """Parsed workflow step definition."""

    id: str
    agent: str
    action: str
    description: str = ""
    input_keys: list[str] = field(default_factory=list)
    output_key: str | None = None
    required: bool = True
    gate: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowPhase:
    """A named phase containing ordered steps."""

    name: str
    title: str
    description: str
    steps: list[WorkflowStep]


@dataclass
class WorkflowDefinition:
    """Parsed workflow YAML."""

    name: str
    title: str
    description: str
    phases: list[WorkflowPhase]
    shared_state_keys: list[str] = field(default_factory=list)
    quality_gates: list[dict[str, Any]] = field(default_factory=list)


class WorkflowEngine:
    """
    Executes YAML-defined workflows step-by-step.

    Each step invokes an agent via the AgentExecutor, passes relevant
    state context, and stores outputs in SharedState. Quality gates
    are evaluated between phases.
    """

    def __init__(
        self,
        agents: dict[str, AgentDefinition],
        tool_registry: Any,
        state: Any,
        llm_client: Any,
        workflows_dir: str | Path | None = None,
        checklists_dir: str | Path | None = None,
    ) -> None:
        self.agents = agents
        self.tool_registry = tool_registry
        self.state = state
        self.llm_client = llm_client

        if workflows_dir is None:
            workflows_dir = Path(__file__).parent.parent.parent / "workflows"
        self.workflows_dir = Path(workflows_dir)

        if checklists_dir is None:
            checklists_dir = Path(__file__).parent.parent.parent / "checklists"
        self.checklists_dir = Path(checklists_dir)

    def list_workflows(self) -> list[str]:
        """List available workflow names."""
        return [p.stem for p in sorted(self.workflows_dir.glob("*.yaml"))]

    def load_workflow(self, name: str) -> WorkflowDefinition:
        """Load and parse a workflow YAML file."""
        path = self.workflows_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Workflow not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        wf = raw.get("workflow", raw)
        metadata = wf.get("metadata", wf)

        # Parse phases and steps
        phases = []
        raw_phases = wf.get("phases")
        raw_steps = wf.get("steps")

        if raw_phases:
            # Phased workflow (like full_measurement_cycle)
            for phase_data in raw_phases:
                steps = [self._parse_step(s) for s in phase_data.get("steps", [])]
                phases.append(
                    WorkflowPhase(
                        name=phase_data.get("name", ""),
                        title=phase_data.get("title", ""),
                        description=phase_data.get("description", ""),
                        steps=steps,
                    )
                )
        elif raw_steps:
            # Flat workflow (like quick_optimization)
            steps = [self._parse_step(s) for s in raw_steps]
            phases.append(
                WorkflowPhase(
                    name="main",
                    title=metadata.get("title", name),
                    description=metadata.get("description", ""),
                    steps=steps,
                )
            )

        return WorkflowDefinition(
            name=metadata.get("name", name),
            title=metadata.get("title", name),
            description=metadata.get("description", ""),
            phases=phases,
            shared_state_keys=wf.get("shared_state", []),
            quality_gates=wf.get("quality_gates", []),
        )

    def run(
        self,
        workflow_name: str,
        user_context: str = "",
        on_step_complete: Any = None,
    ) -> WorkflowResult:
        """
        Execute a workflow end-to-end.

        Args:
            workflow_name: Name of the workflow YAML file (without extension).
            user_context: Additional context from the user about their objectives.
            on_step_complete: Optional callback(step_result) called after each step.

        Returns:
            WorkflowResult with all step results.
        """
        workflow = self.load_workflow(workflow_name)
        logger.info("Starting workflow: %s (%d phases)", workflow.name, len(workflow.phases))

        result = WorkflowResult(workflow_name=workflow.name, status="in_progress")
        total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        for phase in workflow.phases:
            logger.info("Entering phase: %s", phase.title)

            for step in phase.steps:
                step_result = self._execute_step(step, workflow, user_context)

                # Accumulate usage
                if step_result.response:
                    for k, v in step_result.response.usage.items():
                        total_usage[k] = total_usage.get(k, 0) + v

                if step_result.status == "completed":
                    result.steps_completed.append(step_result)
                elif step_result.status == "skipped":
                    result.steps_skipped.append(step_result)
                else:
                    result.steps_failed.append(step_result)

                    if step.required:
                        logger.error(
                            "Required step '%s' failed: %s",
                            step.id,
                            step_result.error,
                        )
                        result.status = "failed"
                        result.total_usage = total_usage
                        return result

                if on_step_complete:
                    on_step_complete(step_result)

                # Check quality gate after this step
                gate_result = self._evaluate_gate(step, workflow)
                if gate_result == "halt":
                    logger.warning("Quality gate halted workflow after step '%s'", step.id)
                    result.status = "halted"
                    result.total_usage = total_usage
                    return result

        result.status = "completed"
        result.total_usage = total_usage
        logger.info("Workflow '%s' completed successfully.", workflow.name)
        return result

    def _execute_step(
        self,
        step: WorkflowStep,
        workflow: WorkflowDefinition,
        user_context: str,
    ) -> StepResult:
        """Execute a single workflow step."""
        # Check if agent exists
        if step.agent not in self.agents:
            return StepResult(
                step_id=step.id,
                agent_name=step.agent,
                status="failed",
                error=f"Agent '{step.agent}' not found.",
            )

        agent_def = self.agents[step.agent]
        logger.info(
            "Executing step '%s' with agent '%s' (%s)",
            step.id,
            step.agent,
            agent_def.persona.name,
        )

        # Build the step prompt
        prompt = self._build_step_prompt(step, user_context)

        # Create executor for this agent
        executor = AgentExecutor(
            agent=agent_def,
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            state=self.state,
        )

        try:
            response = executor.run(
                message=prompt,
                extra_context=f"Workflow: {workflow.title}\nStep: {step.id} — {step.description}",
            )

            # Store the response content in state if output key is specified
            if step.output_key:
                self.state.set(
                    step.output_key,
                    response.content,
                    source_agent=step.agent,
                )

            return StepResult(
                step_id=step.id,
                agent_name=step.agent,
                status="completed",
                response=response,
            )

        except Exception as e:
            logger.error("Step '%s' raised exception: %s", step.id, e)
            return StepResult(
                step_id=step.id,
                agent_name=step.agent,
                status="failed",
                error=str(e),
            )

    def _build_step_prompt(self, step: WorkflowStep, user_context: str) -> str:
        """Build the prompt for a workflow step."""
        parts = [
            f"## Workflow Step: {step.id}",
            "",
            step.description,
            "",
        ]

        # Include input references
        if step.input_keys:
            parts.append("### Available Inputs")
            parts.append("")
            for key in step.input_keys:
                if self.state.has(key):
                    summary = self.state.get(key)
                    if hasattr(summary, "__len__") and not isinstance(summary, str):
                        parts.append(f"- **{key}**: Available (use your tools to access)")
                    else:
                        parts.append(f"- **{key}**: {str(summary)[:200]}")
                else:
                    parts.append(f"- **{key}**: Not yet available")
            parts.append("")

        if step.output_key:
            parts.append(f"### Expected Output: `{step.output_key}`")
            parts.append("")

        if user_context:
            parts.extend(
                [
                    "### User Context",
                    "",
                    user_context,
                    "",
                ]
            )

        parts.extend(
            [
                "### Instructions",
                "",
                "Use your available tools to complete this step. "
                "Provide a clear, detailed response with your findings and recommendations.",
            ]
        )

        return "\n".join(parts)

    def _evaluate_gate(
        self,
        step: WorkflowStep,
        workflow: WorkflowDefinition,
    ) -> str | None:
        """Evaluate quality gate after a step. Returns 'halt', 'retry', or None."""
        if not step.gate:
            return None

        # Find the gate definition
        gate_def = None
        for gate in workflow.quality_gates:
            if gate.get("id") == step.gate or gate.get("checklist") == step.gate:
                gate_def = gate
                break

        if gate_def is None:
            logger.warning("Quality gate '%s' not found in workflow definition.", step.gate)
            return None

        # Load checklist
        checklist_name = gate_def.get("checklist", step.gate)
        checklist_path = self.checklists_dir / f"{checklist_name}.md"

        if not checklist_path.exists():
            logger.warning("Checklist file not found: %s", checklist_path)
            return None

        # For v0.1, quality gates log a warning but don't block
        # Full evaluation will use the LLM to assess checklist against state
        logger.info(
            "Quality gate '%s' passed (evaluation is advisory in v0.1).",
            checklist_name,
        )
        return None

    def _parse_step(self, step_data: dict[str, Any]) -> WorkflowStep:
        """Parse a raw step dict from YAML into a WorkflowStep."""
        input_keys = step_data.get("input", [])
        if isinstance(input_keys, str):
            input_keys = [input_keys]

        return WorkflowStep(
            id=step_data.get("id", ""),
            agent=step_data.get("agent", ""),
            action=step_data.get("action", ""),
            description=step_data.get("description", ""),
            input_keys=input_keys,
            output_key=step_data.get("output"),
            required=step_data.get("required", True),
            gate=step_data.get("gate"),
            config=step_data.get("config", {}),
        )
