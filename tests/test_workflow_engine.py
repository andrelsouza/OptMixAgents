"""Tests for WorkflowEngine — YAML loading, step execution, quality gates."""

import pytest

from optmix.core.workflow_engine import (
    WorkflowDefinition,
    WorkflowEngine,
    WorkflowPhase,
    WorkflowResult,
    WorkflowStep,
)


@pytest.fixture()
def agents():
    """Load all agent definitions."""
    from optmix.agents.schema import AgentLoader

    loader = AgentLoader()
    return loader.load_all()


@pytest.fixture()
def mock_llm():
    """Create a MockLLMClient."""
    from optmix.core.llm import MockLLMClient

    return MockLLMClient()


@pytest.fixture()
def state():
    """Create a fresh SharedState."""
    from optmix.core.state import SharedState

    return SharedState()


@pytest.fixture()
def registry():
    """Create the default tool registry."""
    from optmix.tools import create_default_registry

    return create_default_registry()


@pytest.fixture()
def engine(agents, registry, state, mock_llm):
    """Create a WorkflowEngine with mock LLM."""
    return WorkflowEngine(
        agents=agents,
        tool_registry=registry,
        state=state,
        llm_client=mock_llm,
    )


class TestWorkflowLoading:
    """Test YAML workflow loading and parsing."""

    def test_list_workflows(self, engine):
        workflows = engine.list_workflows()
        assert "quick_optimization" in workflows
        assert "full_measurement_cycle" in workflows

    def test_load_quick_optimization(self, engine):
        wf = engine.load_workflow("quick_optimization")
        assert isinstance(wf, WorkflowDefinition)
        assert wf.name == "quick-optimization"
        assert len(wf.phases) == 1
        # Flat workflow → 1 phase with all steps
        steps = wf.phases[0].steps
        assert len(steps) == 5
        step_ids = [s.id for s in steps]
        assert "validate_data" in step_ids
        assert "fit_model" in step_ids
        assert "optimize_budget" in step_ids

    def test_load_full_measurement_cycle(self, engine):
        wf = engine.load_workflow("full_measurement_cycle")
        assert isinstance(wf, WorkflowDefinition)
        assert len(wf.phases) == 4
        phase_names = [p.name for p in wf.phases]
        assert phase_names == ["strategize", "model", "optimize", "activate"]

    def test_load_nonexistent_workflow_raises(self, engine):
        with pytest.raises(FileNotFoundError, match="Workflow not found"):
            engine.load_workflow("nonexistent_workflow")

    def test_step_parsing(self, engine):
        wf = engine.load_workflow("quick_optimization")
        step = wf.phases[0].steps[0]
        assert isinstance(step, WorkflowStep)
        assert step.id == "validate_data"
        assert step.agent == "analyst"
        assert step.required is True

    def test_step_input_keys(self, engine):
        wf = engine.load_workflow("quick_optimization")
        # auto_features step has input: [validated_data]
        auto_features = next(s for s in wf.phases[0].steps if s.id == "auto_features")
        assert "validated_data" in auto_features.input_keys

    def test_step_output_key(self, engine):
        wf = engine.load_workflow("quick_optimization")
        validate = next(s for s in wf.phases[0].steps if s.id == "validate_data")
        assert validate.output_key == "validated_data"

    def test_full_cycle_quality_gates(self, engine):
        wf = engine.load_workflow("full_measurement_cycle")
        assert len(wf.quality_gates) == 2
        gate_ids = [g["id"] for g in wf.quality_gates]
        assert "data-readiness" in gate_ids
        assert "model-validation" in gate_ids

    def test_full_cycle_shared_state_keys(self, engine):
        wf = engine.load_workflow("full_measurement_cycle")
        assert "fitted_model" in wf.shared_state_keys
        assert "optimal_allocation" in wf.shared_state_keys


class TestWorkflowExecution:
    """Test workflow step execution with mock LLM."""

    def test_run_quick_optimization_completes(self, engine, mock_llm, state):
        """With mock LLM responses, the workflow should complete all steps."""
        # Pre-load data into state so tools can work
        from optmix.data.samples import load_sample

        df = load_sample("ecommerce")
        state.set("raw_data", df, source_agent="user")

        # Add enough mock responses for each step (agent will get one response per step)
        for _ in range(10):
            mock_llm.add_response("Step completed successfully.")

        result = engine.run("quick_optimization", user_context="Test run")
        assert isinstance(result, WorkflowResult)
        assert result.workflow_name == "quick-optimization"
        # All steps should complete (mock LLM gives immediate text responses, no tool calls)
        assert len(result.steps_completed) == 5
        assert result.status == "completed"
        assert result.success is True

    def test_on_step_complete_callback(self, engine, mock_llm, state):
        """Callback should fire after each step."""
        from optmix.data.samples import load_sample

        df = load_sample("ecommerce")
        state.set("raw_data", df, source_agent="user")

        for _ in range(10):
            mock_llm.add_response("Done.")

        step_log = []
        engine.run(
            "quick_optimization",
            user_context="Test",
            on_step_complete=lambda sr: step_log.append(sr.step_id),
        )
        assert len(step_log) == 5
        assert step_log[0] == "validate_data"

    def test_failed_required_step_halts_workflow(self, engine, state):
        """If agent not found, required step fails and workflow stops."""
        wf = WorkflowDefinition(
            name="test",
            title="Test",
            description="",
            phases=[
                WorkflowPhase(
                    name="main",
                    title="Main",
                    description="",
                    steps=[
                        WorkflowStep(
                            id="bad_step",
                            agent="nonexistent_agent",
                            action="noop",
                            required=True,
                        ),
                    ],
                )
            ],
        )
        # Manually execute by injecting workflow
        from unittest.mock import patch

        with patch.object(engine, "load_workflow", return_value=wf):
            result = engine.run("test")

        assert result.status == "failed"
        assert len(result.steps_failed) == 1
        assert result.steps_failed[0].error == "Agent 'nonexistent_agent' not found."

    def test_optional_failed_step_continues(self, engine, mock_llm, state):
        """Non-required steps that fail should not stop the workflow."""
        from optmix.data.samples import load_sample

        df = load_sample("ecommerce")
        state.set("raw_data", df, source_agent="user")

        wf = WorkflowDefinition(
            name="test",
            title="Test",
            description="",
            phases=[
                WorkflowPhase(
                    name="main",
                    title="Main",
                    description="",
                    steps=[
                        WorkflowStep(
                            id="optional_bad",
                            agent="nonexistent_agent",
                            action="noop",
                            required=False,
                        ),
                        WorkflowStep(
                            id="good_step",
                            agent="analyst",
                            action="validate",
                            required=True,
                        ),
                    ],
                )
            ],
        )

        mock_llm.add_response("Validated successfully.")

        from unittest.mock import patch

        with patch.object(engine, "load_workflow", return_value=wf):
            result = engine.run("test")

        assert result.status == "completed"
        assert len(result.steps_failed) == 1
        assert len(result.steps_completed) == 1

    def test_step_stores_output_in_state(self, engine, mock_llm, state):
        """Step output_key should store response content in state."""
        from optmix.data.samples import load_sample

        df = load_sample("ecommerce")
        state.set("raw_data", df, source_agent="user")

        mock_llm.add_response("Analysis complete: data looks good.")

        wf = WorkflowDefinition(
            name="test",
            title="Test",
            description="",
            phases=[
                WorkflowPhase(
                    name="main",
                    title="Main",
                    description="",
                    steps=[
                        WorkflowStep(
                            id="analyze",
                            agent="analyst",
                            action="validate",
                            output_key="analysis_output",
                            required=True,
                        ),
                    ],
                )
            ],
        )

        from unittest.mock import patch

        with patch.object(engine, "load_workflow", return_value=wf):
            engine.run("test")

        assert state.has("analysis_output")
        assert "Analysis complete" in state.get("analysis_output")


class TestWorkflowResult:
    """Test WorkflowResult dataclass."""

    def test_summary_formatting(self):
        result = WorkflowResult(
            workflow_name="test",
            status="completed",
            total_usage={"input_tokens": 1000, "output_tokens": 500},
        )
        summary = result.summary()
        assert "test" in summary
        assert "completed" in summary
        assert "1,000" in summary

    def test_total_steps_count(self):
        from optmix.core.workflow_engine import StepResult

        result = WorkflowResult(
            workflow_name="test",
            status="completed",
            steps_completed=[
                StepResult(step_id="a", agent_name="x", status="completed"),
                StepResult(step_id="b", agent_name="y", status="completed"),
            ],
            steps_skipped=[
                StepResult(step_id="c", agent_name="z", status="skipped"),
            ],
        )
        assert result.total_steps == 3
        assert result.success is True
