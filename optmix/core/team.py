"""
OptMixTeam — The main entry point for interacting with OptMix agents.

Orchestrates specialized marketing agents that collaborate through a shared
state graph using the Agent-as-Code architecture.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from optmix.agents.schema import AgentDefinition, AgentLoader

logger = logging.getLogger(__name__)


class OptMixTeam:
    """
    A collaborative team of AI marketing agents powered by MMM.

    Usage:
        team = OptMixTeam(llm="anthropic/claude-sonnet-4-20250514")
        team.chat("What's driving our CAC increase in Q4?")
        team.invoke("modeler", "Fit a Bayesian MMM on the uploaded data")
    """

    def __init__(
        self,
        llm: str = "anthropic/claude-sonnet-4-20250514",
        api_key: str | None = None,
        agents_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm
        self.api_key = api_key
        self.config = config or {}

        # Load agent definitions
        self._loader = AgentLoader(agents_dir)
        self._agents: dict[str, AgentDefinition] = self._loader.load_all()

        # Initialize shared state
        from optmix.core.state import SharedState

        self._state = SharedState()

        # Initialize tool registry
        from optmix.tools import create_default_registry

        self._tool_registry = create_default_registry()

        # Initialize LLM client (lazy — only when needed)
        self._llm_client: Any = None

        # Initialize orchestrator router
        self._router: Any = None

        # Conversation history
        self._history: list[dict[str, Any]] = []

        # Inter-agent delegation depth tracker (prevents infinite loops)
        self._delegation_depth = 0

        # Register the delegation tool so agents can collaborate
        self._register_delegation_tool()

    def _register_delegation_tool(self) -> None:
        """Register the delegate_to_agent tool for inter-agent communication."""
        from optmix.tools.registry import ToolParameter, ToolSchema

        MAX_DELEGATION_DEPTH = 3

        # All non-orchestrator agents can delegate
        agent_names = [n for n in self._agents if n != "orchestrator"]

        # Build a description of each agent for the tool docs
        agent_descriptions = []
        for name in agent_names:
            role = self._agents[name].persona.role
            agent_descriptions.append(f"  - {name}: {role}")
        agents_help = "\n".join(agent_descriptions)

        schema = ToolSchema(
            name="delegate_to_agent",
            description=(
                "Delegate a task to another specialist agent on the team. "
                "Use this when the user's request requires expertise from a different agent. "
                "For example, the strategist can ask the modeler to fit an MMM, "
                "or the analyst can ask the optimizer for budget recommendations.\n\n"
                "Available agents:\n" + agents_help
            ),
            parameters=[
                ToolParameter(
                    name="agent_name",
                    type="string",
                    description=f"Name of the agent to delegate to. One of: {', '.join(agent_names)}",
                    required=True,
                    enum=agent_names,
                ),
                ToolParameter(
                    name="task",
                    type="string",
                    description=(
                        "Clear, specific description of the task to delegate. "
                        "Include all relevant context the other agent needs."
                    ),
                    required=True,
                ),
            ],
            returns_description="The delegated agent's response with their analysis or results.",
            agent_scope=agent_names,
        )

        team_ref = self  # closure reference

        def delegate_to_agent(state: Any, agent_name: str, task: str) -> dict[str, Any]:
            if team_ref._delegation_depth >= MAX_DELEGATION_DEPTH:
                return {
                    "status": "error",
                    "message": (
                        f"Maximum delegation depth ({MAX_DELEGATION_DEPTH}) reached. "
                        "Please answer with the information gathered so far."
                    ),
                    "summary": "Delegation depth limit reached",
                }

            if agent_name not in team_ref._agents:
                available = ", ".join(team_ref.agents)
                return {
                    "status": "error",
                    "message": f"Unknown agent '{agent_name}'. Available: {available}",
                    "summary": f"Unknown agent: {agent_name}",
                }

            team_ref._delegation_depth += 1
            try:
                agent_def = team_ref._agents[agent_name]
                response = team_ref._execute_agent(agent_def, task)
                return {
                    "status": "success",
                    "agent": agent_name,
                    "response": response,
                    "summary": f"{agent_name} completed the task",
                }
            except Exception as e:
                logger.error("Delegation to '%s' failed: %s", agent_name, e)
                return {
                    "status": "error",
                    "message": f"Delegation to '{agent_name}' failed: {str(e)}",
                    "summary": f"Delegation failed: {str(e)}",
                }
            finally:
                team_ref._delegation_depth -= 1

        self._tool_registry.register(schema, delegate_to_agent)

    def _get_llm_client(self) -> Any:
        """Lazily initialize the LLM client."""
        if self._llm_client is None:
            from optmix.core.llm import create_llm_client

            # Parse provider from llm string (e.g., "anthropic/claude-sonnet-4-20250514")
            # Also supports provider-only strings like "mock"
            parts = self.llm.split("/", 1)
            if len(parts) > 1:
                provider = parts[0]
                model = parts[1]
            else:
                provider = parts[0]
                model = None

            self._llm_client = create_llm_client(
                provider=provider,
                model=model,
                api_key=self.api_key,
            )
        return self._llm_client

    def _get_router(self) -> Any:
        """Lazily initialize the orchestrator router."""
        if self._router is None:
            from optmix.core.orchestrator import OrchestratorRouter

            try:
                llm = self._get_llm_client()
            except (ValueError, ImportError):
                llm = None
                logger.warning("LLM client not available. Using keyword-only routing.")

            self._router = OrchestratorRouter(
                agents=self._agents,
                llm_client=llm,
            )
        return self._router

    @property
    def agents(self) -> list[str]:
        """List available agent names."""
        return list(self._agents.keys())

    @property
    def state(self) -> Any:
        """Access the shared state."""
        return self._state

    def get_agent(self, name: str) -> AgentDefinition:
        """Get an agent definition by name."""
        if name not in self._agents:
            available = ", ".join(self.agents)
            raise ValueError(f"Unknown agent '{name}'. Available: {available}")
        return self._agents[name]

    def chat(self, message: str) -> str:
        """
        Send a message to the team. The Orchestrator routes to the right agent.

        Args:
            message: Natural language message or question.

        Returns:
            Agent response as a string.
        """
        self._history.append({"role": "user", "content": message})

        # Route through orchestrator
        router = self._get_router()
        decision = router.route(message, state=self._state)
        target_agent = decision.agent_name

        logger.info(
            "Routing to '%s' (%s confidence): %s",
            target_agent,
            decision.confidence,
            decision.reasoning,
        )

        agent = self._agents[target_agent]
        response = self._execute_agent(agent, message)

        self._history.append(
            {
                "role": "assistant",
                "agent": target_agent,
                "content": response,
            }
        )
        return response

    def invoke(self, agent_name: str, message: str, **kwargs: Any) -> str:
        """
        Directly invoke a specific agent, bypassing the orchestrator.

        Args:
            agent_name: Name of the agent to invoke.
            message: Task or question for the agent.
            **kwargs: Additional parameters passed to the agent.

        Returns:
            Agent response as a string.
        """
        agent = self.get_agent(agent_name)
        return self._execute_agent(agent, message, **kwargs)

    def run(
        self,
        data: Any = None,
        data_path: str | None = None,
        sample_dataset: str | None = None,
        objective: str = "",
        constraints: dict[str, Any] | None = None,
        workflow: str = "quick_optimization",
        on_step_complete: Any = None,
    ) -> Any:
        """
        Run a complete workflow on the provided data.

        Args:
            data: Marketing data as DataFrame.
            data_path: Path to CSV file.
            sample_dataset: Name of built-in sample dataset.
            objective: Business objective in natural language.
            constraints: Budget constraints and channel limits.
            workflow: Name of the workflow to execute.
            on_step_complete: Callback(step_result) after each step.

        Returns:
            WorkflowResult with model, allocation, and report.
        """
        import pandas as pd

        # Load data into state
        if data is not None:
            self._state.set("raw_data", data, source_agent="user")
        elif data_path:
            df = pd.read_csv(data_path)
            self._state.set("raw_data", df, source_agent="user")
        elif sample_dataset:
            from optmix.data.samples import load_sample

            df = load_sample(sample_dataset)
            self._state.set("raw_data", df, source_agent="user")

        if objective:
            self._state.set("objective", objective, source_agent="user")
        if constraints:
            self._state.set("constraints", constraints, source_agent="user")

        # Run workflow
        from optmix.core.workflow_engine import WorkflowEngine

        engine = WorkflowEngine(
            agents=self._agents,
            tool_registry=self._tool_registry,
            state=self._state,
            llm_client=self._get_llm_client(),
        )

        return engine.run(
            workflow_name=workflow,
            user_context=objective,
            on_step_complete=on_step_complete,
        )

    def load_data(self, path: str | None = None, sample: str | None = None) -> str:
        """
        Convenience method to load data into the team's shared state.

        Args:
            path: Path to a CSV file.
            sample: Name of a built-in sample dataset.

        Returns:
            Summary of loaded data.
        """
        if sample:
            from optmix.data.samples import load_sample

            df = load_sample(sample)
            self._state.set("raw_data", df, source_agent="user")
            return f"Loaded sample dataset '{sample}': {len(df)} rows, {len(df.columns)} columns"
        elif path:
            import pandas as pd

            df = pd.read_csv(path)
            self._state.set("raw_data", df, source_agent="user")
            return f"Loaded '{path}': {len(df)} rows, {len(df.columns)} columns"
        else:
            raise ValueError("Provide either 'path' or 'sample' parameter.")

    def _execute_agent(self, agent: AgentDefinition, message: str, **kwargs: Any) -> str:
        """Execute an agent with the given message via LLM tool-calling loop."""
        try:
            llm = self._get_llm_client()
        except (ValueError, ImportError):
            # Graceful fallback when no LLM is available
            return self._execute_agent_offline(agent, message)

        from optmix.core.executor import AgentExecutor

        executor = AgentExecutor(
            agent=agent,
            llm_client=llm,
            tool_registry=self._tool_registry,
            state=self._state,
        )

        # Build conversation context from recent history
        recent_history = self._history[-10:] if len(self._history) > 10 else self._history
        # Filter to just role/content for the LLM
        conv_history = [
            {"role": h["role"], "content": h["content"]} for h in recent_history if h.get("content")
        ]

        response = executor.run(
            message=message,
            conversation_history=conv_history[:-1] if conv_history else None,
        )

        return response.content

    def _execute_agent_offline(self, agent: AgentDefinition, message: str) -> str:
        """Fallback execution when no LLM is configured."""
        return (
            f"[{agent.persona.name} — {agent.metadata.title}]\n"
            f"Received: {message}\n"
            f"Available tools: {', '.join(agent.tools)}\n\n"
            f"Note: No LLM API key configured. Set ANTHROPIC_API_KEY or OPTMIX_API_KEY "
            f"environment variable to enable full agent execution.\n"
            f"You can still use the MMM engine directly via optmix.mmm module."
        )

    def compile_agents(self, output_dir: str | Path | None = None) -> list[Path]:
        """Compile all agent YAML files to IDE-ready Markdown."""
        compiled = []
        for name in self._agents:
            path = self._loader.compile_to_markdown(name, output_dir)
            compiled.append(path)
        return compiled

    def __repr__(self) -> str:
        return f"OptMixTeam(llm='{self.llm}', agents=[{', '.join(self.agents)}])"
