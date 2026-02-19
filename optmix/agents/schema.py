"""
Agent schema definition and YAML loader.

Agents in OptMix follow the Agent-as-Code pattern:
each agent is a self-contained YAML file with persona, tools, menu, and dependencies.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AgentPersona(BaseModel):
    """Defines an agent's identity and communication style."""

    name: str
    role: str
    identity: str
    communication_style: str
    focus: list[str] = Field(default_factory=list)
    core_principles: list[str] = Field(default_factory=list)


class MenuItem(BaseModel):
    """A single menu item mapping a trigger to a workflow."""

    trigger: str
    title: str
    workflow: str | None = None
    description: str = ""


class AgentMetadata(BaseModel):
    """Agent identification and classification."""

    name: str
    title: str
    version: str = "1.0.0"
    module: str = "optmix"
    hasSidecar: bool = False


class AgentDependencies(BaseModel):
    """Templates, checklists, and data an agent requires."""

    templates: list[str] = Field(default_factory=list)
    checklists: list[str] = Field(default_factory=list)
    data: list[str] = Field(default_factory=list)


class AgentDefinition(BaseModel):
    """Complete agent definition loaded from YAML."""

    metadata: AgentMetadata
    persona: AgentPersona
    tools: list[str] = Field(default_factory=list)
    menu: list[MenuItem] = Field(default_factory=list)
    dependencies: AgentDependencies = Field(default_factory=AgentDependencies)
    routing_rules: list[dict[str, str]] | None = None

    @property
    def system_prompt(self) -> str:
        """Compile the agent definition into an LLM system prompt."""
        p = self.persona
        sections = [
            f"# Agent: {self.metadata.title}",
            f"You are **{p.name}**, {p.role}.\n",
            f"## Identity\n{p.identity}\n",
            f"## Communication Style\n{p.communication_style}\n",
            "## Focus Areas",
            *[f"- {f}" for f in p.focus],
            "",
            "## Core Principles",
            *[f"- {pr}" for pr in p.core_principles],
            "",
            "## Available Tools",
            *[f"- `{t}`" for t in self.tools],
            "",
            "## Commands",
            *[f"- **{m.trigger}**: {m.title} — {m.description}" for m in self.menu],
        ]
        return "\n".join(sections)


class AgentLoader:
    """Loads agent definitions from YAML files."""

    def __init__(self, agents_dir: str | Path | None = None) -> None:
        if agents_dir is None:
            agents_dir = Path(__file__).parent.parent.parent / "agents"
        self.agents_dir = Path(agents_dir)

    def load(self, agent_name: str) -> AgentDefinition:
        """Load a single agent by name."""
        path = self.agents_dir / f"{agent_name}.agent.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Agent definition not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        agent_data = raw.get("agent", raw)
        return AgentDefinition(**agent_data)

    def load_all(self) -> dict[str, AgentDefinition]:
        """Load all agent definitions from the agents directory."""
        agents = {}
        for path in sorted(self.agents_dir.glob("*.agent.yaml")):
            name = path.stem.replace(".agent", "")
            agents[name] = self.load(name)
        return agents

    def list_agents(self) -> list[str]:
        """List all available agent names."""
        return [
            p.stem.replace(".agent", "")
            for p in sorted(self.agents_dir.glob("*.agent.yaml"))
        ]

    def compile_to_markdown(self, agent_name: str, output_dir: str | Path | None = None) -> Path:
        """Compile a YAML agent definition to an IDE-ready Markdown file."""
        agent = self.load(agent_name)
        if output_dir is None:
            output_dir = self.agents_dir / "_compiled"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        md_content = self._build_markdown(agent)
        output_path = output_dir / f"{agent_name}.md"
        output_path.write_text(md_content)
        return output_path

    def _build_markdown(self, agent: AgentDefinition) -> str:
        """Build the compiled Markdown content for an agent."""
        p = agent.persona
        lines = [
            f"# {agent.metadata.title} — {p.name}",
            "",
            "---",
            "",
            "## Activation",
            "",
            "When this file is loaded, adopt the following persona and capabilities.",
            "",
            f"**Module:** {agent.metadata.module}",
            f"**Version:** {agent.metadata.version}",
            "",
            "---",
            "",
            "## Persona",
            "",
            f"**Name:** {p.name}",
            f"**Role:** {p.role}",
            "",
            f"### Identity\n\n{p.identity}",
            "",
            f"### Communication Style\n\n{p.communication_style}",
            "",
            "### Focus Areas",
            "",
            *[f"- {f}" for f in p.focus],
            "",
            "### Core Principles",
            "",
            *[f"- {pr}" for pr in p.core_principles],
            "",
            "---",
            "",
            "## Tools",
            "",
            *[f"- `{t}`" for t in agent.tools],
            "",
            "---",
            "",
            "## Menu",
            "",
            *[
                f"### {m.title}\n- **Trigger:** `{m.trigger}`\n- **Workflow:** `{m.workflow}`\n- {m.description}\n"
                for m in agent.menu
            ],
            "---",
            "",
            "## Dependencies",
            "",
            "### Templates",
            *[f"- {t}" for t in agent.dependencies.templates],
            "",
            "### Checklists",
            *[f"- {c}" for c in agent.dependencies.checklists],
            "",
            "### Data",
            *[f"- {d}" for d in agent.dependencies.data],
        ]
        return "\n".join(lines) + "\n"
