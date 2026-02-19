"""
OptMix CLI — Command-line interface for the marketing measurement team.

Usage:
    optmix setup                   # Interactive setup wizard
    optmix chat                    # Interactive chat with the full team
    optmix agent strategist        # Load a specific agent
    optmix run --data spend.csv    # Run a full measurement cycle
    optmix compile                 # Compile agent YAMLs to Markdown
    optmix team                    # Show all available agents
"""

from __future__ import annotations

import sys
from typing import Any

try:
    import click
except ImportError:
    print("CLI requires 'click'. Install with: pip install click")
    sys.exit(1)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _parse_llm_flag(llm: str | None) -> tuple[str | None, str | None]:
    """Parse --llm flag into (provider, model). Returns (None, None) if not set."""
    if not llm:
        return None, None
    parts = llm.split("/", 1)
    if len(parts) > 1:
        return parts[0], parts[1]
    return parts[0], None


def _resolve_and_build_team(
    llm: str | None,
    api_key: str | None,
) -> Any:
    """Resolve config and create an OptMixTeam, or show setup hint and exit."""
    from optmix.core.config import resolve_config

    provider, model = _parse_llm_flag(llm)
    config = resolve_config(provider=provider, model=model, api_key=api_key)

    from optmix.core.team import OptMixTeam

    return OptMixTeam(llm=config.llm_string, api_key=config.api_key)


@click.group()
@click.version_option(version="0.1.0", prog_name="optmix")
def cli() -> None:
    """OptMix — AI Marketing Agents powered by Marketing Mix Modeling."""
    pass


@cli.command()
def setup() -> None:
    """Interactive setup wizard — configure your LLM provider and API key."""
    from optmix.core.config import (
        DEFAULT_MODELS,
        SUPPORTED_PROVIDERS,
        OptMixConfig,
        load_config,
        save_config,
    )

    console.print(
        Panel(
            "[bold]OptMix Setup Wizard[/bold]\n\n"
            "Configure your LLM provider so you can use OptMix agents.\n"
            "Your settings will be saved to [cyan]~/.optmix/config.yaml[/cyan].",
            border_style="blue",
        )
    )

    # Step 1: Choose provider
    console.print("\n[bold]Step 1/3:[/bold] Choose your LLM provider\n")
    for i, p in enumerate(SUPPORTED_PROVIDERS, 1):
        console.print(f"  [{i}] {p.capitalize()}")

    while True:
        choice = click.prompt(
            "\nProvider",
            type=click.IntRange(1, len(SUPPORTED_PROVIDERS)),
            default=1,
        )
        provider = SUPPORTED_PROVIDERS[choice - 1]
        break

    # Step 2: Enter API key
    console.print(f"\n[bold]Step 2/3:[/bold] Enter your {provider.capitalize()} API key\n")
    existing = load_config()
    hint = ""
    if existing.api_key and existing.provider == provider:
        masked = existing.api_key[:8] + "..." + existing.api_key[-4:]
        hint = f" (current: {masked})"

    api_key = click.prompt(
        f"API key{hint}",
        hide_input=True,
        default=existing.api_key if existing.provider == provider else "",
        show_default=False,
    )

    if not api_key:
        console.print("\n[red]No API key provided. Setup cancelled.[/red]")
        return

    # Step 3: Choose model
    default_model = DEFAULT_MODELS.get(provider, "")
    console.print("\n[bold]Step 3/3:[/bold] Choose model (press Enter for default)\n")
    model = click.prompt("Model", default=default_model, show_default=True)

    # Save
    config = OptMixConfig(provider=provider, model=model, api_key=api_key)
    path = save_config(config)

    console.print(
        Panel(
            f"[green]Config saved to {path}[/green]\n\n"
            f"  Provider: [bold]{provider}[/bold]\n"
            f"  Model:    [bold]{model}[/bold]\n\n"
            "Run [cyan]optmix chat[/cyan] to start talking to your agents!",
            title="Setup Complete",
            border_style="green",
        )
    )

    # Optional: test connection
    if click.confirm("\nTest the connection now?", default=True):
        console.print()
        with console.status("[bold blue]Testing connection...[/bold blue]"):
            try:
                from optmix.core.llm import create_llm_client

                client = create_llm_client(
                    provider=provider,
                    model=model,
                    api_key=api_key,
                )
                response = client.chat(
                    system="You are a helpful assistant.",
                    messages=[{"role": "user", "content": "Say 'OptMix connected!' in 3 words or fewer."}],
                    max_tokens=20,
                )
                console.print(f"[green]Connection OK![/green] Response: {response.content}")
            except Exception as e:
                console.print(f"[yellow]Connection test failed:[/yellow] {e}")
                console.print("[dim]Your config is saved. You can fix the key and re-run setup later.[/dim]")


@cli.command()
@click.option("--llm", default=None, help="LLM provider/model (e.g. anthropic/claude-sonnet-4-5-20250929)")
@click.option("--api-key", default=None, help="API key (overrides saved config)")
def chat(llm: str | None, api_key: str | None) -> None:
    """Start an interactive chat with the OptMix team."""
    team = _resolve_and_build_team(llm, api_key)

    console.print(
        Panel(
            "[bold]Welcome to OptMix[/bold] — Your AI Marketing Measurement Team\n\n"
            f"Agents ready: {', '.join(team.agents)}\n\n"
            "[dim]Commands:[/dim]\n"
            "  [cyan]load <path.csv>[/cyan]  — Load a CSV file\n"
            "  [cyan]sample <name>[/cyan]    — Load sample data (ecommerce, retail_chain, saas_b2b)\n"
            "  [cyan]state[/cyan]            — Show current shared state\n"
            "  [cyan]quit[/cyan]             — Exit\n\n"
            "Type your question or command to get started.",
            title="OptMix",
            border_style="blue",
        )
    )

    while True:
        try:
            message = console.input("\n[bold blue]You:[/bold blue] ")
        except (EOFError, KeyboardInterrupt):
            break

        stripped = message.strip()
        if not stripped:
            continue

        if stripped.lower() in ("quit", "exit", "q"):
            console.print("\n[dim]Goodbye! May your ROAS be ever in your favor.[/dim]")
            break

        # Handle special commands
        _SAMPLE_NAMES = ("ecommerce", "retail_chain", "saas_b2b")

        if stripped.lower().startswith("load ") or stripped.lower().startswith("sample "):
            arg = stripped.split(None, 1)[1] if " " in stripped else ""
            arg_lower = arg.lower()

            # Detect sample dataset names anywhere in the argument
            matched_sample = None
            for sname in _SAMPLE_NAMES:
                if sname in arg_lower:
                    matched_sample = sname
                    break

            try:
                if matched_sample:
                    result = team.load_data(sample=matched_sample)
                else:
                    result = team.load_data(path=arg)
                console.print(f"\n[green]{result}[/green]")
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
            continue

        if stripped.lower() == "state":
            summary = team.state.to_summary()
            if summary:
                console.print(f"\n[cyan]{summary}[/cyan]")
            else:
                console.print("\n[dim]No state yet. Load some data to get started.[/dim]")
            continue

        # Send to team
        try:
            with console.status("[bold blue]Thinking...[/bold blue]"):
                response = team.chat(message)
            console.print(f"\n{response}")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            console.print("[dim]Tip: Run 'optmix setup' to configure your LLM provider.[/dim]")


@cli.command()
@click.argument("agent_name")
@click.option("--llm", default=None, help="LLM provider/model (e.g. openai/gpt-4o)")
@click.option("--api-key", default=None, help="API key (overrides saved config)")
def agent(agent_name: str, llm: str | None, api_key: str | None) -> None:
    """Load and interact with a specific agent."""
    team = _resolve_and_build_team(llm, api_key)

    try:
        agent_def = team.get_agent(agent_name)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    p = agent_def.persona
    console.print(
        Panel(
            f"[bold]{p.name}[/bold] — {p.role}\n\n"
            f"{p.identity.strip()}\n\n"
            f"[dim]Tools: {', '.join(agent_def.tools)}[/dim]",
            title=f"{agent_def.metadata.title}",
            border_style="green",
        )
    )

    # Show menu
    table = Table(title="Available Commands")
    table.add_column("Trigger", style="cyan")
    table.add_column("Action", style="white")
    for item in agent_def.menu:
        table.add_row(item.trigger, item.title)
    console.print(table)

    while True:
        try:
            message = console.input(f"\n[bold green]You -> {p.name}:[/bold green] ")
        except (EOFError, KeyboardInterrupt):
            break

        if message.strip().lower() in ("quit", "exit", "q"):
            break

        try:
            with console.status(f"[bold green]{p.name} is thinking...[/bold green]"):
                response = team.invoke(agent_name, message)
            console.print(f"\n{response}")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


@cli.command()
@click.option("--data", "data_path", type=click.Path(exists=True), help="Path to CSV data")
@click.option("--sample", "sample_name", type=click.Choice(["ecommerce", "retail_chain", "saas_b2b"]), help="Use built-in sample data")
@click.option("--target", default="revenue", help="Target variable column name")
@click.option("--budget", type=float, default=None, help="Total budget for optimization")
@click.option("--workflow", default="quick_optimization", help="Workflow to execute")
@click.option("--llm", default=None, help="LLM provider/model (e.g. anthropic/claude-sonnet-4-5-20250929)")
@click.option("--api-key", default=None, help="API key (overrides saved config)")
def run(
    data_path: str | None,
    sample_name: str | None,
    target: str,
    budget: float | None,
    workflow: str,
    llm: str | None,
    api_key: str | None,
) -> None:
    """Run a complete measurement workflow on your data."""
    if not data_path and not sample_name:
        console.print("[red]Error:[/red] Provide --data or --sample")
        return

    team = _resolve_and_build_team(llm, api_key)

    constraints: dict[str, Any] = {}
    objective = f"Analyze marketing effectiveness targeting '{target}'"
    if budget:
        objective += f" and optimize budget allocation for ${budget:,.0f}"

    def on_step(step_result: Any) -> None:
        status_icon = {"completed": "[green]OK[/green]", "failed": "[red]FAIL[/red]", "skipped": "[yellow]SKIP[/yellow]"}
        icon = status_icon.get(step_result.status, step_result.status)
        console.print(f"  {icon} Step: {step_result.step_id} ({step_result.agent_name})")

    console.print(
        Panel(
            f"[bold]Running workflow:[/bold] {workflow}\n"
            f"[bold]Data:[/bold] {data_path or sample_name}\n"
            f"[bold]Objective:[/bold] {objective}",
            title="OptMix Workflow",
            border_style="blue",
        )
    )

    try:
        result = team.run(
            data_path=data_path,
            sample_dataset=sample_name,
            objective=objective,
            constraints=constraints,
            workflow=workflow,
            on_step_complete=on_step,
        )

        console.print(f"\n{result.summary()}")

        # Show final report if available
        report = team.state.get("executive_report")
        if report:
            console.print(Panel(str(report)[:2000], title="Executive Report", border_style="green"))

    except Exception as e:
        console.print(f"\n[red]Workflow failed: {e}[/red]")


@cli.command()
@click.option("--agents-dir", type=click.Path(exists=True), help="Path to agent YAML files")
@click.option("--output-dir", type=click.Path(), help="Output directory for compiled files")
def compile(agents_dir: str | None, output_dir: str | None) -> None:
    """Compile agent YAML definitions to IDE-ready Markdown."""
    from optmix.agents.schema import AgentLoader

    loader = AgentLoader(agents_dir)
    agents = loader.list_agents()

    console.print(f"\n[bold]Compiling {len(agents)} agents...[/bold]\n")

    for name in agents:
        path = loader.compile_to_markdown(name, output_dir)
        console.print(f"  [green]OK[/green] {name} -> {path}")

    console.print(f"\n[green]Done! {len(agents)} agents compiled.[/green]")


@cli.command()
def team() -> None:
    """Show all available agents and their capabilities."""
    from optmix.agents.schema import AgentLoader

    loader = AgentLoader()
    agents = loader.load_all()

    table = Table(title="OptMix Agent Team")
    table.add_column("Agent", style="cyan", width=12)
    table.add_column("Persona", style="bold", width=18)
    table.add_column("Role", width=30)
    table.add_column("Tools", style="dim", width=30)

    for name, agent_def in agents.items():
        p = agent_def.persona
        tools_str = ", ".join(agent_def.tools[:3])
        if len(agent_def.tools) > 3:
            tools_str += f" (+{len(agent_def.tools) - 3} more)"
        table.add_row(name, p.name, p.role, tools_str)

    console.print(table)


if __name__ == "__main__":
    cli()
