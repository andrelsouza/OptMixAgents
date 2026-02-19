"""
Report generation tools for the Reporter agent (Nora).

Generates Markdown reports, charts, and action plans from MMM results.
Tools in this module render Jinja2 templates and create visualizations.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from optmix.tools.registry import ToolParameter, ToolSchema

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"

# --- Schemas ---

GENERATE_REPORT_SCHEMA = ToolSchema(
    name="generate_markdown_report",
    description="Generate an executive report in Markdown from the fitted model, optimization results, and EDA. Covers model performance, channel ROAS, budget allocation, and saturation analysis.",
    parameters=[],
    returns_description="Full Markdown report string with model summary, channel performance, and recommendations.",
    agent_scope=["reporter"],
)

GENERATE_CHART_SCHEMA = ToolSchema(
    name="generate_chart",
    description="Generate a chart from model results. Returns PNG image bytes.",
    parameters=[
        ToolParameter(
            name="chart_type",
            type="string",
            description="Type of chart to generate.",
            required=False,
            default="contributions",
            enum=["contributions", "saturation", "budget_comparison", "roas"],
        ),
        ToolParameter(
            name="channel",
            type="string",
            description="Specific channel for saturation chart. Omit for all channels.",
            required=False,
        ),
    ],
    returns_description="Chart image bytes and metadata.",
    agent_scope=["reporter"],
)

CREATE_ACTION_PLAN_SCHEMA = ToolSchema(
    name="create_action_plan",
    description="Create a prioritized action plan from model insights and optimization results. Generates actionable recommendations ranked by impact and effort.",
    parameters=[],
    returns_description="Markdown action plan with prioritized recommendations.",
    agent_scope=["reporter"],
)


def generate_markdown_report(state: Any, **params: Any) -> dict[str, Any]:
    """
    Generate a full executive report in Markdown from current state.

    Reads model results, optimization results, and EDA from state,
    then renders the executive-report.md template.
    """
    try:
        from jinja2 import Template
    except ImportError:
        return {
            "status": "error",
            "message": "jinja2 is required for report generation. Install with: pip install jinja2",
        }

    # Gather data from state
    model_result = _get_state(state, "model_result")
    optimal_allocation = _get_state(state, "optimal_allocation")
    eda_report = _get_state(state, "eda_report")
    scenario_results = _get_state(state, "scenario_results")

    if model_result is None:
        return {
            "status": "error",
            "message": "No model results found in state. Fit a model first.",
            "summary": "Cannot generate report: no model has been fitted yet.",
        }

    # Build template context
    context = _build_report_context(model_result, optimal_allocation, eda_report, scenario_results)

    # Load and render template
    template_path = TEMPLATES_DIR / "executive-report.md"
    template_text = template_path.read_text() if template_path.exists() else _fallback_template()

    try:
        template = Template(template_text, undefined=_SilentUndefined)
        report = template.render(**context)
    except Exception as e:
        logger.error("Template rendering failed: %s", e)
        report = _generate_plain_report(context)

    # Store in state
    _set_state(state, "executive_report", report, "reporter")

    return {
        "status": "success",
        "summary": f"Executive report generated ({len(report)} chars, {report.count(chr(10))} lines)",
        "report": report,
    }


def generate_chart(state: Any, **params: Any) -> dict[str, Any]:
    """
    Generate a chart from model results.

    Supported chart_type values:
    - "contributions": Channel contribution waterfall/bar chart
    - "saturation": Saturation curves for all or specified channels
    - "budget_comparison": Current vs optimal budget allocation
    - "roas": ROAS comparison bar chart
    """
    chart_type = params.get("chart_type", "contributions")

    model_result = _get_state(state, "model_result")
    if model_result is None:
        return {
            "status": "error",
            "message": "No model results in state. Fit a model first.",
            "summary": "Cannot generate chart without a fitted model.",
        }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "status": "error",
            "message": "matplotlib required for charts. Install with: pip install matplotlib",
        }

    try:
        if chart_type == "contributions":
            fig = _chart_contributions(model_result, plt)
        elif chart_type == "saturation":
            fitted_model = _get_state(state, "fitted_model")
            channel = params.get("channel")
            fig = _chart_saturation(fitted_model, channel, plt)
        elif chart_type == "budget_comparison":
            allocation = _get_state(state, "optimal_allocation")
            fig = _chart_budget_comparison(allocation, plt)
        elif chart_type == "roas":
            fig = _chart_roas(model_result, plt)
        else:
            return {
                "status": "error",
                "message": f"Unknown chart type: '{chart_type}'. Use: contributions, saturation, budget_comparison, roas",
            }

        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        return {
            "status": "success",
            "summary": f"Generated {chart_type} chart",
            "chart_type": chart_type,
            "chart_bytes": buf.getvalue(),
        }

    except Exception as e:
        logger.error("Chart generation failed: %s", e)
        return {
            "status": "error",
            "message": f"Chart generation failed: {e}",
            "summary": f"Failed to generate {chart_type} chart: {e}",
        }


def create_action_plan(state: Any, **params: Any) -> dict[str, Any]:
    """
    Create a prioritized action plan from optimization results.

    Generates actionable recommendations based on model insights,
    budget optimization, and saturation analysis.
    """
    model_result = _get_state(state, "model_result")
    optimal_allocation = _get_state(state, "optimal_allocation")

    if model_result is None:
        return {
            "status": "error",
            "message": "No model results found. Fit a model first.",
            "summary": "Cannot create action plan without model results.",
        }

    actions = []

    # Analyze channel performance
    if hasattr(model_result, "channel_roas"):
        roas = model_result.channel_roas
        channel_share = getattr(model_result, "channel_share", {})

        # Find underperforming channels (low ROAS, high share)
        for ch, r in sorted(roas.items(), key=lambda x: x[1]):
            share = channel_share.get(ch, 0)
            if r < 1.0:
                actions.append(
                    {
                        "title": f"Review {ch} spend",
                        "description": f"ROAS of {r:.2f}x is below breakeven. Currently {share * 100:.1f}% of effect share.",
                        "impact": "High",
                        "effort": "Low",
                        "timeline": "Immediate",
                        "priority": 1,
                    }
                )

        # Find high-performing channels that could take more budget
        for ch, r in sorted(roas.items(), key=lambda x: x[1], reverse=True):
            if r > 3.0:
                actions.append(
                    {
                        "title": f"Explore scaling {ch}",
                        "description": f"ROAS of {r:.2f}x suggests room for growth. Check saturation curve before scaling.",
                        "impact": "High",
                        "effort": "Medium",
                        "timeline": "Next sprint",
                        "priority": 2,
                    }
                )
                break

    # Budget reallocation recommendations
    if optimal_allocation and hasattr(optimal_allocation, "allocation"):
        prev = getattr(optimal_allocation, "previous_allocation", None)
        if prev:
            for ch, opt_spend in optimal_allocation.allocation.items():
                prev_spend = prev.get(ch, 0)
                if prev_spend > 0:
                    change_pct = ((opt_spend - prev_spend) / prev_spend) * 100
                    if abs(change_pct) > 15:
                        direction = "Increase" if change_pct > 0 else "Decrease"
                        actions.append(
                            {
                                "title": f"{direction} {ch} by {abs(change_pct):.0f}%",
                                "description": f"Shift from ${prev_spend:,.0f} to ${opt_spend:,.0f}/period.",
                                "impact": "Medium",
                                "effort": "Low",
                                "timeline": "Next budget cycle",
                                "priority": 2,
                            }
                        )

        lift_pct = getattr(optimal_allocation, "expected_lift_pct", None)
        if lift_pct:
            actions.insert(
                0,
                {
                    "title": "Implement optimized budget allocation",
                    "description": f"Rebalancing budget is expected to yield +{lift_pct:.1f}% lift.",
                    "impact": "High",
                    "effort": "Medium",
                    "timeline": "Next budget cycle",
                    "priority": 1,
                },
            )

    # Sort by priority
    actions.sort(key=lambda a: a.get("priority", 99))

    # Format as Markdown
    plan_lines = ["# Action Plan", "", "## Priority Recommendations", ""]
    for i, action in enumerate(actions, 1):
        plan_lines.extend(
            [
                f"### {i}. {action['title']}",
                "",
                action["description"],
                "",
                f"- **Impact:** {action['impact']}",
                f"- **Effort:** {action['effort']}",
                f"- **Timeline:** {action['timeline']}",
                "",
            ]
        )

    plan_text = "\n".join(plan_lines)
    _set_state(state, "action_plan", plan_text, "reporter")

    return {
        "status": "success",
        "summary": f"Action plan created with {len(actions)} recommendations",
        "action_plan": plan_text,
        "actions_count": len(actions),
    }


# --- Helpers ---


def _get_state(state: Any, key: str) -> Any:
    """Get a value from state (supports both dict and SharedState)."""
    if hasattr(state, "get"):
        return state.get(key)
    if isinstance(state, dict):
        return state.get(key)
    return None


def _set_state(state: Any, key: str, value: Any, agent: str) -> None:
    """Set a value in state (supports both dict and SharedState)."""
    if hasattr(state, "set"):
        try:
            state.set(key, value, source_agent=agent)
        except TypeError:
            state[key] = value
    elif isinstance(state, dict):
        state[key] = value


def _build_report_context(
    model_result: Any,
    optimal_allocation: Any,
    eda_report: Any,
    scenario_results: Any,
) -> dict[str, Any]:
    """Build Jinja2 template context from state objects."""
    ctx: dict[str, Any] = {
        "generated_date": datetime.now().strftime("%Y-%m-%d"),
        "optmix_version": "0.1.0",
    }

    # Model info
    if model_result and hasattr(model_result, "model_type"):
        ctx["model_type"] = model_result.model_type
        ctx["n_observations"] = model_result.n_observations
        ctx["target_variable"] = model_result.target_variable
        ctx["n_channels"] = len(model_result.channels)
        ctx["r_squared"] = f"{model_result.r_squared:.3f}" if model_result.r_squared else "N/A"
        ctx["mape"] = f"{model_result.mape:.1f}" if model_result.mape else "N/A"
        ctx["date_range"] = f"{model_result.n_observations} periods"

        # Channel details
        channels = []
        for ch in model_result.channels:
            roas = model_result.channel_roas.get(ch, 0)
            share = model_result.channel_share.get(ch, 0)
            channels.append(
                {
                    "name": ch,
                    "spend": "—",
                    "contribution": "—",
                    "roas": f"{roas:.2f}",
                    "share": f"{share * 100:.1f}",
                }
            )
        ctx["channels"] = channels

        # Estimation method (dynamic based on model type)
        if model_result.model_type == "BayesianMMM":
            ctx["estimation_method"] = "Bayesian MCMC (PyMC-Marketing)"
            ctx["adstock_type"] = "Geometric (learned)"
            ctx["saturation_type"] = "Logistic (learned)"
            ctx["validation_approach"] = "R², MAPE, Rhat convergence, HDI credible intervals"
        else:
            ctx["estimation_method"] = "Ridge Regression (L2 regularization)"
            ctx["adstock_type"] = "Geometric"
            ctx["saturation_type"] = "Hill"
            ctx["validation_approach"] = "R², MAPE, residual analysis"

    # Optimization
    if optimal_allocation and hasattr(optimal_allocation, "allocation"):
        ctx["total_budget"] = f"{optimal_allocation.total_budget:,.0f}"
        lift_pct = getattr(optimal_allocation, "expected_lift_pct", None)
        ctx["expected_lift"] = f"+{lift_pct:.1f}%" if lift_pct else "N/A"
        ctx["confidence_interval"] = str(getattr(optimal_allocation, "confidence_interval", "N/A"))

        allocation = []
        for ch, opt in optimal_allocation.allocation.items():
            prev = (optimal_allocation.previous_allocation or {}).get(ch, 0)
            change = ((opt - prev) / prev * 100) if prev > 0 else 0
            mroas = optimal_allocation.channel_marginal_roas.get(ch, 0)
            allocation.append(
                {
                    "name": ch,
                    "current": f"{prev:,.0f}",
                    "optimal": f"{opt:,.0f}",
                    "change": f"{change:+.1f}%",
                    "mroas": f"{mroas:.2f}",
                }
            )
        ctx["allocation"] = allocation

    # Saturation analysis
    saturated = []
    growth = []
    if optimal_allocation and hasattr(optimal_allocation, "channel_saturation_pct"):
        for ch, pct in optimal_allocation.channel_saturation_pct.items():
            entry = {"name": ch, "saturation_pct": f"{pct:.0f}"}
            if pct > 70:
                entry["recommendation"] = "Consider reducing spend or diversifying"
                saturated.append(entry)
            else:
                entry["recommendation"] = "Room for incremental spend"
                growth.append(entry)
    ctx["saturated_channels"] = saturated
    ctx["growth_channels"] = growth

    # Placeholders for LLM-generated content
    ctx["executive_summary"] = params_or_default(
        model_result, optimal_allocation, "Model fitted successfully. See details below."
    )
    ctx["channel_findings"] = "See channel performance table above for detailed metrics."
    ctx["saturation_analysis"] = (
        "Saturation analysis identifies channels approaching diminishing returns."
    )
    ctx["recommendations"] = "Based on the model results, the following actions are recommended."
    ctx["scenarios"] = []
    ctx["priority_actions"] = []

    return ctx


def params_or_default(model: Any, opt: Any, default: str) -> str:
    """Generate a basic executive summary from available data."""
    parts = []
    if model and hasattr(model, "r_squared") and model.r_squared:
        parts.append(
            f"The marketing mix model explains {model.r_squared * 100:.1f}% of variance in {model.target_variable}."
        )
    if model and hasattr(model, "channels"):
        parts.append(f"{len(model.channels)} channels were analyzed.")
    if opt and hasattr(opt, "expected_lift_pct") and opt.expected_lift_pct:
        parts.append(
            f"Budget optimization suggests a potential +{opt.expected_lift_pct:.1f}% improvement."
        )
    return " ".join(parts) if parts else default


def _generate_plain_report(context: dict[str, Any]) -> str:
    """Fallback plain-text report if template rendering fails."""
    lines = [
        "# Marketing Mix Model — Executive Report",
        "",
        f"Generated: {context.get('generated_date', 'N/A')}",
        "",
        "## Model Overview",
        f"- Type: {context.get('model_type', 'N/A')}",
        f"- R²: {context.get('r_squared', 'N/A')}",
        f"- MAPE: {context.get('mape', 'N/A')}%",
        f"- Channels: {context.get('n_channels', 'N/A')}",
        "",
        "## Channel Performance",
    ]
    for ch in context.get("channels", []):
        lines.append(f"- {ch['name']}: ROAS {ch['roas']}x, {ch['share']}% share")
    return "\n".join(lines)


def _fallback_template() -> str:
    """Minimal template if executive-report.md is not found."""
    return """# Marketing Mix Model Report
Generated: {{ generated_date }}

## Model: {{ model_type }}
- R²: {{ r_squared }}
- MAPE: {{ mape }}%
- Channels: {{ n_channels }}

{{ executive_summary }}
"""


# --- Chart helpers ---


def _chart_contributions(model_result: Any, plt: Any) -> Any:
    """Bar chart of channel contributions."""
    channels = model_result.channels
    shares = [model_result.channel_share.get(ch, 0) * 100 for ch in channels]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(channels, shares, color="#4A90D9")
    ax.set_xlabel("Share of Marketing Effect (%)")
    ax.set_title("Channel Contribution Share")
    ax.invert_yaxis()

    for bar, val in zip(bars, shares, strict=True):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def _chart_saturation(fitted_model: Any, channel: str | None, plt: Any) -> Any:
    """Saturation curve plot."""
    if fitted_model is None:
        raise ValueError("No fitted model available for saturation chart")

    curves = fitted_model.get_saturation_curves(channel=channel)

    fig, ax = plt.subplots(figsize=(10, 6))
    for ch_name, curve_df in curves.items():
        ax.plot(curve_df["spend"], curve_df["response"], label=ch_name, linewidth=2)

    ax.set_xlabel("Spend")
    ax.set_ylabel("Response (saturated)")
    ax.set_title("Channel Saturation Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _chart_budget_comparison(allocation: Any, plt: Any) -> Any:
    """Side-by-side bar chart comparing current vs optimal budget."""
    if allocation is None:
        raise ValueError("No allocation data available")

    import numpy as np

    channels = list(allocation.allocation.keys())
    optimal = [allocation.allocation[ch] for ch in channels]
    current = [(allocation.previous_allocation or {}).get(ch, 0) for ch in channels]

    x = np.arange(len(channels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    if any(v > 0 for v in current):
        ax.bar(x - width / 2, current, width, label="Current", color="#999999")
    ax.bar(x + width / 2, optimal, width, label="Optimal", color="#4A90D9")

    ax.set_ylabel("Budget ($)")
    ax.set_title("Current vs Optimal Budget Allocation")
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    return fig


def _chart_roas(model_result: Any, plt: Any) -> Any:
    """ROAS bar chart by channel."""
    channels = model_result.channels
    roas_vals = [model_result.channel_roas.get(ch, 0) for ch in channels]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#27ae60" if r >= 1 else "#e74c3c" for r in roas_vals]
    bars = ax.barh(channels, roas_vals, color=colors)
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1, label="Breakeven")
    ax.set_xlabel("ROAS")
    ax.set_title("Return on Ad Spend by Channel")
    ax.invert_yaxis()
    ax.legend()

    for bar, val in zip(bars, roas_vals, strict=True):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}x",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


class _SilentUndefined:
    """Jinja2 undefined that returns empty strings instead of raising."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __str__(self) -> str:
        return ""

    def __iter__(self) -> Any:
        return iter([])

    def __bool__(self) -> bool:
        return False

    def __getattr__(self, name: str) -> _SilentUndefined:
        return _SilentUndefined()


REPORT_TOOL_SCHEMAS: list[tuple[ToolSchema, Any]] = [
    (GENERATE_REPORT_SCHEMA, generate_markdown_report),
    (GENERATE_CHART_SCHEMA, generate_chart),
    (CREATE_ACTION_PLAN_SCHEMA, create_action_plan),
]
