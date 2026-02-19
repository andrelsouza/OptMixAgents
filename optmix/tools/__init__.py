"""
OptMix Tools — callable functions for AI marketing agents.

Exposes the ToolRegistry and a factory to create the default registry
pre-populated with all built-in tools.
"""

from __future__ import annotations

from optmix.tools.registry import ToolParameter, ToolRegistry, ToolSchema


def create_default_registry() -> ToolRegistry:
    """Create a ToolRegistry pre-populated with all built-in tools."""
    from optmix.tools.data_tools import DATA_TOOL_SCHEMAS
    from optmix.tools.mmm_tools import MMM_TOOL_SCHEMAS
    from optmix.tools.optimization_tools import OPTIMIZATION_TOOL_SCHEMAS
    from optmix.tools.report_tools import REPORT_TOOL_SCHEMAS
    from optmix.tools.strategy_tools import STRATEGY_TOOL_SCHEMAS

    registry = ToolRegistry()

    all_tools = (
        DATA_TOOL_SCHEMAS
        + MMM_TOOL_SCHEMAS
        + OPTIMIZATION_TOOL_SCHEMAS
        + REPORT_TOOL_SCHEMAS
        + STRATEGY_TOOL_SCHEMAS
    )

    for schema, func in all_tools:
        registry.register(schema, func)

    return registry


__all__ = [
    "ToolParameter",
    "ToolRegistry",
    "ToolSchema",
    "create_default_registry",
]
