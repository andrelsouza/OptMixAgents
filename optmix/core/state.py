"""
Shared State Manager for OptMix agents.

Provides a thread-safe in-memory state store that connects all agents
during a session. When one agent produces output (e.g., the Modeler fits
a model), other agents can read it (e.g., the Optimizer reads the fitted
model to optimize budget).

Each write is attributed to a source agent and tracked in an append-only
history log for full auditability.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class StateEntry:
    """A single entry in the shared state store."""

    value: Any
    source_agent: str | None
    timestamp: datetime
    value_type: str  # e.g. "DataFrame", "ModelResult", "dict"
    summary: str  # human-readable one-liner


def _summarize_value(value: Any) -> tuple[str, str]:
    """Return (value_type, summary) for an arbitrary value.

    Inspects values at runtime to avoid heavy imports at module level.
    """
    cls_name = type(value).__name__
    cls_qualname = f"{type(value).__module__}.{cls_name}"

    # pandas DataFrame
    if cls_qualname.startswith("pandas") and cls_name == "DataFrame":
        cols = list(value.columns)
        cols_preview = ", ".join(cols[:5])
        if len(cols) > 5:
            cols_preview += ", ..."
        return (
            "DataFrame",
            f"DataFrame with {len(value)} rows, {len(cols)} columns: [{cols_preview}]",
        )

    # ModelResult (optmix.mmm.models.base)
    if cls_name == "ModelResult" and "optmix" in cls_qualname:
        r2 = getattr(value, "r_squared", None)
        n_obs = getattr(value, "n_observations", "?")
        channels = getattr(value, "channels", [])
        model_type = getattr(value, "model_type", "MMM")
        r2_str = f"R\u00b2={r2:.3f}" if r2 is not None else "R\u00b2=N/A"
        ch_str = ", ".join(channels[:6])
        if len(channels) > 6:
            ch_str += ", ..."
        return (
            "ModelResult",
            f"{model_type}: {r2_str}, {n_obs} observations, channels: [{ch_str}]",
        )

    # OptimizationResult (optmix.mmm.models.base)
    if cls_name == "OptimizationResult" and "optmix" in cls_qualname:
        allocation = getattr(value, "allocation", {})
        total = getattr(value, "total_budget", 0.0)
        lift_pct = getattr(value, "expected_lift_pct", None)
        lift_str = f", lift: {lift_pct:.1f}%" if lift_pct is not None else ""
        return (
            "OptimizationResult",
            f"Budget ${total:,.0f} across {len(allocation)} channels{lift_str}",
        )

    # dict
    if isinstance(value, dict):
        keys = list(value.keys())
        keys_preview = ", ".join(str(k) for k in keys[:5])
        if len(keys) > 5:
            keys_preview += ", ..."
        return "dict", f"dict with {len(keys)} keys: [{keys_preview}]"

    # str
    if isinstance(value, str):
        preview = value[:100]
        if len(value) > 100:
            preview += "..."
        return "str", preview

    # list / tuple
    if isinstance(value, (list, tuple)):
        type_label = cls_name
        return type_label, f"{type_label} with {len(value)} items"

    # numeric scalars
    if isinstance(value, (int, float)):
        return cls_name, repr(value)

    # fallback
    raw_repr = repr(value)
    if len(raw_repr) > 80:
        raw_repr = raw_repr[:80] + "..."
    return cls_name, f"{cls_name}: {raw_repr}"


class SharedState:
    """Thread-safe shared state that connects all OptMix agents.

    Example::

        state = SharedState()
        state.set("raw_data", df, source_agent="analyst")
        state.set("fitted_model", model_result, source_agent="modeler")

        model = state.get("fitted_model")

        ctx = state.get_context_for_agent("optimizer", relevant_keys=[
            "fitted_model", "model_result", "raw_data",
        ])
    """

    KNOWN_SLOTS: list[str] = [
        "market_context",
        "kpi_framework",
        "data_assessment",
        "raw_data",
        "validated_data",
        "model_ready_data",
        "eda_report",
        "collinearity_report",
        "fitted_model",
        "model_result",
        "diagnostics_report",
        "channel_contributions",
        "optimal_allocation",
        "marginal_roas",
        "scenario_results",
        "executive_report",
        "action_plan",
    ]

    def __init__(self) -> None:
        self._store: dict[str, StateEntry] = {}
        self._history: dict[str, list[tuple[datetime, str | None, str]]] = {}
        self._lock = threading.Lock()

    def set(
        self,
        key: str,
        value: Any,
        source_agent: str | None = None,
    ) -> None:
        """Store a value with automatic summary generation and history tracking."""
        value_type, summary = _summarize_value(value)
        now = datetime.now(timezone.utc)
        entry = StateEntry(
            value=value,
            source_agent=source_agent,
            timestamp=now,
            value_type=value_type,
            summary=summary,
        )
        with self._lock:
            self._store[key] = entry
            self._history.setdefault(key, []).append(
                (now, source_agent, summary),
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve the current value for key, or default if absent."""
        with self._lock:
            entry = self._store.get(key)
        if entry is None:
            return default
        return entry.value

    def get_entry(self, key: str) -> StateEntry | None:
        """Retrieve the full StateEntry for key, or None."""
        with self._lock:
            return self._store.get(key)

    def has(self, key: str) -> bool:
        """Return True if key is populated."""
        with self._lock:
            return key in self._store

    def keys(self) -> list[str]:
        """Return a list of all populated slot names."""
        with self._lock:
            return list(self._store.keys())

    def history(self, key: str) -> list[tuple[datetime, str | None, str]]:
        """Return the write history for key.

        Each entry is (timestamp, source_agent, value_summary).
        """
        with self._lock:
            return list(self._history.get(key, []))

    def clear(self) -> None:
        """Reset all state and history."""
        with self._lock:
            self._store.clear()
            self._history.clear()

    def get_context_for_agent(
        self,
        agent_name: str,
        relevant_keys: list[str] | None = None,
    ) -> str:
        """Serialize relevant state into a text block for an LLM system prompt."""
        with self._lock:
            snapshot = dict(self._store)

        if relevant_keys is not None:
            target_keys = [k for k in relevant_keys if k in snapshot]
        else:
            target_keys = list(snapshot.keys())

        if not target_keys:
            return (
                f"[Shared State for {agent_name}]\n"
                "No relevant state is available yet."
            )

        lines: list[str] = [
            f"[Shared State for {agent_name}]",
            f"The following {len(target_keys)} state entries are available:",
            "",
        ]
        for key in target_keys:
            entry = snapshot[key]
            source = entry.source_agent or "system"
            ts = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
            lines.append(f"- **{key}** ({entry.value_type}, set by {source} at {ts})")
            lines.append(f"  {entry.summary}")

        return "\n".join(lines)

    def to_summary(self) -> str:
        """Return a human-readable summary of all populated state."""
        with self._lock:
            snapshot = dict(self._store)

        if not snapshot:
            return "SharedState: empty (no entries)"

        lines: list[str] = [
            f"SharedState: {len(snapshot)} entries",
            "-" * 50,
        ]
        for key, entry in snapshot.items():
            source = entry.source_agent or "system"
            ts = entry.timestamp.strftime("%H:%M:%S")
            lines.append(f"  {key:.<30s} [{entry.value_type}] (by {source} @ {ts})")
            lines.append(f"    {entry.summary}")
        return "\n".join(lines)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __repr__(self) -> str:
        with self._lock:
            n = len(self._store)
        slots = ", ".join(self.keys()[:5])
        if n > 5:
            slots += ", ..."
        return f"SharedState({n} entries: [{slots}])"
