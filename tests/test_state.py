"""Tests for the SharedState manager."""

from __future__ import annotations

import pandas as pd
import pytest

from optmix.core.state import SharedState
from optmix.mmm.models.base import ModelResult, OptimizationResult


@pytest.fixture
def state() -> SharedState:
    return SharedState()


class TestBasicCRUD:

    def test_set_and_get(self, state: SharedState) -> None:
        state.set("raw_data", {"rows": 100}, source_agent="analyst")
        assert state.get("raw_data") == {"rows": 100}

    def test_get_missing_returns_default(self, state: SharedState) -> None:
        assert state.get("nonexistent") is None
        assert state.get("nonexistent", "fallback") == "fallback"

    def test_has(self, state: SharedState) -> None:
        state.set("market_context", "bull market")
        assert state.has("market_context") is True
        assert state.has("missing") is False

    def test_contains(self, state: SharedState) -> None:
        state.set("kpi", {"goal": "CAC"})
        assert "kpi" in state
        assert "missing" not in state

    def test_keys(self, state: SharedState) -> None:
        assert state.keys() == []
        state.set("a", 1)
        state.set("b", 2)
        assert sorted(state.keys()) == ["a", "b"]

    def test_len(self, state: SharedState) -> None:
        assert len(state) == 0
        state.set("x", 42)
        assert len(state) == 1

    def test_overwrite(self, state: SharedState) -> None:
        state.set("raw_data", "v1")
        state.set("raw_data", "v2")
        assert state.get("raw_data") == "v2"


class TestHistory:

    def test_empty_history(self, state: SharedState) -> None:
        assert state.history("nonexistent") == []

    def test_tracks_writes(self, state: SharedState) -> None:
        state.set("raw_data", "v1", source_agent="analyst")
        state.set("raw_data", "v2", source_agent="modeler")
        hist = state.history("raw_data")
        assert len(hist) == 2
        assert hist[0][1] == "analyst"
        assert hist[1][1] == "modeler"

    def test_timestamps_ordered(self, state: SharedState) -> None:
        state.set("data", "a")
        state.set("data", "b")
        hist = state.history("data")
        assert hist[0][0] <= hist[1][0]


class TestAutoSummary:

    def test_dataframe_summary(self, state: SharedState) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        state.set("raw_data", df)
        entry = state.get_entry("raw_data")
        assert entry.value_type == "DataFrame"
        assert "2 rows" in entry.summary
        assert "3 columns" in entry.summary

    def test_dict_summary(self, state: SharedState) -> None:
        state.set("config", {"a": 1, "b": 2})
        entry = state.get_entry("config")
        assert entry.value_type == "dict"
        assert "2 keys" in entry.summary

    def test_string_summary(self, state: SharedState) -> None:
        state.set("note", "short")
        entry = state.get_entry("note")
        assert entry.value_type == "str"
        assert entry.summary == "short"

    def test_long_string_truncated(self, state: SharedState) -> None:
        state.set("essay", "A" * 200)
        entry = state.get_entry("essay")
        assert entry.summary.endswith("...")

    def test_model_result_summary(self, state: SharedState) -> None:
        result = ModelResult(
            model_type="RidgeMMM", target_variable="revenue",
            date_column="date", channels=["tv", "search"], n_observations=104,
            r_squared=0.87,
        )
        state.set("model", result)
        entry = state.get_entry("model")
        assert entry.value_type == "ModelResult"
        assert "0.870" in entry.summary
        assert "104" in entry.summary

    def test_optimization_result_summary(self, state: SharedState) -> None:
        result = OptimizationResult(
            allocation={"tv": 50000, "search": 30000},
            total_budget=80000, expected_lift_pct=12.5,
        )
        state.set("alloc", result)
        entry = state.get_entry("alloc")
        assert entry.value_type == "OptimizationResult"
        assert "2 channels" in entry.summary
        assert "12.5%" in entry.summary


class TestContextForAgent:

    def test_empty_state(self, state: SharedState) -> None:
        ctx = state.get_context_for_agent("optimizer")
        assert "No relevant state" in ctx

    def test_with_data(self, state: SharedState) -> None:
        state.set("raw_data", {"rows": 100}, source_agent="analyst")
        ctx = state.get_context_for_agent("modeler")
        assert "raw_data" in ctx
        assert "analyst" in ctx

    def test_filtered_keys(self, state: SharedState) -> None:
        state.set("raw_data", "d")
        state.set("fitted_model", "m")
        ctx = state.get_context_for_agent("opt", relevant_keys=["fitted_model"])
        assert "fitted_model" in ctx
        assert "raw_data" not in ctx


class TestClear:

    def test_clears_everything(self, state: SharedState) -> None:
        state.set("a", 1)
        state.set("b", 2)
        state.clear()
        assert len(state) == 0
        assert state.keys() == []
        assert state.history("a") == []
        assert state.get("a") is None


class TestToSummary:

    def test_empty(self, state: SharedState) -> None:
        assert "empty" in state.to_summary()

    def test_populated(self, state: SharedState) -> None:
        state.set("raw_data", {"r": 1}, source_agent="analyst")
        s = state.to_summary()
        assert "1 entries" in s
        assert "raw_data" in s
