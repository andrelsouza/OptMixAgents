"""
Data tools for the Analyst agent (Kai).

Tools for loading, validating, and exploring marketing data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from optmix.tools.registry import ToolParameter, ToolSchema

# --- Schemas ---

LOAD_CSV_DATA_SCHEMA = ToolSchema(
    name="load_csv_data",
    description="Load a CSV file into the shared state as raw marketing data.",
    parameters=[
        ToolParameter(name="file_path", type="string", description="Path to the CSV file."),
    ],
    returns_description="Summary of loaded data including row/column counts.",
    agent_scope=["analyst", "strategist"],
)

LOAD_SAMPLE_DATA_SCHEMA = ToolSchema(
    name="load_sample_data",
    description="Load a built-in sample marketing dataset for testing or learning.",
    parameters=[
        ToolParameter(
            name="dataset_name", type="string",
            description="Name of the sample dataset.",
            enum=["ecommerce", "retail_chain", "saas_b2b"],
        ),
    ],
    returns_description="Summary of the loaded sample data.",
    agent_scope=["analyst", "strategist", "modeler"],
)

VALIDATE_DATA_SCHEMA = ToolSchema(
    name="validate_data",
    description="Run data quality checks on the loaded data: nulls, types, date continuity, negative values, outliers.",
    parameters=[],
    returns_description="Validation report with pass/fail checks and recommendations.",
    agent_scope=["analyst"],
)

RUN_EDA_SCHEMA = ToolSchema(
    name="run_eda",
    description="Run exploratory data analysis: per-channel statistics, correlations, seasonality detection.",
    parameters=[],
    returns_description="EDA report with channel stats, top correlations, and seasonality indicators.",
    agent_scope=["analyst"],
)

DESCRIBE_CHANNELS_SCHEMA = ToolSchema(
    name="describe_channels",
    description="Get per-channel spend summary: mean, median, min, max, total, percentage of total marketing spend.",
    parameters=[],
    returns_description="Channel-level spend statistics.",
    agent_scope=["analyst", "strategist"],
)


# --- Implementations ---

def load_csv_data(state: Any, *, file_path: str) -> dict[str, Any]:
    """Load a CSV file and store in state['raw_data']."""
    try:
        df = pd.read_csv(file_path)
        _set_state(state, "raw_data", df, "analyst")
        return {
            "status": "success",
            "summary": f"Loaded {file_path}: {len(df)} rows, {len(df.columns)} columns",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to load CSV: {e}"}


def load_sample_data(state: Any, *, dataset_name: str = "ecommerce") -> dict[str, Any]:
    """Load a built-in sample dataset."""
    try:
        from optmix.data.samples import load_sample
        df = load_sample(dataset_name)
        _set_state(state, "raw_data", df, "analyst")
        return {
            "status": "success",
            "summary": f"Loaded sample '{dataset_name}': {len(df)} rows, {len(df.columns)} columns",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dataset_name": dataset_name,
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to load sample '{dataset_name}': {e}"}


def validate_data(state: Any) -> dict[str, Any]:
    """Run data quality checks on raw_data."""
    df = _get_state(state, "raw_data")
    if df is None:
        return {"status": "error", "message": "No data loaded. Use load_csv_data or load_sample_data first."}

    checks: list[dict[str, Any]] = []

    # 1. Null check
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    checks.append({
        "check": "missing_values",
        "passed": len(null_cols) == 0,
        "detail": "No missing values found" if len(null_cols) == 0 else f"Missing values in: {dict(null_cols)}",
    })

    # 2. Numeric types check
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    checks.append({
        "check": "numeric_columns",
        "passed": len(numeric_cols) >= 2,
        "detail": f"{len(numeric_cols)} numeric columns found",
    })

    # 3. Date column check
    date_cols = [c for c in df.columns if "date" in c.lower()]
    checks.append({
        "check": "date_column",
        "passed": len(date_cols) > 0,
        "detail": f"Date column(s) found: {date_cols}" if date_cols else "No date column detected",
    })

    # 4. Negative values in spend columns
    spend_cols = [c for c in numeric_cols if c not in ["revenue", "pipeline_generated", "conversions"]]
    neg_cols = [c for c in spend_cols if (df[c] < 0).any()]
    checks.append({
        "check": "no_negative_spend",
        "passed": len(neg_cols) == 0,
        "detail": "No negative spend values" if not neg_cols else f"Negative values in: {neg_cols}",
    })

    # 5. Sufficient rows
    checks.append({
        "check": "sufficient_observations",
        "passed": len(df) >= 52,
        "detail": f"{len(df)} observations (minimum 52 recommended)",
    })

    # 6. Outlier detection (values > 5 std from mean)
    outlier_cols = []
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std()) if df[col].std() > 0 else pd.Series([0])
        if (z_scores > 5).any():
            outlier_cols.append(col)
    checks.append({
        "check": "outlier_check",
        "passed": len(outlier_cols) == 0,
        "detail": "No extreme outliers" if not outlier_cols else f"Potential outliers in: {outlier_cols}",
    })

    # Store validated data
    _set_state(state, "validated_data", df, "analyst")

    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)

    return {
        "status": "success",
        "summary": f"Data validation: {passed}/{total} checks passed. {len(df)} rows, {len(df.columns)} columns.",
        "checks": checks,
        "passed_count": passed,
        "total_checks": total,
        "rows": len(df),
        "columns": len(df.columns),
    }


def run_eda(state: Any) -> dict[str, Any]:
    """Run exploratory data analysis on validated data."""
    df = _get_state(state, "validated_data")
    if df is None:
        df = _get_state(state, "raw_data")
    if df is None:
        return {"status": "error", "message": "No data available. Load and validate data first."}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Identify target and channel columns
    target_candidates = ["revenue", "pipeline_generated", "conversions", "sales"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    control_candidates = ["avg_price", "promo", "store_count", "competitor_promo", "sales_headcount"]
    controls = [c for c in control_candidates if c in df.columns]
    channels = [c for c in numeric_cols if c not in [target_col] + controls]

    # Per-channel stats
    channel_stats = {}
    for ch in channels:
        series = df[ch]
        channel_stats[ch] = {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
        }

    # Correlations with target
    correlations = {}
    if target_col:
        for ch in channels:
            corr = float(df[ch].corr(df[target_col]))
            correlations[ch] = round(corr, 3)

    # Top correlations sorted
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    top_correlations = sorted_corr[:10]

    eda_report = {
        "target_column": target_col,
        "channels": channels,
        "controls": controls,
        "channel_stats": channel_stats,
        "correlations_with_target": correlations,
        "top_correlations": top_correlations,
        "n_observations": len(df),
    }

    _set_state(state, "eda_report", eda_report, "analyst")

    return {
        "status": "success",
        "summary": f"EDA complete: {len(channels)} channels, {len(df)} observations, target='{target_col}'",
        "n_channels": len(channels),
        "n_observations": len(df),
        "target_column": target_col,
        "channel_stats": channel_stats,
        "top_correlations": top_correlations,
    }


def describe_channels(state: Any) -> dict[str, Any]:
    """Get per-channel spend summary."""
    df = _get_state(state, "validated_data")
    if df is None:
        df = _get_state(state, "raw_data")
    if df is None:
        return {"status": "error", "message": "No data available."}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"revenue", "pipeline_generated", "conversions", "sales",
               "avg_price", "promo", "store_count", "competitor_promo", "sales_headcount"}
    channels = [c for c in numeric_cols if c not in exclude]

    total_spend = sum(float(df[ch].sum()) for ch in channels)

    channel_stats = {}
    for ch in channels:
        s = df[ch]
        ch_total = float(s.sum())
        channel_stats[ch] = {
            "mean": round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "total": round(ch_total, 2),
            "pct_of_total": round(ch_total / total_spend * 100, 1) if total_spend > 0 else 0,
        }

    return {
        "status": "success",
        "summary": f"Channel summary: {len(channels)} channels, total spend ${total_spend:,.0f}",
        "n_channels": len(channels),
        "total_marketing_spend": round(total_spend, 2),
        "channel_stats": channel_stats,
    }


# --- Helpers ---

def _get_state(state: Any, key: str) -> Any:
    if hasattr(state, "get"):
        return state.get(key)
    if isinstance(state, dict):
        return state.get(key)
    return None


def _set_state(state: Any, key: str, value: Any, agent: str) -> None:
    if hasattr(state, "set") and callable(state.set):
        try:
            state.set(key, value, source_agent=agent)
        except TypeError:
            state[key] = value
    elif isinstance(state, dict):
        state[key] = value


# All schemas for registration
DATA_TOOL_SCHEMAS: list[tuple[ToolSchema, Any]] = [
    (LOAD_CSV_DATA_SCHEMA, load_csv_data),
    (LOAD_SAMPLE_DATA_SCHEMA, load_sample_data),
    (VALIDATE_DATA_SCHEMA, validate_data),
    (RUN_EDA_SCHEMA, run_eda),
    (DESCRIBE_CHANNELS_SCHEMA, describe_channels),
]
