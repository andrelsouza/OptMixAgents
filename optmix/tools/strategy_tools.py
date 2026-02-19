"""
Strategy tools for the Strategist agent (Maya).

Industry benchmarks, channel taxonomy, and data readiness assessment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from optmix.tools.registry import ToolParameter, ToolSchema

KNOWLEDGE_DIR = Path(__file__).parent.parent.parent / "knowledge"

# --- Schemas ---

LOAD_INDUSTRY_BENCHMARKS_SCHEMA = ToolSchema(
    name="load_industry_benchmarks",
    description="Load industry benchmark data for ROAS, adstock rates, and model fit thresholds. Use to sanity-check model outputs.",
    parameters=[
        ToolParameter(
            name="industry",
            type="string",
            description="Industry vertical.",
            required=True,
            enum=["ecommerce", "retail", "saas_b2b"],
        ),
    ],
    returns_description="Benchmark data with typical ROAS ranges, adstock rates, and model fit thresholds.",
    agent_scope=["strategist", "modeler"],
)

LOAD_CHANNEL_TAXONOMY_SCHEMA = ToolSchema(
    name="load_channel_taxonomy",
    description="Load channel classification taxonomy with channel categories, typical characteristics, and funnel position.",
    parameters=[],
    returns_description="Channel taxonomy data.",
    agent_scope=["strategist"],
)

ASSESS_DATA_READINESS_SCHEMA = ToolSchema(
    name="assess_data_readiness",
    description="Evaluate whether the loaded data is ready for Marketing Mix Modeling. Checks data volume, channel count, spend variation, date coverage, and target variable presence.",
    parameters=[],
    returns_description="Data readiness checklist with pass/fail criteria and recommendations.",
    agent_scope=["strategist"],
)


# --- Implementations ---


def load_industry_benchmarks(state: Any, *, industry: str) -> dict[str, Any]:
    """Load industry benchmarks from knowledge YAML."""
    benchmarks_path = KNOWLEDGE_DIR / "industry-benchmarks.yaml"
    if not benchmarks_path.exists():
        return {"status": "error", "message": f"Benchmarks file not found: {benchmarks_path}"}

    try:
        with open(benchmarks_path) as f:
            raw = yaml.safe_load(f)

        benchmarks = raw.get("benchmarks", raw)

        # Extract industry-specific data
        roas = benchmarks.get("roas_ranges", {}).get(industry, {})
        adstock = benchmarks.get("adstock_ranges", {})
        base_share = benchmarks.get("base_revenue_share", {}).get(industry, {})
        model_fit = benchmarks.get("model_fit", {}).get(industry, {})

        return {
            "status": "success",
            "summary": f"Loaded benchmarks for '{industry}': {len(roas)} channel ROAS ranges, adstock rates, model fit thresholds.",
            "industry": industry,
            "roas_ranges": roas,
            "adstock_ranges": adstock,
            "base_revenue_share": base_share,
            "model_fit_thresholds": model_fit,
        }

    except Exception as e:
        return {"status": "error", "message": f"Failed to load benchmarks: {e}"}


def load_channel_taxonomy(state: Any) -> dict[str, Any]:
    """Load channel taxonomy from knowledge YAML."""
    taxonomy_path = KNOWLEDGE_DIR / "channel-taxonomy.yaml"
    if not taxonomy_path.exists():
        return {"status": "error", "message": f"Taxonomy file not found: {taxonomy_path}"}

    try:
        with open(taxonomy_path) as f:
            raw = yaml.safe_load(f)

        taxonomy = raw.get("taxonomy", raw)
        categories = list(taxonomy.keys()) if isinstance(taxonomy, dict) else []

        return {
            "status": "success",
            "summary": f"Channel taxonomy loaded with {len(categories)} categories.",
            "taxonomy": taxonomy,
            "categories": categories,
        }

    except Exception as e:
        return {"status": "error", "message": f"Failed to load taxonomy: {e}"}


def assess_data_readiness(state: Any) -> dict[str, Any]:
    """Evaluate data readiness for MMM."""
    import pandas as pd

    df = _get_state(state, "validated_data")
    if df is None:
        df = _get_state(state, "raw_data")
    if df is None:
        return {"status": "error", "message": "No data loaded. Load data first."}

    checklist: list[dict[str, Any]] = []
    import numpy as np

    # Check 1: Sufficient observations
    checklist.append(
        {
            "criterion": "observation_count",
            "passed": len(df) >= 52,
            "detail": f"{len(df)} observations ({'sufficient' if len(df) >= 52 else 'insufficient, need 52+'})",
            "recommendation": None
            if len(df) >= 52
            else "At least 52 weekly observations (1 year) recommended.",
        }
    )

    # Check 2: Channel count
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        "revenue",
        "pipeline_generated",
        "conversions",
        "sales",
        "avg_price",
        "promo",
        "store_count",
        "competitor_promo",
        "sales_headcount",
    }
    channels = [c for c in numeric_cols if c not in exclude]
    checklist.append(
        {
            "criterion": "channel_count",
            "passed": len(channels) >= 3,
            "detail": f"{len(channels)} marketing channels detected",
            "recommendation": None
            if len(channels) >= 3
            else "At least 3 channels needed for meaningful MMM.",
        }
    )

    # Check 3: Spend variation
    low_variation = []
    for col in channels:
        series = df[col]
        if series.std() == 0:
            low_variation.append(col)
            continue
        cv = series.std() / series.mean() if series.mean() > 0 else 0
        if cv < 0.1:
            low_variation.append(col)
    checklist.append(
        {
            "criterion": "spend_variation",
            "passed": len(low_variation) == 0,
            "detail": "All channels have sufficient variation"
            if not low_variation
            else f"Low variation in: {', '.join(low_variation)}",
            "recommendation": None
            if not low_variation
            else "Channels with flat spend cannot be modeled reliably.",
        }
    )

    # Check 4: Date column
    date_cols = [c for c in df.columns if "date" in c.lower()]
    checklist.append(
        {
            "criterion": "date_column",
            "passed": len(date_cols) > 0,
            "detail": f"Date column found: {date_cols[0]}"
            if date_cols
            else "No date column detected",
            "recommendation": None if date_cols else "Add a date column for time-series modeling.",
        }
    )

    # Check 5: Target variable
    target_candidates = ["revenue", "pipeline_generated", "conversions", "sales"]
    found_targets = [c for c in target_candidates if c in df.columns]
    checklist.append(
        {
            "criterion": "target_variable",
            "passed": len(found_targets) > 0,
            "detail": f"Target variable(s): {', '.join(found_targets)}"
            if found_targets
            else "No standard target found",
            "recommendation": None
            if found_targets
            else "Add a target column (revenue, conversions, etc.).",
        }
    )

    # Check 6: Date range covers at least 1 year
    sufficient_range = False
    if date_cols:
        try:
            dates = pd.to_datetime(df[date_cols[0]])
            date_span = (dates.max() - dates.min()).days
            sufficient_range = date_span >= 350
        except Exception:
            pass
    checklist.append(
        {
            "criterion": "date_range_coverage",
            "passed": sufficient_range,
            "detail": f"Date range: {date_span} days"
            if date_cols and sufficient_range
            else "Insufficient date range",
            "recommendation": None
            if sufficient_range
            else "At least 1 year of data recommended for seasonality.",
        }
    )

    passed_count = sum(1 for c in checklist if c["passed"])
    total_count = len(checklist)

    if passed_count == total_count:
        readiness = "ready"
    elif passed_count >= total_count - 1:
        readiness = "ready_with_caveats"
    else:
        readiness = "not_ready"

    return {
        "status": "success",
        "summary": f"Data readiness: {passed_count}/{total_count} criteria passed. Status: {readiness}",
        "readiness": readiness,
        "passed_count": passed_count,
        "total_count": total_count,
        "checklist": checklist,
    }


# --- Helpers ---


def _get_state(state: Any, key: str) -> Any:
    if hasattr(state, "get"):
        return state.get(key)
    if isinstance(state, dict):
        return state.get(key)
    return None


STRATEGY_TOOL_SCHEMAS: list[tuple[ToolSchema, Any]] = [
    (LOAD_INDUSTRY_BENCHMARKS_SCHEMA, load_industry_benchmarks),
    (LOAD_CHANNEL_TAXONOMY_SCHEMA, load_channel_taxonomy),
    (ASSESS_DATA_READINESS_SCHEMA, assess_data_readiness),
]
