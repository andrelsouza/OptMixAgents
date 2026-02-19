"""
Sample datasets for learning and testing OptMix.

Generates realistic synthetic marketing data with known ground truth,
so you can validate that the models recover the true parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optmix.mmm.transforms.adstock import geometric_adstock
from optmix.mmm.transforms.saturation import hill_saturation


def load_sample(name: str = "ecommerce") -> pd.DataFrame:
    """
    Load a sample marketing dataset.

    Args:
        name: Dataset name. Options:
            - "ecommerce": DTC ecommerce with 8 channels, 104 weeks
            - "retail_chain": Brick-and-mortar with TV/Radio/OOH, 156 weeks
            - "saas_b2b": B2B SaaS with long sales cycles, 104 weeks

    Returns:
        DataFrame with date, channel spends, controls, and target.
    """
    generators = {
        "ecommerce": _generate_ecommerce,
        "retail_chain": _generate_retail,
        "saas_b2b": _generate_saas,
    }

    if name not in generators:
        available = ", ".join(generators.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    return generators[name]()


def _generate_ecommerce(seed: int = 42) -> pd.DataFrame:
    """Generate a DTC ecommerce dataset with 8 channels."""
    rng = np.random.default_rng(seed)
    n_weeks = 104

    dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")

    # Channel spends (weekly, in USD)
    channels = {
        "google_search": rng.lognormal(10.5, 0.3, n_weeks),
        "google_shopping": rng.lognormal(10.0, 0.4, n_weeks),
        "meta_ads": rng.lognormal(10.8, 0.35, n_weeks),
        "tiktok_ads": rng.lognormal(9.5, 0.5, n_weeks),
        "youtube": rng.lognormal(9.8, 0.3, n_weeks),
        "email": rng.lognormal(8.0, 0.2, n_weeks),
        "affiliate": rng.lognormal(9.0, 0.4, n_weeks),
        "display": rng.lognormal(9.5, 0.3, n_weeks),
    }

    # True parameters (ground truth for validation)
    true_params = {
        "google_search": {"decay": 0.2, "half_sat": 40000, "slope": 1.5, "coef": 2.5},
        "google_shopping": {"decay": 0.15, "half_sat": 30000, "slope": 1.2, "coef": 2.0},
        "meta_ads": {"decay": 0.4, "half_sat": 60000, "slope": 1.8, "coef": 1.8},
        "tiktok_ads": {"decay": 0.3, "half_sat": 20000, "slope": 2.0, "coef": 1.5},
        "youtube": {"decay": 0.6, "half_sat": 25000, "slope": 1.3, "coef": 1.2},
        "email": {"decay": 0.1, "half_sat": 5000, "slope": 1.0, "coef": 3.0},
        "affiliate": {"decay": 0.1, "half_sat": 15000, "slope": 1.5, "coef": 2.2},
        "display": {"decay": 0.5, "half_sat": 20000, "slope": 1.4, "coef": 0.8},
    }

    # Generate target (revenue) from true model
    base_revenue = 150_000  # Base weekly revenue without marketing
    week_idx = np.arange(n_weeks)

    # Seasonality (annual cycle with Q4 spike)
    seasonality = 1 + 0.15 * np.sin(2 * np.pi * week_idx / 52) + 0.10 * np.sin(4 * np.pi * week_idx / 52)
    # Q4 holiday boost
    month = dates.month
    q4_boost = np.where((month >= 11) | (month == 1), 1.2, 1.0)
    seasonality *= q4_boost

    # Trend (slight growth)
    trend = 1 + 0.001 * week_idx

    # Channel contributions
    revenue = base_revenue * seasonality * trend
    for ch, spend in channels.items():
        p = true_params[ch]
        adstocked = geometric_adstock(spend, p["decay"])
        saturated = hill_saturation(adstocked, p["half_sat"], p["slope"])
        revenue += p["coef"] * saturated * spend.mean()

    # Add noise
    revenue += rng.normal(0, revenue * 0.05)
    revenue = np.maximum(revenue, 0)

    # Controls
    avg_price = 45 + rng.normal(0, 3, n_weeks)
    promo_flag = rng.binomial(1, 0.15, n_weeks).astype(float)

    # Build DataFrame
    df = pd.DataFrame({"date": dates})
    for ch, spend in channels.items():
        df[ch] = np.round(spend, 2)
    df["avg_price"] = np.round(avg_price, 2)
    df["promo"] = promo_flag
    df["revenue"] = np.round(revenue, 2)

    return df


def _generate_retail(seed: int = 123) -> pd.DataFrame:
    """Generate a retail chain dataset with traditional + digital channels."""
    rng = np.random.default_rng(seed)
    n_weeks = 156

    dates = pd.date_range("2023-01-01", periods=n_weeks, freq="W-MON")

    channels = {
        "tv_national": rng.lognormal(12.0, 0.2, n_weeks),
        "tv_local": rng.lognormal(10.5, 0.3, n_weeks),
        "radio": rng.lognormal(9.5, 0.25, n_weeks),
        "ooh": rng.lognormal(10.0, 0.15, n_weeks),
        "print": rng.lognormal(9.0, 0.3, n_weeks),
        "google_ads": rng.lognormal(10.5, 0.35, n_weeks),
        "meta_ads": rng.lognormal(10.0, 0.3, n_weeks),
    }

    base = 500_000
    week_idx = np.arange(n_weeks)
    seasonality = 1 + 0.2 * np.sin(2 * np.pi * week_idx / 52)
    revenue = base * seasonality + rng.normal(0, 20000, n_weeks)

    for _ch, spend in channels.items():
        contribution = spend * rng.uniform(0.5, 2.0) * 0.01
        revenue += contribution

    revenue = np.maximum(revenue, 0)

    df = pd.DataFrame({"date": dates})
    for ch, spend in channels.items():
        df[ch] = np.round(spend, 2)
    df["store_count"] = 150 + np.cumsum(rng.binomial(1, 0.02, n_weeks))
    df["competitor_promo"] = rng.binomial(1, 0.1, n_weeks).astype(float)
    df["revenue"] = np.round(revenue, 2)

    return df


def _generate_saas(seed: int = 456) -> pd.DataFrame:
    """Generate a B2B SaaS dataset with long consideration cycles."""
    rng = np.random.default_rng(seed)
    n_weeks = 104

    dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")

    channels = {
        "google_search": rng.lognormal(9.5, 0.3, n_weeks),
        "linkedin_ads": rng.lognormal(10.0, 0.4, n_weeks),
        "content_syndication": rng.lognormal(9.0, 0.3, n_weeks),
        "webinars": rng.lognormal(8.5, 0.5, n_weeks),
        "events": rng.lognormal(10.5, 0.6, n_weeks),
        "sdr_outbound": rng.lognormal(10.0, 0.2, n_weeks),
    }

    base_pipeline = 200_000
    week_idx = np.arange(n_weeks)
    trend = 1 + 0.003 * week_idx
    pipeline = base_pipeline * trend + rng.normal(0, 15000, n_weeks)

    for _ch, spend in channels.items():
        # B2B has longer adstock
        adstocked = geometric_adstock(spend, decay=0.6)
        pipeline += adstocked * rng.uniform(0.01, 0.05) * 0.1

    pipeline = np.maximum(pipeline, 0)

    df = pd.DataFrame({"date": dates})
    for ch, spend in channels.items():
        df[ch] = np.round(spend, 2)
    df["sales_headcount"] = 20 + np.cumsum(rng.binomial(1, 0.05, n_weeks))
    df["pipeline_generated"] = np.round(pipeline, 2)

    return df
