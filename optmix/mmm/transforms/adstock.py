"""
Adstock transformations for Marketing Mix Modeling.

Adstock captures the carryover effect of advertising — the idea that an ad
seen today continues to have effect in future periods. These transforms are
applied to raw spend data before modeling.
"""

from __future__ import annotations

import numpy as np


def geometric_adstock(spend: np.ndarray, decay: float = 0.5) -> np.ndarray:
    """
    Apply geometric (exponential) adstock decay.

    The simplest and most common adstock model. Each period's effect decays
    geometrically: effect_t = spend_t + decay * effect_{t-1}

    Args:
        spend: Array of spend values over time.
        decay: Decay rate between 0 and 1. Higher = longer memory.
            - 0.0: No carryover (each period is independent)
            - 0.3: Short memory (~3 periods of meaningful effect)
            - 0.7: Long memory (~10 periods)
            - 0.9: Very long memory (~30 periods, typical for TV)

    Returns:
        Adstocked spend array of same length.

    Example:
        >>> spend = np.array([100, 0, 0, 0, 0])
        >>> geometric_adstock(spend, decay=0.5)
        array([100.  ,  50.  ,  25.  ,  12.5 ,   6.25])
    """
    if not 0 <= decay <= 1:
        raise ValueError(f"Decay rate must be between 0 and 1, got {decay}")

    result = np.zeros_like(spend, dtype=float)
    result[0] = spend[0]
    for t in range(1, len(spend)):
        result[t] = spend[t] + decay * result[t - 1]
    return result


def weibull_adstock(
    spend: np.ndarray, shape: float = 1.0, scale: float = 1.0, max_lag: int = 13
) -> np.ndarray:
    """
    Apply Weibull PDF adstock (flexible decay shape).

    More flexible than geometric — can model delayed peak effects (e.g., TV
    campaigns that peak in effectiveness a few days after airing).

    Args:
        spend: Array of spend values.
        shape: Weibull shape parameter.
            - shape < 1: Fast initial decay (similar to geometric)
            - shape = 1: Exponential decay
            - shape > 1: Delayed peak then decay (bell-shaped effect)
        scale: Weibull scale parameter (controls spread).
        max_lag: Maximum number of lag periods to consider.

    Returns:
        Adstocked spend array.
    """
    # Generate Weibull weights.  Evaluate the PDF at lags 1..max_lag (not 0)
    # to avoid the degenerate x=0 point where the PDF is 0 (shape>1) or
    # infinite (shape<1).  This ensures every lag has a positive weight.
    lags = np.arange(1, max_lag + 1, dtype=float)
    weights = (shape / scale) * (lags / scale) ** (shape - 1) * np.exp(-((lags / scale) ** shape))

    # Normalize
    total = weights.sum()
    if total > 0:
        weights = weights / total

    # Apply convolution
    result = np.convolve(spend, weights, mode="full")[: len(spend)]
    return result


def delayed_adstock(spend: np.ndarray, decay: float = 0.5, delay: int = 1) -> np.ndarray:
    """
    Geometric adstock with a fixed delay before effect begins.

    Useful for channels with known lag between spend and impact
    (e.g., direct mail, OOH installations).

    Args:
        spend: Array of spend values.
        decay: Geometric decay rate (0-1).
        delay: Number of periods before the effect starts.

    Returns:
        Adstocked spend array with delayed effect.
    """
    if delay < 0:
        raise ValueError(f"Delay must be non-negative, got {delay}")

    # Shift spend by delay periods
    delayed = np.zeros_like(spend, dtype=float)
    if delay < len(spend):
        delayed[delay:] = spend[: len(spend) - delay]

    return geometric_adstock(delayed, decay)
