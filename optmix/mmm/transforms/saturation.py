"""
Saturation (diminishing returns) transformations for MMM.

Saturation curves model the fact that each additional dollar spent on a channel
produces less incremental effect than the previous dollar. These transforms
convert raw (adstocked) spend into expected response.
"""

from __future__ import annotations

import numpy as np


def hill_saturation(spend: np.ndarray, half_sat: float, slope: float = 1.0) -> np.ndarray:
    """
    Hill function (sigmoidal) saturation curve.

    The most widely used saturation function in MMM. Models the relationship:
    response = spend^slope / (spend^slope + half_sat^slope)

    Args:
        spend: Array of (adstocked) spend values.
        half_sat: The spend level at which 50% of maximum response is achieved.
            This is the "inflection point" — spending below this has high marginal
            returns; above it, returns diminish rapidly.
        slope: Controls the steepness of the curve.
            - slope = 1: Gentle S-curve (Michaelis-Menten)
            - slope = 2: Steeper transition
            - slope > 3: Very sharp threshold effect

    Returns:
        Saturated response values between 0 and 1.

    Example:
        >>> spend = np.array([0, 25000, 50000, 100000, 200000])
        >>> hill_saturation(spend, half_sat=50000, slope=2.0)
        array([0.   , 0.2  , 0.5  , 0.8  , 0.941])
    """
    if half_sat <= 0:
        raise ValueError(f"half_sat must be positive, got {half_sat}")

    spend_safe = np.maximum(spend, 0)
    numerator = spend_safe**slope
    denominator = spend_safe**slope + half_sat**slope
    return np.where(denominator > 0, numerator / denominator, 0.0)


def logistic_saturation(spend: np.ndarray, midpoint: float, steepness: float = 1.0) -> np.ndarray:
    """
    Logistic (sigmoid) saturation curve.

    Classic S-curve: response = 1 / (1 + exp(-steepness * (spend - midpoint)))
    Shifted to start at 0 when spend is 0.

    Args:
        spend: Array of spend values.
        midpoint: Spend level at the inflection point.
        steepness: Controls how quickly saturation occurs.

    Returns:
        Saturated response values between 0 and 1.
    """
    raw = 1.0 / (1.0 + np.exp(-steepness * (spend - midpoint)))
    baseline = 1.0 / (1.0 + np.exp(-steepness * (0 - midpoint)))
    ceiling = 1.0 / (1.0 + np.exp(-steepness * (spend.max() * 2 - midpoint)))

    # Normalize to 0-1 range
    normalized = (raw - baseline) / (ceiling - baseline) if ceiling > baseline else raw
    return np.clip(normalized, 0, 1)


def michaelis_menten(spend: np.ndarray, vmax: float, km: float) -> np.ndarray:
    """
    Michaelis-Menten saturation (from enzyme kinetics, used in pharma MMM).

    response = vmax * spend / (km + spend)

    Args:
        spend: Array of spend values.
        vmax: Maximum achievable response (asymptote).
        km: Spend level at which response = vmax/2.

    Returns:
        Response values (not normalized — scale depends on vmax).
    """
    if km <= 0:
        raise ValueError(f"km must be positive, got {km}")

    spend_safe = np.maximum(spend, 0)
    return vmax * spend_safe / (km + spend_safe)


def power_saturation(spend: np.ndarray, exponent: float = 0.5) -> np.ndarray:
    """
    Power function (concave) saturation.

    The simplest diminishing returns model: response = spend^exponent

    Args:
        spend: Array of spend values.
        exponent: Power exponent, typically 0 < exponent < 1.
            - 0.5: Square root (moderate diminishing returns)
            - 0.3: Strong diminishing returns
            - 0.8: Mild diminishing returns

    Returns:
        Transformed spend values.
    """
    if exponent <= 0:
        raise ValueError(f"Exponent must be positive, got {exponent}")

    return np.power(np.maximum(spend, 0), exponent)
