#!/usr/bin/env python3
"""
OptMix Quickstart — See results in 2 minutes.

No API key needed! This example uses the built-in MMM engine directly
to analyze sample marketing data and optimize budget allocation.

Usage:
    python examples/quickstart.py
"""

from optmix.data.samples import load_sample
from optmix.mmm.models.ridge_mmm import RidgeMMM
from optmix.mmm.optimizer.budget_optimizer import BudgetOptimizer

# ── 1. Load sample data ────────────────────────────────────────────
print("=" * 60)
print("  OptMix Quickstart - Marketing Mix Modeling in 2 minutes")
print("=" * 60)

df = load_sample("ecommerce")
print(f"\nLoaded ecommerce dataset: {len(df)} weeks, {len(df.columns)} columns")
print(f"Columns: {', '.join(df.columns.tolist())}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# ── 2. Fit a Ridge MMM ─────────────────────────────────────────────
print("\n--- Fitting Marketing Mix Model ---")

channel_cols = [
    "google_search", "google_shopping", "meta_ads", "tiktok_ads",
    "youtube", "email", "affiliate", "display",
]
control_cols = ["avg_price", "promo"]
print(f"Channels: {', '.join(channel_cols)}")
print(f"Controls: {', '.join(control_cols)}")

model = RidgeMMM()
result = model.fit(
    data=df,
    target="revenue",
    date_col="date",
    channels=channel_cols,
    controls=control_cols,
)

print(f"Model R-squared: {result.r_squared:.3f}")

# ── 3. Channel ROAS ────────────────────────────────────────────────
print("\n--- Channel Performance (ROAS) ---")
print(f"{'Channel':<25} {'ROAS':>8} {'Share':>10}")
print("-" * 46)

for channel in sorted(result.channel_roas, key=result.channel_roas.get, reverse=True):
    roas = result.channel_roas[channel]
    share = result.channel_share.get(channel, 0) * 100
    print(f"{channel:<25} {roas:>7.2f}x {share:>8.1f}%")

# ── 4. Budget Optimization ─────────────────────────────────────────
print("\n--- Budget Optimization ---")

total_spend = {ch: float(df[ch].sum()) for ch in channel_cols}
current_weekly = {ch: total_spend[ch] / len(df) for ch in channel_cols}
weekly_budget = sum(current_weekly.values())

optimizer = BudgetOptimizer(model=model)
optimized = optimizer.optimize(
    total_budget=weekly_budget,
    current_allocation=current_weekly,
)

print(f"Total weekly budget: ${weekly_budget:,.0f}")
print(f"\n{'Channel':<25} {'Current':>10} {'Optimized':>10} {'Change':>10}")
print("-" * 58)

for channel in channel_cols:
    current = current_weekly[channel]
    optimal = optimized.allocation.get(channel, 0)
    change = optimal - current
    sign = "+" if change >= 0 else ""
    print(f"{channel:<25} ${current:>8,.0f} ${optimal:>8,.0f} {sign}${change:>7,.0f}")

if optimized.expected_lift_pct is not None:
    print(f"\nExpected revenue lift: {optimized.expected_lift_pct:+.1f}%")

# ── 5. Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Done! Next steps:")
print("  - Run `optmix setup` to configure an LLM provider")
print("  - Run `optmix chat` for AI-powered analysis")
print("  - Try your own data: team.load_data(path='your_data.csv')")
print("=" * 60)
