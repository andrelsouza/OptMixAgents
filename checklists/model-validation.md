# Model Validation Checklist

Quality gate to verify the MMM is trustworthy before using it for optimization.

## Statistical Fit

- [ ] R² > 0.70 (model explains majority of target variance)
- [ ] MAPE < 15% (predictions are within 15% of actuals on average)
- [ ] Residuals show no systematic patterns (visual inspection)
- [ ] No significant autocorrelation in residuals (Durbin-Watson test)

## Bayesian Diagnostics (if applicable)

- [ ] All chains converged (R-hat < 1.05 for all parameters)
- [ ] Effective sample size > 400 for all parameters
- [ ] No divergent transitions (or < 1% of total samples)
- [ ] Posterior predictive check passes (data falls within predicted intervals)
- [ ] Priors are not dominating the posterior (prior sensitivity check)

## Business Sense

- [ ] All channel coefficients have expected sign (positive for spend → revenue)
- [ ] Channel ROAS values are within plausible industry ranges
- [ ] Largest channel contributors align with business intuition
- [ ] Base revenue (intercept) is reasonable for the business
- [ ] Seasonality pattern matches known business cycles

## Adstock & Saturation

- [ ] Adstock decay rates are plausible for each channel type
  - Digital: 0.1-0.4 (short memory)
  - TV: 0.5-0.9 (long memory)
  - OOH: 0.3-0.7 (medium memory)
- [ ] Saturation curves show diminishing returns (not linear)
- [ ] Half-saturation points are within observed spend ranges
- [ ] No channel shows increasing returns at high spend (likely misspecification)

## Robustness

- [ ] Model is stable across different time windows (rolling validation)
- [ ] Removing any single channel doesn't dramatically change other estimates
- [ ] Results are consistent across model specifications (Ridge vs Bayesian)
- [ ] Out-of-sample predictions are reasonable (holdout validation)

## Documentation

- [ ] Model card generated with all specifications and assumptions
- [ ] Known limitations are documented
- [ ] Data transformations are logged and reproducible
- [ ] Version and timestamp recorded
