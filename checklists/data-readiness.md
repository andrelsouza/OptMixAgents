# Data Readiness Checklist

Quality gate to verify data supports the measurement objectives before modeling.

## Completeness

- [ ] Target variable (revenue/conversions) has no missing periods
- [ ] All channel spend columns have complete time series
- [ ] Date range covers at least 52 weeks (1 full seasonal cycle)
- [ ] Ideally 104+ weeks for robust seasonality estimation
- [ ] No unexplained gaps > 2 consecutive periods

## Granularity

- [ ] Time granularity is consistent (weekly recommended for MMM)
- [ ] Channel definitions are mutually exclusive (no double-counting)
- [ ] Geographic aggregation level is appropriate for the business question
- [ ] Spend data represents actual spend, not budgeted/planned

## Quality

- [ ] Spend values are non-negative
- [ ] No obvious data entry errors (e.g., spend = $999,999,999)
- [ ] Currency is consistent across all channels
- [ ] Inflation/deflation adjustments applied if time range > 2 years
- [ ] Zero-spend periods are true zeros, not missing data

## Variance

- [ ] Each channel has meaningful spend variation over time
- [ ] No channel is 100% correlated with another (collinearity check)
- [ ] Target variable shows sufficient variation to model
- [ ] At least some channels were paused/reduced during the period

## Controls

- [ ] Seasonality patterns are identifiable in the data
- [ ] Major external events are documented (COVID, competitor launches)
- [ ] Pricing data included if price varied during the period
- [ ] Promotion/sale flags included if applicable
- [ ] Macroeconomic controls available if needed (CPI, unemployment)

## Business Context

- [ ] Channel taxonomy matches how the business makes decisions
- [ ] Attribution windows align with typical conversion paths
- [ ] Business objective is clearly defined and measurable
- [ ] Stakeholders agree on the target metric
