# Good First Issues

Curated list of beginner-friendly issues for new contributors. Each issue is self-contained with clear scope, affected files, and acceptance criteria.

Pick one, open an issue on GitHub referencing this list, and submit a PR!

---

## 1. Add public re-exports to empty `__init__.py` files

**Labels:** `good first issue`, `refactor`

Six `__init__.py` files are empty, forcing users to know internal module paths. Add re-exports so `from optmix.mmm import RidgeMMM` works.

**Files to change:**
- `optmix/mmm/__init__.py` — export `RidgeMMM`, `BayesianMMM`, `BaseMMM`, `ModelResult`, `OptimizationResult`
- `optmix/mmm/optimizer/__init__.py` — export `BudgetOptimizer`, `ChannelConstraint`
- `optmix/mmm/transforms/__init__.py` — export all adstock and saturation functions
- `optmix/data/__init__.py` — export `load_sample`

**Acceptance criteria:**
- `from optmix.mmm import RidgeMMM` works
- `from optmix.data import load_sample` works
- Each `__init__.py` has an `__all__` list
- Existing tests still pass

---

## 2. Fix `UnboundLocalError` in `assess_data_readiness`

**Labels:** `good first issue`, `bug`

If `pd.to_datetime()` fails in `strategy_tools.py:175-187`, the variable `date_span` is never assigned but referenced in the f-string on line 187, causing `UnboundLocalError`.

**File:** `optmix/tools/strategy_tools.py` (lines 175-187)

**Fix:** Initialize `date_span = 0` before the `try` block, or restructure the conditional to not reference `date_span` when it might not exist.

**Acceptance criteria:**
- No `UnboundLocalError` when date parsing fails
- Add a test that triggers the failure path
- Existing tests still pass

---

## 3. Add input validation to `RidgeMMM.fit()` and `BayesianMMM.fit()`

**Labels:** `good first issue`, `enhancement`

Both models accept empty DataFrames, missing columns, and invalid inputs without clear error messages. The errors surface deep in numpy/scikit-learn instead.

**Files:** `optmix/mmm/models/ridge_mmm.py`, `optmix/mmm/models/bayesian_mmm.py`

**Validations to add at the top of `fit()`:**
- `data` is not empty → `ValueError("data must not be empty")`
- `target` column exists in `data` → `ValueError(f"target column '{target}' not found in data")`
- `date_col` column exists in `data` → `ValueError(f"date column '{date_col}' not found in data")`
- All `channels` columns exist in `data` (if provided)

**Acceptance criteria:**
- Clear `ValueError` with helpful message for each invalid input
- Add tests in a new `tests/test_ridge_mmm.py` covering each validation
- Existing tests still pass

---

## 4. Add test coverage for `logistic_saturation` and `michaelis_menten`

**Labels:** `good first issue`, `tests`

These two saturation functions in `optmix/mmm/transforms/saturation.py` have zero test coverage.

**File to create:** Add tests to `tests/test_core.py` in `TestSaturationTransforms`

**Tests to write:**
- `logistic_saturation(0, ...)` returns ~0
- `logistic_saturation` at very large spend returns ~1
- `logistic_saturation` is monotonically increasing
- `michaelis_menten(spend=km, km, vmax)` returns `vmax / 2`
- `michaelis_menten` with `km <= 0` raises `ValueError`
- Both functions handle numpy arrays correctly

**Acceptance criteria:**
- At least 6 new test cases
- All tests pass with `python -m pytest tests/test_core.py -v`

---

## 5. Create `tests/test_ridge_mmm.py` — direct model tests

**Labels:** `good first issue`, `tests`

`RidgeMMM` is only tested indirectly through the tool layer. Add direct unit tests.

**File to create:** `tests/test_ridge_mmm.py`

**Tests to write:**
- `fit()` returns a `ModelResult` with expected fields populated
- `r_squared` is between 0 and 1 on sample data
- `channel_roas` keys match input channels
- `predict()` returns array of correct length
- `get_saturation_curves()` returns dict of DataFrames with `spend` and `response` columns
- `get_channel_contributions()` returns DataFrame with date + channel columns
- `get_marginal_roas()` returns a positive float for valid channels

**Acceptance criteria:**
- At least 7 test cases
- Use `load_sample("ecommerce")` for test data
- All tests pass

---

## 6. Add `total_budget` and `objective` validation to `BudgetOptimizer`

**Labels:** `good first issue`, `enhancement`

`BudgetOptimizer.optimize()` silently accepts `total_budget <= 0` and unknown `objective` strings.

**File:** `optmix/mmm/optimizer/budget_optimizer.py` (lines 41-62)

**Fix:**
- `if total_budget <= 0: raise ValueError(f"total_budget must be positive, got {total_budget}")`
- `if objective != "maximize_revenue": raise ValueError(f"Unknown objective '{objective}'. Supported: 'maximize_revenue'")`

**Acceptance criteria:**
- `ValueError` raised for invalid inputs with clear message
- Add tests for both validation cases
- Existing tests still pass

---

## 7. Extract duplicated `_get_state`/`_set_state` helpers to shared module

**Labels:** `good first issue`, `refactor`

The same `_get_state` and `_set_state` helper functions are copy-pasted in three files:
- `optmix/tools/data_tools.py` (lines 283-298)
- `optmix/tools/mmm_tools.py` (lines 267-282)
- `optmix/tools/optimization_tools.py` (lines 186-200)

**Fix:**
1. Create `optmix/tools/_helpers.py` with the shared functions
2. Import from the shared module in all three files
3. Add proper type hints (`state: SharedState | dict`)

**Acceptance criteria:**
- Single source of truth for state helpers
- All three tool modules import from `_helpers.py`
- Existing tests still pass

---

## 8. Add Weibull adstock edge case tests

**Labels:** `good first issue`, `tests`

`weibull_adstock` in `tests/test_core.py` has only one test (output shape). Missing edge cases.

**File:** `tests/test_core.py`, class `TestAdstockTransforms`

**Tests to write:**
- `shape < 1`: fast decay (peak at first lag)
- `shape > 1`: delayed peak (the main use case for Weibull)
- `max_lag=1`: minimal lag
- Weights are normalized (sum to ~1)
- Output has same length as input

**Acceptance criteria:**
- At least 4 new test cases
- All tests pass

---

## 9. Add ground-truth parameters to `retail_chain` and `saas_b2b` samples

**Labels:** `good first issue`, `enhancement`

The `ecommerce` sample uses proper adstock + saturation transforms with known parameters, making it useful for validating model recovery. The `retail_chain` and `saas_b2b` generators use simplified random multipliers instead.

**File:** `optmix/data/samples.py` (functions `_generate_retail`, `_generate_saas`)

**Fix:**
- Apply `geometric_adstock` and `hill_saturation` (or `weibull_adstock` for retail) to channel spend
- Store known true parameters that can be compared against fitted results
- Maintain the same output columns and shape

**Acceptance criteria:**
- Both generators use proper MMM transforms
- Model fitted on the generated data recovers approximately correct ROAS ordering
- Existing tests still pass

---

## 10. Add a `cpg` (consumer packaged goods) sample dataset

**Labels:** `good first issue`, `enhancement`

The project has 3 sample datasets but none exercise the Weibull adstock transform (delayed peak effects common in TV/print advertising).

**File:** `optmix/data/samples.py`

**What to add:**
- New `_generate_cpg()` function with channels: `tv`, `print`, `radio`, `digital`, `in_store_promo`
- Use `weibull_adstock` for TV and print (delayed peak effects)
- Use `geometric_adstock` for digital channels
- ~156 weeks of data
- Register in the `generators` dict in `load_sample()`

**Acceptance criteria:**
- `load_sample("cpg")` returns a valid DataFrame
- At least 5 channels with realistic spend ranges
- RidgeMMM can fit the data without errors
- Add to the docstring in `load_sample()`

---

## 11. Validate `provider` in `resolve_config`

**Labels:** `good first issue`, `enhancement`

`resolve_config(provider="typo")` silently creates an invalid config. The error only surfaces later when creating the LLM client.

**File:** `optmix/core/config.py` (function `resolve_config`)

**Fix:** Add validation against `SUPPORTED_PROVIDERS` list at the end of `resolve_config`, before returning.

**Acceptance criteria:**
- `ValueError` raised for unknown providers
- Error message lists supported providers
- Add test case to `tests/test_config.py`

---

## 12. Improve error logging in `load_config`

**Labels:** `good first issue`, `enhancement`

`load_config` uses a bare `except Exception` that swallows all errors silently, including permission errors and YAML syntax errors.

**File:** `optmix/core/config.py` (lines 67-71)

**Fix:**
- Use `except FileNotFoundError` for the "file doesn't exist" case (return defaults)
- Use `except yaml.YAMLError` for malformed YAML (log warning, return defaults)
- Let `PermissionError` and other unexpected errors propagate

**Acceptance criteria:**
- `FileNotFoundError` returns defaults silently
- `yaml.YAMLError` logs a warning and returns defaults
- `PermissionError` propagates to the caller
- Add tests for each error path

---

## 13. Add `BudgetOptimizer` constraint tests

**Labels:** `good first issue`, `tests`

The optimizer is only tested in the unconstrained case. No tests verify that channel min/max constraints work.

**File to create or extend:** `tests/test_tools.py` or new `tests/test_budget_optimizer.py`

**Tests to write:**
- Optimize with `min_spend` constraint → allocation respects minimum
- Optimize with `max_spend` constraint → allocation respects maximum
- `binding_constraints` list is populated correctly
- `run_scenario` with percentage changes produces correct allocation
- Optimizer with unfitted model raises `RuntimeError`

**Acceptance criteria:**
- At least 5 new test cases
- All tests pass

---

## How to Contribute

1. Pick an issue from this list
2. Open a GitHub issue referencing the number (e.g., "Good First Issue #3: Add input validation")
3. Fork the repo and create a branch: `git checkout -b fix/issue-description`
4. Make your changes
5. Run tests: `python -m pytest tests/ -v -m "not slow"`
6. Run linting: `ruff check optmix/ tests/ && ruff format --check optmix/ tests/`
7. Submit a PR using the PR template
