# Changelog

All notable changes to OptMix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Interactive setup wizard** (`optmix setup`) — choose provider, enter API key, select model, test connection
- **Multi-provider LLM support** — Anthropic and OpenAI, with automatic tool format translation
- **Persistent config** at `~/.optmix/config.yaml` with precedence: CLI flags > env vars > config file > defaults
- **Inter-agent delegation** — agents can delegate tasks to each other via `delegate_to_agent` tool (max depth 3)
- **BayesianMMM** — full PyMC-Marketing integration with posterior sampling and credible intervals
- **Smart CLI data loading** — `load ecommerce` works alongside full file paths
- **Working quickstart example** (`examples/quickstart.py`) — runs in 2 minutes, no API key needed
- **CI/CD pipeline** (`.github/workflows/ci.yml`) — matrix testing on Python 3.10/3.11/3.12 + ruff linting
- **GitHub templates** — bug report, feature request, PR template
- **CODE_OF_CONDUCT.md** — Contributor Covenant v2.1
- **Good First Issues list** — 13 curated issues for new contributors
- Tests for config, OpenAI client, CLI setup, delegation, BayesianMMM (188 total tests)

### Changed
- CLI commands (`chat`, `agent`, `run`) now use config resolution — no more mandatory `--llm` flag
- `--llm` flag default changed from hardcoded model to `None` (resolved from config)

## [0.1.0] — 2026-02-07

### Added
- Initial project structure and architecture
- 6 agent definitions (Strategist, Analyst, Modeler, Optimizer, Reporter, Orchestrator)
- Agent-as-Code schema with YAML loader and Markdown compiler
- Core MMM engine with base model interface
- RidgeMMM (fast baseline model with scikit-learn)
- Adstock transforms: geometric, Weibull, delayed
- Saturation transforms: Hill, logistic, Michaelis-Menten, power
- Budget optimizer with constrained optimization (scipy)
- Scenario simulator for what-if analysis
- CLI with `chat`, `agent`, `compile`, `team` commands
- 3 sample datasets: ecommerce, retail_chain, saas_b2b
- 2 workflow definitions: full_measurement_cycle, quick_optimization
- Quality gate checklists: data readiness, model validation
- Measurement brief template
- Industry benchmarks knowledge base
- Channel taxonomy knowledge base
- Unit tests for transforms, schema, and data generation
- MIT License, CONTRIBUTING.md, README.md
