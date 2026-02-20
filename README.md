<p align="center">
  <strong>OptMix</strong>
</p>

<h3 align="center">AI Marketing Agents powered by Marketing Mix Modeling</h3>

<p align="center">
  <em>Agent-as-Code for Marketing Measurement & Strategy</em>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#the-agents">The Agents</a> •
  <a href="#mmm-engine">MMM Engine</a> •
  <a href="#workflows">Workflows</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License"/>
  <img src="https://img.shields.io/badge/Agent--as--Code-orange.svg" alt="Agent-as-Code"/>
  <img src="https://img.shields.io/badge/MMM-PyMC%20Marketing-red.svg" alt="PyMC Marketing"/>
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"/>
</p>

---

## Why OptMix?

Marketing teams spend millions across channels but can't answer the basic question: **"What's actually working?"**

Traditional MMM tools are expensive consulting engagements. Generic AI chatbots don't understand adstock, saturation curves, or incrementality. And no one has combined structured, persona-driven AI agents with the statistical rigor of Marketing Mix Modeling.

**OptMix fixes that.**

Using an **Agent-as-Code** architecture — where specialized AI personas are defined as self-contained configuration files with personas, capabilities, workflows, and dependencies — OptMix creates a **virtual marketing measurement team** where each agent is a domain expert with MMM tools at their fingertips.

```
You: "Our CAC increased 23% last quarter. What's driving it and how do we fix it?"

Strategist Maya: "Let me pull the context. Loading your channel data..."
Analyst Kai:     "Running decomposition. TV adstock is carrying over into wasted frequency..."
Modeler Priya:   "Bayesian MMM fitted. Saturation curves show Meta is past inflection point..."
Optimizer Ravi:  "Reallocation ready. Shifting 18% from Meta to TikTok yields +12% ROAS..."
Reporter Nora:   "Executive summary generated with confidence intervals and scenario comparison."
```

---

## Quickstart

### 2-Minute Demo (No API Key Needed)

```bash
git clone https://github.com/optmix-ai/optmix.git
cd optmix
pip install -e "."
python examples/quickstart.py
```

This loads sample data, fits a Ridge MMM, shows channel ROAS, and optimizes budget allocation — all locally.

### Full Setup (with AI Agents)

```bash
# Install with LLM support
pip install -e ".[llm]"

# Interactive setup wizard — choose provider, enter API key, test connection
optmix setup

# Start chatting with your marketing team
optmix chat
```

The setup wizard saves your config to `~/.optmix/config.yaml`. Supports **Anthropic** and **OpenAI** providers.

You can always override with CLI flags:

```bash
optmix chat --llm "openai/gpt-4o" --api-key "sk-..."
```

### Load Your Data

Inside `optmix chat`:

```
> load ecommerce              # built-in sample dataset
> load /path/to/your/data.csv # your own CSV
```

---

## The Agents

Each agent is a fully defined persona with specialized tools, defined as YAML configuration files. Agents can **delegate tasks to each other** automatically — ask Maya a modeling question and she'll consult Priya.

| Agent | Persona | Role | Core Tools |
|-------|---------|------|------------|
| **Strategist** | Maya Chen | Sets business context, defines KPIs, creates measurement briefs | `load_industry_benchmarks`, `load_channel_taxonomy`, `assess_data_readiness` |
| **Analyst** | Kai Nakamura | Data validation, EDA, feature engineering, anomaly detection | `validate_data`, `run_eda`, `describe_channels` |
| **Modeler** | Priya Sharma | Builds & fits MMM models, runs diagnostics, extracts contributions | `fit_mmm_model`, `get_model_diagnostics`, `get_channel_contributions` |
| **Optimizer** | Ravi Santos | Budget allocation, scenario simulation, constrained optimization | `optimize_budget`, `run_scenario`, `get_marginal_roas` |
| **Reporter** | Nora Lindqvist | Generates executive reports, visualizations, and action plans | `generate_markdown_report`, `generate_chart`, `create_action_plan` |
| **Orchestrator** | — | Routes tasks, manages shared state, ensures context handoffs | Automatic routing based on query content |

### Agent Interaction Example

```python
from optmix.core.team import OptMixTeam

team = OptMixTeam(llm="anthropic/claude-sonnet-4-20250514", api_key="sk-...")

# Natural conversation — the Orchestrator routes to the right agent
team.chat("We spent $2M last quarter across Google, Meta, TV, and OOH. Revenue was $8M but declining.")

# Load data and let agents analyze it
team.load_data(sample="ecommerce")
team.chat("Fit a Ridge MMM and show me the ROAS by channel")
team.chat("Now optimize the budget for $500K monthly")
```

---

## MMM Engine

The core differentiator. Every agent has access to a shared **MMM Engine** — a Python toolkit for marketing mix modeling.

### Supported Models

| Model | Backend | Use Case | Speed |
|-------|---------|----------|-------|
| **RidgeMMM** | scikit-learn | Quick baseline, deterministic, great for rapid iteration | Instant |
| **BayesianMMM** | [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) | Full Bayesian inference, posterior distributions, credible intervals | Thorough |

### Usage

```python
from optmix.data.samples import load_sample
from optmix.mmm.models.ridge_mmm import RidgeMMM
from optmix.mmm.optimizer.budget_optimizer import BudgetOptimizer

# Load data
df = load_sample("ecommerce")

# Fit model
model = RidgeMMM()
result = model.fit(
    data=df,
    target="revenue",
    date_col="date",
    channels=["google_search", "meta_ads", "tiktok_ads", "youtube"],
)

# Inspect results
print(result.r_squared)          # Model fit quality
print(result.channel_roas)       # ROAS per channel
contributions = result.channel_contributions  # DataFrame over time

# Optimize budget
optimizer = BudgetOptimizer(model=model)
optimized = optimizer.optimize(
    total_budget=500_000,
    constraints={
        "google_search": {"min": 50_000, "max": 200_000},
        "meta_ads": {"min": 30_000},
    },
)

print(optimized.allocation)          # Optimal spend per channel
print(optimized.expected_lift_pct)   # Expected lift vs current
```

### Transforms

```python
from optmix.mmm.transforms.adstock import geometric_adstock, weibull_adstock, delayed_adstock
from optmix.mmm.transforms.saturation import hill_saturation, logistic_saturation, michaelis_menten
```

| Transform | Type | Use Case |
|-----------|------|----------|
| `geometric_adstock` | Adstock | Standard exponential decay (TV, radio) |
| `weibull_adstock` | Adstock | Delayed peak effects (print, brand campaigns) |
| `delayed_adstock` | Adstock | Fixed delay before decay starts |
| `hill_saturation` | Saturation | Classic diminishing returns curve |
| `logistic_saturation` | Saturation | S-curve with inflection point |
| `michaelis_menten` | Saturation | Enzyme kinetics model adapted for marketing |

### Scenario Simulator

```python
optimizer = BudgetOptimizer(model=model)

# "What if we cut TV by 30% and move it to digital?"
scenario = optimizer.run_scenario(
    base_allocation=current_spend,
    changes={"tv": -0.30, "meta_ads": +0.15, "tiktok_ads": +0.15},
)

print(scenario.expected_lift_pct)  # Revenue change %
```

---

## Workflows

Workflows are YAML files that define structured, multi-step processes with quality gates between phases.

```yaml
# workflows/full_measurement_cycle.yaml (simplified)
workflow:
  name: full-measurement-cycle
  title: "End-to-End Marketing Measurement"
  phases:
    - name: strategize
      steps:
        - id: market_context
          agent: strategist
          action: market-context
        - id: data_readiness
          agent: strategist
          action: data-readiness
          gate: data-readiness-checklist

    - name: model
      steps:
        - id: validate_data
          agent: analyst
          action: validate
        - id: run_eda
          agent: analyst
          action: eda
        - id: fit_model
          agent: modeler
          action: fit
        - id: run_diagnostics
          agent: modeler
          action: diagnostics
          gate: model-validation-checklist

    - name: optimize
      steps:
        - id: optimize_budget
          agent: optimizer
          action: optimize
        - id: calculate_mroas
          agent: optimizer
          action: marginal-roas

    - name: activate
      steps:
        - id: executive_report
          agent: reporter
          action: report
        - id: action_plan
          agent: reporter
          action: action-plan
```

Run with:

```bash
optmix run --data marketing_spend.csv --target revenue --budget 500000
```

---

## Architecture

```
optmix/
├── agents/                     # Agent-as-Code definitions (YAML)
│   ├── strategist.agent.yaml
│   ├── analyst.agent.yaml
│   ├── modeler.agent.yaml
│   ├── optimizer.agent.yaml
│   ├── reporter.agent.yaml
│   └── orchestrator.agent.yaml
│
├── core/                       # Framework core
│   ├── team.py                 # OptMixTeam — main entry point
│   ├── executor.py             # Agent executor (LLM tool-calling loop)
│   ├── llm.py                  # LLM clients (Anthropic, OpenAI)
│   ├── config.py               # Config persistence (~/.optmix/config.yaml)
│   ├── state.py                # Shared state between agents
│   ├── schema.py               # Agent YAML schema loader
│   └── workflow_engine.py      # Workflow execution engine
│
├── mmm/                        # Marketing Mix Modeling engine
│   ├── models/
│   │   ├── base.py             # BaseMMM interface + ModelResult
│   │   ├── ridge_mmm.py        # Fast scikit-learn baseline
│   │   └── bayesian_mmm.py     # PyMC-Marketing Bayesian MMM
│   ├── transforms/
│   │   ├── adstock.py          # Geometric, Weibull, delayed
│   │   └── saturation.py       # Hill, logistic, Michaelis-Menten
│   └── optimizer/
│       └── budget_optimizer.py # Constrained budget optimization
│
├── tools/                      # Agent tools (callable functions)
│   ├── data_tools.py           # Data loading, validation, EDA
│   ├── mmm_tools.py            # Model fitting and diagnostics
│   ├── optimization_tools.py   # Budget and scenario tools
│   ├── strategy_tools.py       # Market context and KPIs
│   └── report_tools.py         # Report generation
│
├── data/
│   └── samples.py              # Synthetic sample datasets
│
├── cli/
│   └── main.py                 # CLI: setup, chat, agent, run
│
├── workflows/                  # YAML workflow definitions
├── templates/                  # Report templates
├── checklists/                 # Quality gate checklists
├── knowledge/                  # Domain knowledge for agents
│
├── examples/
│   └── quickstart.py           # 2-minute working demo
├── tests/                      # 188 tests (170 fast + 18 slow)
├── pyproject.toml
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── CHANGELOG.md
```

---

## Configuration

OptMix resolves configuration in this order:

1. **CLI flags**: `--llm "openai/gpt-4o" --api-key "sk-..."`
2. **Environment variables**: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `OPTMIX_API_KEY`
3. **Config file**: `~/.optmix/config.yaml` (created by `optmix setup`)
4. **Defaults**: Anthropic with `claude-sonnet-4-5-20250929`

---

## Sample Datasets

```python
from optmix.data.samples import load_sample

df = load_sample("ecommerce")     # DTC with 8 channels, 104 weeks
df = load_sample("retail_chain")  # Brick-and-mortar with TV/Radio/OOH, 156 weeks
df = load_sample("saas_b2b")      # B2B SaaS with long sales cycles, 104 weeks
```

---

## Roadmap

### v0.1 — Foundation (Done)
- [x] Project structure and Agent-as-Code schema
- [x] 6 agent definitions with personas and tools
- [x] RidgeMMM (fast baseline model)
- [x] Adstock and saturation transforms
- [x] Budget optimizer with constraints
- [x] CLI with chat, agent, run commands
- [x] 3 sample datasets
- [x] Workflow engine with quality gates

### v0.2 — Multi-Provider & Collaboration (Current)
- [x] Interactive setup wizard (`optmix setup`)
- [x] Anthropic + OpenAI LLM support
- [x] Persistent config at `~/.optmix/config.yaml`
- [x] BayesianMMM (PyMC-Marketing integration)
- [x] Inter-agent delegation
- [x] CI/CD pipeline
- [ ] Public re-exports in `__init__.py` files
- [ ] Input validation on model boundaries

### v0.3 — Connectors & Polish
- [ ] Google Ads, Meta Ads, GA4 connectors
- [ ] BigQuery/Snowflake data sources
- [ ] HTML/PDF report generation
- [ ] Model persistence and versioning
- [ ] Additional sample datasets (CPG, pharma)

### v0.4 — Advanced
- [ ] Incrementality testing integration
- [ ] Geo-lift experiment design
- [ ] Real-time dashboard (Streamlit)
- [ ] Multi-touch attribution comparison

### v1.0 — Production Ready
- [ ] API server mode
- [ ] Custom agent creation wizard
- [ ] Model card generation
- [ ] Automated model refresh pipelines

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

New here? Check out [GOOD_FIRST_ISSUES.md](GOOD_FIRST_ISSUES.md) for 13 curated issues.

**Key areas we need help:**
- **MMM Models** — Additional model backends, custom transformations
- **Agent Personas** — Refine agent behaviors and domain knowledge
- **Connectors** — New data source integrations (Google Ads, Meta, BigQuery)
- **Tests** — Coverage for transforms, models, and optimizer edge cases
- **Documentation** — Tutorials, cookbooks, case studies

---

## Philosophy

We believe that **measurement should be accessible**, not locked behind $200K consulting engagements. And that AI agents need **structure and personas** to deliver reliable, trustworthy results — not just raw LLM output.

### Inspired by

- The **Agent-as-Code** movement — the idea that AI agents should be defined as declarative configuration files with personas, tools, and workflows, rather than imperative code.
- **[PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing)** for Bayesian Marketing Mix Modeling.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>OptMix</strong> — Because great marketing deserves great measurement.<br/>
  <em>Built with care by marketers who code and data scientists who market.</em>
</p>
