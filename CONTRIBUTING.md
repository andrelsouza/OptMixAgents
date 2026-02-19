# Contributing to OptMix

First off, thank you for considering contributing to OptMix! This project thrives on community input from marketing scientists, data engineers, ML practitioners, and marketing professionals.

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating a bug report, please check existing issues. When creating a report, include:

- A clear, descriptive title
- Steps to reproduce the behavior
- Expected vs actual behavior
- Your environment (Python version, OS, package versions)
- Sample data or code snippet if possible

### 💡 Suggesting Features

Feature requests are welcome! Please include:

- A clear description of the problem you're trying to solve
- How this feature fits into the OptMix workflow
- Whether it affects agents, the MMM engine, connectors, or workflows
- Any reference implementations or papers

### 🔧 Pull Requests

#### First Time?

1. Fork the repo and clone locally
2. Create a branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Make your changes
5. Run tests: `pytest`
6. Run linting: `ruff check . && ruff format .`
7. Push and open a PR

#### What We're Looking For

**MMM Models & Transforms**
- New model backends (Prophet-based, neural MMM, etc.)
- Custom adstock/saturation functions
- Model diagnostics and validation tools
- Benchmarking against known datasets

**Agent Definitions**
- Improved personas and communication styles
- New domain-specific agents (Brand Analyst, CRM Specialist, etc.)
- Better tool definitions and workflows
- Agent knowledge base expansions

**Data Connectors**
- New platform integrations (TikTok Ads, LinkedIn Ads, Snapchat, etc.)
- Data warehouse connectors (Snowflake, Redshift, Databricks)
- CDP integrations (Segment, mParticle)
- Improved data normalization and merging

**Visualization & Reports**
- Chart templates and themes
- Report format improvements
- Dashboard components
- Interactive visualization tools

**Documentation & Examples**
- Tutorials and cookbooks
- Case studies with sample data
- Video walkthroughs
- Translation to other languages

**Sample Datasets**
- Synthetic marketing datasets with known ground truth
- Industry-specific data generators
- Benchmark datasets for model comparison

## Development Setup

```bash
git clone https://github.com/optmix-ai/optmix.git
cd optmix
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pre-commit install
```

## Code Style

- We use **ruff** for linting and formatting
- Type hints are required for all public functions
- Docstrings follow Google style
- Maximum line length: 100 characters

## Agent Development Guidelines

When creating or modifying agents, follow the Agent-as-Code pattern:

1. **YAML First**: Define the agent in `.agent.yaml` before writing any Python
2. **Persona Matters**: Give agents distinct personalities — it affects output quality
3. **Tools Are Functions**: Each tool maps to a callable Python function
4. **Workflows Are YAML**: Multi-step processes are defined declaratively
5. **Checklists Are Gates**: Quality gates must pass before moving to next phase

## Commit Messages

Use conventional commits:

```
feat: add TikTok Ads connector
fix: correct adstock decay calculation for weekly data
docs: add tutorial for custom agent creation
test: add integration tests for BayesianMMM
refactor: simplify budget optimizer constraint handling
```

## Testing

- Unit tests go in `tests/unit/`
- Integration tests go in `tests/integration/`
- All MMM model changes must include tests with known-output datasets
- Agent changes should include conversation simulation tests

## Code of Conduct

Be kind, be constructive, be respectful. We're building tools to democratize marketing measurement — that mission requires collaboration from diverse perspectives.

## Questions?

Open a Discussion on GitHub or reach out to the maintainers. No question is too basic.

---

Thank you for helping make marketing measurement accessible to everyone! 🚀
