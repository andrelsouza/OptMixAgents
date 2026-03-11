"""
OptMix Configuration — persistent config for LLM provider and credentials.

Stores user preferences at ~/.optmix/config.yaml with secure file permissions.
Supports a precedence chain: CLI flags > env vars > config file > defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".optmix"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

SUPPORTED_PROVIDERS = ["anthropic", "openai"]

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4o",
}

# Env var names per provider (checked in order)
_PROVIDER_ENV_KEYS: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY", "OPTMIX_API_KEY"],
    "openai": ["OPENAI_API_KEY", "OPTMIX_API_KEY"],
}


@dataclass
class OptMixConfig:
    """User configuration for OptMix."""

    provider: str = "anthropic"
    model: str = ""
    api_key: str = ""

    @property
    def llm_string(self) -> str:
        """Return 'provider/model' string for CLI display."""
        model = self.model or DEFAULT_MODELS.get(self.provider, "")
        return f"{self.provider}/{model}"

    def is_configured(self) -> bool:
        """True if we have a provider and API key."""
        return bool(self.provider and self.api_key)


def load_config(config_path: Path | None = None) -> OptMixConfig:
    """
    Load config from YAML file.

    Returns defaults if file doesn't exist.
    """
    path = config_path or CONFIG_FILE

    if not path.exists():
        return OptMixConfig()

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        return OptMixConfig()
    except yaml.YAMLError:
        logger.warning("Failed to parse config at %s, using defaults.", path)
        return OptMixConfig()

    return OptMixConfig(
        provider=raw.get("provider", "anthropic"),
        model=raw.get("model", ""),
        api_key=raw.get("api_key", ""),
    )


def save_config(config: OptMixConfig, config_path: Path | None = None) -> Path:
    """
    Save config to YAML with restricted permissions (0o600).

    Returns the path written to.
    """
    path = config_path or CONFIG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "provider": config.provider,
        "model": config.model,
        "api_key": config.api_key,
    }

    path.write_text(yaml.dump(data, default_flow_style=False))
    path.chmod(0o600)
    return path


def _resolve_env_key(provider: str) -> str:
    """Look up API key from environment variables for the given provider."""
    env_names = _PROVIDER_ENV_KEYS.get(provider, ["OPTMIX_API_KEY"])
    for name in env_names:
        val = os.environ.get(name)
        if val:
            return val
    return ""


def resolve_config(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    config_path: Path | None = None,
) -> OptMixConfig:
    """
    Build a resolved config using precedence: CLI flags > env vars > config file > defaults.

    Args:
        provider: CLI-supplied provider (overrides everything).
        model: CLI-supplied model (overrides everything).
        api_key: CLI-supplied API key (overrides everything).
        config_path: Custom config file path.

    Returns:
        Fully resolved OptMixConfig.

    Raises:
        ValueError: If provider is not in SUPPORTED_PROVIDERS.
    """
    # Validate provider if provided
    if provider is not None and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider: '{provider}'. Supported providers: {SUPPORTED_PROVIDERS}"
        )

    file_cfg = load_config(config_path)

    # Provider: CLI flag > config file > default
    resolved_provider = provider or file_cfg.provider or "anthropic"

    # Validate resolved provider from config file
    if resolved_provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider in config: '{resolved_provider}'. Supported providers: {SUPPORTED_PROVIDERS}"
        )

    # Model: CLI flag > config file > default for provider
    resolved_model = model or file_cfg.model or DEFAULT_MODELS.get(resolved_provider, "")

    # API key: CLI flag > env var > config file
    resolved_key = api_key or _resolve_env_key(resolved_provider) or file_cfg.api_key

    return OptMixConfig(
        provider=resolved_provider,
        model=resolved_model,
        api_key=resolved_key,
    )
