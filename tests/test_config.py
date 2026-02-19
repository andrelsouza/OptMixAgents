"""Tests for OptMixConfig — load, save, resolve with precedence chain."""

import stat

import pytest
import yaml

from optmix.core.config import (
    DEFAULT_MODELS,
    OptMixConfig,
    load_config,
    resolve_config,
    save_config,
)


@pytest.fixture()
def tmp_config(tmp_path):
    """Return a temp config file path."""
    return tmp_path / "config.yaml"


class TestOptMixConfig:
    """Test the OptMixConfig dataclass."""

    def test_defaults(self):
        cfg = OptMixConfig()
        assert cfg.provider == "anthropic"
        assert cfg.model == ""
        assert cfg.api_key == ""

    def test_is_configured_true(self):
        cfg = OptMixConfig(provider="anthropic", api_key="sk-test")
        assert cfg.is_configured() is True

    def test_is_configured_false_no_key(self):
        cfg = OptMixConfig(provider="anthropic", api_key="")
        assert cfg.is_configured() is False

    def test_is_configured_false_no_provider(self):
        cfg = OptMixConfig(provider="", api_key="sk-test")
        assert cfg.is_configured() is False

    def test_llm_string_with_model(self):
        cfg = OptMixConfig(provider="anthropic", model="claude-sonnet-4-5-20250929")
        assert cfg.llm_string == "anthropic/claude-sonnet-4-5-20250929"

    def test_llm_string_uses_default_model(self):
        cfg = OptMixConfig(provider="openai", model="")
        assert cfg.llm_string == f"openai/{DEFAULT_MODELS['openai']}"


class TestLoadConfig:
    """Test loading config from YAML file."""

    def test_nonexistent_file_returns_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nope.yaml")
        assert cfg.provider == "anthropic"
        assert cfg.api_key == ""

    def test_loads_valid_config(self, tmp_config):
        tmp_config.write_text(
            yaml.dump(
                {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-openai-test",
                }
            )
        )
        cfg = load_config(tmp_config)
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.api_key == "sk-openai-test"

    def test_partial_config_fills_defaults(self, tmp_config):
        tmp_config.write_text(yaml.dump({"provider": "openai"}))
        cfg = load_config(tmp_config)
        assert cfg.provider == "openai"
        assert cfg.model == ""
        assert cfg.api_key == ""

    def test_corrupted_file_returns_defaults(self, tmp_config):
        tmp_config.write_text("{{{{not yaml at all")
        cfg = load_config(tmp_config)
        assert cfg.provider == "anthropic"


class TestSaveConfig:
    """Test saving config to YAML file."""

    def test_creates_directory_and_file(self, tmp_path):
        path = tmp_path / "subdir" / "config.yaml"
        cfg = OptMixConfig(provider="openai", model="gpt-4o", api_key="sk-test")
        save_config(cfg, path)
        assert path.exists()

    def test_file_permissions(self, tmp_config):
        cfg = OptMixConfig(provider="anthropic", api_key="secret")
        save_config(cfg, tmp_config)
        mode = stat.S_IMODE(tmp_config.stat().st_mode)
        assert mode == 0o600

    def test_round_trip(self, tmp_config):
        original = OptMixConfig(provider="openai", model="gpt-4o-mini", api_key="sk-abc123")
        save_config(original, tmp_config)
        loaded = load_config(tmp_config)
        assert loaded.provider == original.provider
        assert loaded.model == original.model
        assert loaded.api_key == original.api_key


class TestResolveConfig:
    """Test config resolution with precedence chain."""

    def test_cli_flags_override_everything(self, tmp_config):
        # Save a config file
        save_config(OptMixConfig(provider="openai", model="gpt-4o", api_key="file-key"), tmp_config)
        # CLI flags should win
        cfg = resolve_config(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            api_key="cli-key",
            config_path=tmp_config,
        )
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-sonnet-4-5-20250929"
        assert cfg.api_key == "cli-key"

    def test_env_var_overrides_config_file(self, tmp_config, monkeypatch):
        save_config(OptMixConfig(provider="anthropic", api_key="file-key"), tmp_config)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        cfg = resolve_config(config_path=tmp_config)
        assert cfg.api_key == "env-key"

    def test_config_file_used_as_fallback(self, tmp_config, monkeypatch):
        # Clear env vars that could interfere
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPTMIX_API_KEY", raising=False)
        save_config(
            OptMixConfig(provider="anthropic", model="claude-3-haiku", api_key="file-key"),
            tmp_config,
        )
        cfg = resolve_config(config_path=tmp_config)
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-3-haiku"
        assert cfg.api_key == "file-key"

    def test_defaults_when_nothing_configured(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPTMIX_API_KEY", raising=False)
        cfg = resolve_config(config_path=tmp_path / "nope.yaml")
        assert cfg.provider == "anthropic"
        assert cfg.model == DEFAULT_MODELS["anthropic"]
        assert cfg.api_key == ""

    def test_openai_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")
        cfg = resolve_config(provider="openai", config_path=tmp_path / "nope.yaml")
        assert cfg.api_key == "sk-openai-env"

    def test_optmix_api_key_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPTMIX_API_KEY", "sk-optmix-shared")
        cfg = resolve_config(config_path=tmp_path / "nope.yaml")
        assert cfg.api_key == "sk-optmix-shared"
