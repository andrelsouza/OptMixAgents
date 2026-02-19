"""Tests for CLI setup wizard and config-aware commands."""

import pytest
import yaml
from click.testing import CliRunner

from optmix.cli.main import cli


@pytest.fixture()
def runner():
    return CliRunner()


class TestSetupWizard:
    """Test the 'optmix setup' interactive command."""

    def test_setup_saves_config(self, runner, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        monkeypatch.setattr("optmix.core.config.CONFIG_FILE", config_path)
        monkeypatch.setattr("optmix.core.config.CONFIG_DIR", tmp_path)

        # Simulate: choose Anthropic (1), enter API key, accept default model, skip test
        result = runner.invoke(cli, ["setup"], input="1\nsk-test-key-12345\n\nn\n")

        assert result.exit_code == 0
        assert config_path.exists()
        data = yaml.safe_load(config_path.read_text())
        assert data["provider"] == "anthropic"
        assert data["api_key"] == "sk-test-key-12345"
        assert "Setup Complete" in result.output

    def test_setup_openai_provider(self, runner, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        monkeypatch.setattr("optmix.core.config.CONFIG_FILE", config_path)
        monkeypatch.setattr("optmix.core.config.CONFIG_DIR", tmp_path)

        # Choose OpenAI (2), enter key, accept default model, skip test
        result = runner.invoke(cli, ["setup"], input="2\nsk-openai-key\n\nn\n")

        assert result.exit_code == 0
        data = yaml.safe_load(config_path.read_text())
        assert data["provider"] == "openai"
        assert data["api_key"] == "sk-openai-key"

    def test_setup_empty_key_cancels(self, runner, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        monkeypatch.setattr("optmix.core.config.CONFIG_FILE", config_path)
        monkeypatch.setattr("optmix.core.config.CONFIG_DIR", tmp_path)

        # Choose Anthropic, empty key
        result = runner.invoke(cli, ["setup"], input="1\n\n")

        assert result.exit_code == 0
        assert "cancelled" in result.output.lower() or not config_path.exists()


class TestCLIHelpers:
    """Test the CLI helper functions."""

    def test_parse_llm_flag_with_provider_and_model(self):
        from optmix.cli.main import _parse_llm_flag

        provider, model = _parse_llm_flag("openai/gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_parse_llm_flag_provider_only(self):
        from optmix.cli.main import _parse_llm_flag

        provider, model = _parse_llm_flag("anthropic")
        assert provider == "anthropic"
        assert model is None

    def test_parse_llm_flag_none(self):
        from optmix.cli.main import _parse_llm_flag

        provider, model = _parse_llm_flag(None)
        assert provider is None
        assert model is None


class TestCLIConfigIntegration:
    """Test that CLI commands use config resolution."""

    def test_chat_help_shows_llm_option(self, runner):
        result = runner.invoke(cli, ["chat", "--help"])
        assert "--llm" in result.output
        assert "--api-key" in result.output

    def test_run_help_shows_llm_option(self, runner):
        result = runner.invoke(cli, ["run", "--help"])
        assert "--llm" in result.output

    def test_agent_help_shows_llm_option(self, runner):
        result = runner.invoke(cli, ["agent", "--help"])
        assert "--llm" in result.output

    def test_setup_in_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert "setup" in result.output
