"""Tests for the content moderation module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.utils.moderation import ModerationEngine, ModerationResult, check_prompt


# ── ModerationEngine unit tests ───────────────────────────────────────────────


class TestModerationEngine:
    def test_is_loaded_false_before_load(self):
        engine = ModerationEngine()
        assert not engine.is_loaded

    def test_load_failure_leaves_engine_unloaded(self):
        engine = ModerationEngine()
        with patch("app.utils.moderation.ModerationEngine.load") as mock_load:
            mock_load.side_effect = RuntimeError("no GPU")
            try:
                engine.load("google/gemma-3-1b-it")
            except RuntimeError:
                pass
        assert not engine.is_loaded

    def test_parse_pipeline_output_safe(self):
        engine = ModerationEngine()
        raw = [{"generated_text": [{"role": "assistant", "content": "SAFE"}]}]
        assert engine._parse_pipeline_output(raw) == "SAFE"

    def test_parse_pipeline_output_unsafe(self):
        engine = ModerationEngine()
        raw = [{"generated_text": [{"role": "assistant", "content": "UNSAFE"}]}]
        assert engine._parse_pipeline_output(raw) == "UNSAFE"

    def test_check_with_local_model_raises_when_not_loaded(self):
        engine = ModerationEngine()
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.check_with_local_model("test prompt")

    def test_check_with_local_model_safe(self):
        engine = ModerationEngine()
        engine._pipeline = MagicMock(
            return_value=[{"generated_text": [{"role": "assistant", "content": "SAFE"}]}]
        )
        result = engine.check_with_local_model("a beautiful sunset")
        assert result.is_safe is True
        assert result.source == "local"

    def test_check_with_local_model_unsafe(self):
        engine = ModerationEngine()
        engine._pipeline = MagicMock(
            return_value=[{"generated_text": [{"role": "assistant", "content": "UNSAFE"}]}]
        )
        result = engine.check_with_local_model("explicit harmful content")
        assert result.is_safe is False
        assert result.source == "local"
        assert "Flagged by local model" in result.reason

    def test_check_with_agent_api_safe(self):
        engine = ModerationEngine()
        engine._model_id = "google/gemma-3-1b-it"

        mock_choice = MagicMock()
        mock_choice.message.content = "SAFE"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response

        with patch("huggingface_hub.InferenceClient", return_value=mock_client):
            result = engine.check_with_agent_api("a nice landscape", token="hf_test")

        assert result.is_safe is True
        assert result.source == "agent_api"

    def test_check_with_agent_api_unsafe(self):
        engine = ModerationEngine()
        engine._model_id = "google/gemma-3-1b-it"

        mock_choice = MagicMock()
        mock_choice.message.content = "UNSAFE"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response

        with patch("huggingface_hub.InferenceClient", return_value=mock_client):
            result = engine.check_with_agent_api("explicit content", token="hf_test")

        assert result.is_safe is False
        assert result.source == "agent_api"
        assert "Flagged by agent API" in result.reason

    def test_check_uses_local_model_first(self):
        engine = ModerationEngine()
        engine._pipeline = MagicMock(
            return_value=[{"generated_text": [{"role": "assistant", "content": "SAFE"}]}]
        )

        with patch.object(engine, "check_with_agent_api") as mock_api:
            result = engine.check("safe prompt")

        assert result.is_safe is True
        assert result.source == "local"
        mock_api.assert_not_called()

    def test_check_falls_back_to_agent_api_when_local_fails(self):
        engine = ModerationEngine()
        engine._pipeline = MagicMock(side_effect=RuntimeError("inference error"))
        engine._model_id = "google/gemma-3-1b-it"

        fallback_result = ModerationResult(is_safe=True, source="agent_api")
        with patch.object(engine, "check_with_agent_api", return_value=fallback_result) as mock_api:
            result = engine.check("test prompt", hf_token="hf_test")

        assert result.source == "agent_api"
        mock_api.assert_called_once()

    def test_check_fails_open_when_both_unavailable(self):
        engine = ModerationEngine()
        # Local not loaded, no model_id so agent API branch skipped
        result = engine.check("test prompt")
        assert result.is_safe is True
        assert result.source == "none"

    def test_check_falls_open_when_agent_api_also_fails(self):
        engine = ModerationEngine()
        engine._model_id = "google/gemma-3-1b-it"

        with patch.object(engine, "check_with_agent_api", side_effect=Exception("API error")):
            result = engine.check("test prompt", hf_token="hf_test")

        assert result.is_safe is True
        assert result.source == "none"


# ── check_prompt() integration tests ─────────────────────────────────────────


class TestCheckPrompt:
    def test_allows_safe_prompt(self):
        with patch("app.utils.moderation._moderation_engine") as mock_engine:
            mock_engine.check.return_value = ModerationResult(is_safe=True, source="local")
            check_prompt("a beautiful sunset")  # should not raise

    def test_rejects_unsafe_prompt(self):
        with patch("app.utils.moderation._moderation_engine") as mock_engine:
            mock_engine.check.return_value = ModerationResult(
                is_safe=False, reason="Flagged by local model: UNSAFE", source="local"
            )
            with pytest.raises(ValueError, match="Flagged by local model"):
                check_prompt("explicit content")

    def test_skips_when_moderation_disabled(self):
        with patch("app.config.settings") as mock_settings, patch(
            "app.utils.moderation._moderation_engine"
        ) as mock_engine:
            mock_settings.MODERATION_ENABLED = False
            check_prompt("any content")
            mock_engine.check.assert_not_called()


# ── ImageRequest schema integration ──────────────────────────────────────────


class TestImageRequestModeration:
    def test_unsafe_prompt_raises_validation_error(self):
        with patch("app.utils.moderation._moderation_engine") as mock_engine:
            mock_engine.check.return_value = ModerationResult(
                is_safe=False, reason="Flagged by local model: UNSAFE", source="local"
            )
            with pytest.raises(ValidationError, match="content moderation|Flagged"):
                from app.schemas.generate import ImageRequest

                ImageRequest(prompt="explicit content")

    def test_safe_prompt_passes_validation(self):
        with patch("app.utils.moderation._moderation_engine") as mock_engine:
            mock_engine.check.return_value = ModerationResult(is_safe=True, source="local")
            from app.schemas.generate import ImageRequest

            req = ImageRequest(prompt="a mountain landscape")
            assert req.prompt == "a mountain landscape"
