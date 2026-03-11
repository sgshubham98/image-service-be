"""Tests for the content moderation service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.services.moderation_service import ModerationError, check_prompt_safety


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_moderation_response(flagged: bool, categories: dict[str, bool] | None = None):
    """Build a mock httpx.Response that mimics the OpenAI moderation format."""
    if categories is None:
        categories = {}
    payload = {
        "results": [
            {
                "flagged": flagged,
                "categories": categories,
                "category_scores": {k: 0.9 if v else 0.01 for k, v in categories.items()},
            }
        ]
    }
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


# ---------------------------------------------------------------------------
# Tests: moderation disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_prompt_safety_disabled_skips_api():
    """When moderation is disabled the function returns safe without any API call."""
    with patch("app.services.moderation_service.settings") as mock_settings:
        mock_settings.MODERATION_ENABLED = False
        is_safe, reason = await check_prompt_safety("a test prompt")

    assert is_safe is True
    assert reason is None


@pytest.mark.asyncio
async def test_check_prompt_safety_no_api_key_skips_api():
    """When enabled but no API key is set, the function returns safe with a warning."""
    with patch("app.services.moderation_service.settings") as mock_settings:
        mock_settings.MODERATION_ENABLED = True
        mock_settings.MODERATION_API_KEY = ""
        is_safe, reason = await check_prompt_safety("a test prompt")

    assert is_safe is True
    assert reason is None


# ---------------------------------------------------------------------------
# Tests: safe prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_prompt_safety_safe_prompt():
    """A prompt that the API says is safe returns (True, None)."""
    mock_response = _make_moderation_response(flagged=False)

    with patch("app.services.moderation_service.settings") as mock_settings, patch(
        "app.services.moderation_service.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_settings.MODERATION_ENABLED = True
        mock_settings.MODERATION_API_KEY = "test-key"
        mock_settings.MODERATION_API_URL = "https://api.openai.com/v1/moderations"
        mock_settings.MODERATION_TIMEOUT = 10.0

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        is_safe, reason = await check_prompt_safety("a beautiful sunset over the mountains")

    assert is_safe is True
    assert reason is None


# ---------------------------------------------------------------------------
# Tests: unsafe / flagged prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_prompt_safety_flagged_prompt():
    """A flagged prompt returns (False, reason) with the triggered categories."""
    mock_response = _make_moderation_response(
        flagged=True,
        categories={"sexual": True, "violence": False, "hate": False},
    )

    with patch("app.services.moderation_service.settings") as mock_settings, patch(
        "app.services.moderation_service.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_settings.MODERATION_ENABLED = True
        mock_settings.MODERATION_API_KEY = "test-key"
        mock_settings.MODERATION_API_URL = "https://api.openai.com/v1/moderations"
        mock_settings.MODERATION_TIMEOUT = 10.0

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        is_safe, reason = await check_prompt_safety("explicit nsfw content here")

    assert is_safe is False
    assert reason is not None
    assert "sexual" in reason


@pytest.mark.asyncio
async def test_check_prompt_safety_flagged_multiple_categories():
    """Multiple triggered categories are all included in the reason string."""
    mock_response = _make_moderation_response(
        flagged=True,
        categories={"sexual": True, "violence": True, "hate": False},
    )

    with patch("app.services.moderation_service.settings") as mock_settings, patch(
        "app.services.moderation_service.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_settings.MODERATION_ENABLED = True
        mock_settings.MODERATION_API_KEY = "test-key"
        mock_settings.MODERATION_API_URL = "https://api.openai.com/v1/moderations"
        mock_settings.MODERATION_TIMEOUT = 10.0

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        is_safe, reason = await check_prompt_safety("violent explicit content")

    assert is_safe is False
    assert "sexual" in reason
    assert "violence" in reason


@pytest.mark.asyncio
async def test_check_prompt_safety_flagged_no_categories():
    """If the API flags the prompt but provides no categories, reason is generic."""
    mock_response = _make_moderation_response(flagged=True, categories={})

    with patch("app.services.moderation_service.settings") as mock_settings, patch(
        "app.services.moderation_service.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_settings.MODERATION_ENABLED = True
        mock_settings.MODERATION_API_KEY = "test-key"
        mock_settings.MODERATION_API_URL = "https://api.openai.com/v1/moderations"
        mock_settings.MODERATION_TIMEOUT = 10.0

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        is_safe, reason = await check_prompt_safety("some flagged content")

    assert is_safe is False
    assert reason == "unsafe content"


# ---------------------------------------------------------------------------
# Tests: API failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_prompt_safety_http_error_raises_moderation_error():
    """A network-level HTTP error raises ModerationError."""
    with patch("app.services.moderation_service.settings") as mock_settings, patch(
        "app.services.moderation_service.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_settings.MODERATION_ENABLED = True
        mock_settings.MODERATION_API_KEY = "test-key"
        mock_settings.MODERATION_API_URL = "https://api.openai.com/v1/moderations"
        mock_settings.MODERATION_TIMEOUT = 10.0

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        with pytest.raises(ModerationError):
            await check_prompt_safety("test prompt")


@pytest.mark.asyncio
async def test_check_prompt_safety_http_status_error_raises_moderation_error():
    """A non-2xx API response raises ModerationError."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    http_error = httpx.HTTPStatusError(
        "401 Unauthorized",
        request=MagicMock(),
        response=mock_response,
    )
    mock_response.raise_for_status = MagicMock(side_effect=http_error)

    with patch("app.services.moderation_service.settings") as mock_settings, patch(
        "app.services.moderation_service.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_settings.MODERATION_ENABLED = True
        mock_settings.MODERATION_API_KEY = "bad-key"
        mock_settings.MODERATION_API_URL = "https://api.openai.com/v1/moderations"
        mock_settings.MODERATION_TIMEOUT = 10.0

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        with pytest.raises(ModerationError):
            await check_prompt_safety("test prompt")


@pytest.mark.asyncio
async def test_check_prompt_safety_empty_results():
    """An API response with no results array is treated as safe."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = MagicMock()

    with patch("app.services.moderation_service.settings") as mock_settings, patch(
        "app.services.moderation_service.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_settings.MODERATION_ENABLED = True
        mock_settings.MODERATION_API_KEY = "test-key"
        mock_settings.MODERATION_API_URL = "https://api.openai.com/v1/moderations"
        mock_settings.MODERATION_TIMEOUT = 10.0

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        is_safe, reason = await check_prompt_safety("a normal prompt")

    assert is_safe is True
    assert reason is None
