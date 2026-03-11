"""Content moderation service – checks prompt safety via a third-party API."""

from __future__ import annotations

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class ModerationError(Exception):
    """Raised when the moderation API call fails unexpectedly."""


async def check_prompt_safety(prompt: str) -> tuple[bool, str | None]:
    """Check whether a prompt is safe using the configured moderation API.

    Calls the API specified by ``MODERATION_API_URL`` (default: OpenAI
    moderation endpoint) and returns ``(is_safe, reason)``.  When moderation
    is disabled or no API key is configured the function returns ``(True, None)``
    without making any network request.

    Args:
        prompt: The user-supplied generation prompt to evaluate.

    Returns:
        A ``(is_safe, reason)`` tuple.  ``is_safe`` is ``True`` when the prompt
        is considered safe.  When flagged, ``reason`` contains the triggered
        content categories (e.g. ``"sexual, violence"``).

    Raises:
        ModerationError: If the API request fails and the failure cannot be
            handled gracefully.
    """
    if not settings.MODERATION_ENABLED:
        return True, None

    if not settings.MODERATION_API_KEY:
        logger.warning(
            "Content moderation is enabled but MODERATION_API_KEY is not set; "
            "skipping moderation check."
        )
        return True, None

    try:
        async with httpx.AsyncClient(timeout=settings.MODERATION_TIMEOUT) as client:
            response = await client.post(
                settings.MODERATION_API_URL,
                headers={
                    "Authorization": f"Bearer {settings.MODERATION_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"input": prompt},
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Moderation API returned an error status %s: %s",
            exc.response.status_code,
            exc.response.text,
        )
        raise ModerationError(
            f"Moderation API returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        logger.error("Moderation API request failed: %s", exc)
        raise ModerationError(f"Moderation API request failed: {exc}") from exc

    # Parse OpenAI-compatible moderation response:
    # {"results": [{"flagged": bool, "categories": {category: bool, ...}}]}
    results = data.get("results", [])
    if not results:
        return True, None

    result = results[0]
    if result.get("flagged", False):
        categories: dict[str, bool] = result.get("categories", {})
        flagged_categories = [cat for cat, triggered in categories.items() if triggered]
        reason = ", ".join(flagged_categories) if flagged_categories else "unsafe content"
        return False, reason

    return True, None
