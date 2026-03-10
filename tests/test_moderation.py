"""Tests for prompt content moderation."""

import pytest
from pydantic import ValidationError

from app.utils.moderation import check_prompt, BLOCKED_TERMS
from app.schemas.generate import ImageRequest


# ── Unit tests for check_prompt ──────────────────────────────────


class TestCheckPrompt:
    """Unit tests for the moderation utility."""

    def test_safe_prompt_passes(self):
        assert check_prompt("a beautiful sunset over the ocean") == "a beautiful sunset over the ocean"

    def test_safe_prompt_with_common_substrings(self):
        """Words that contain blocked substrings should NOT be flagged."""
        # "therapist" contains a substring that could be a false positive
        assert check_prompt("a therapist in an office") == "a therapist in an office"

    def test_blocks_nsfw_term(self):
        with pytest.raises(ValueError, match="content policy"):
            check_prompt("a nude woman")

    def test_blocks_nsfw_term_case_insensitive(self):
        with pytest.raises(ValueError, match="content policy"):
            check_prompt("NSFW artwork")

    def test_blocks_nsfw_term_mixed_case(self):
        with pytest.raises(ValueError, match="content policy"):
            check_prompt("Explicit Content in a painting")

    def test_blocks_violence_term(self):
        with pytest.raises(ValueError, match="content policy"):
            check_prompt("a scene of dismemberment")

    def test_blocks_child_safety_term(self):
        with pytest.raises(ValueError, match="content policy"):
            check_prompt("child exploitation imagery")

    def test_blocks_hate_term(self):
        with pytest.raises(ValueError, match="content policy"):
            check_prompt("ethnic cleansing propaganda")

    def test_blocks_multi_word_phrase(self):
        with pytest.raises(ValueError, match="content policy"):
            check_prompt("sexual intercourse in public")

    def test_safe_prompt_artistic_context(self):
        """Legitimate art prompts should pass."""
        assert check_prompt("renaissance painting of a classical statue") is not None

    def test_safe_prompt_landscape(self):
        assert check_prompt("a mountain landscape at golden hour") is not None

    def test_all_blocked_terms_are_caught(self):
        """Verify every term in the blocked set is actually detected."""
        for term in BLOCKED_TERMS:
            with pytest.raises(ValueError, match="content policy"):
                check_prompt(f"generate {term} image")


# ── Schema-level integration tests ──────────────────────────────


class TestImageRequestModeration:
    """Moderation is enforced at the Pydantic schema level."""

    def test_safe_prompt_accepted(self):
        req = ImageRequest(prompt="a cat sitting on a windowsill")
        assert req.prompt == "a cat sitting on a windowsill"

    def test_nsfw_prompt_rejected(self):
        with pytest.raises(ValidationError, match="content policy"):
            ImageRequest(prompt="nude figure in a field")

    def test_nsfw_prompt_rejected_in_batch_context(self):
        """BatchRequest inherits moderation via ImageRequest."""
        from app.schemas.batch import BatchRequest

        with pytest.raises(ValidationError, match="content policy"):
            BatchRequest(
                name="test",
                prompts=[
                    ImageRequest(prompt="safe prompt"),
                    ImageRequest(prompt="pornographic material"),
                ],
            )


# ── API endpoint integration tests ──────────────────────────────


@pytest.mark.asyncio
async def test_generate_rejects_nsfw_prompt(client):
    resp = await client.post("/generate", json={"prompt": "nude woman standing"})
    assert resp.status_code == 422
    assert "content policy" in resp.json()["detail"][0]["msg"]


@pytest.mark.asyncio
async def test_generate_sync_rejects_nsfw_prompt(client):
    resp = await client.post("/generate/sync", json={"prompt": "generate hentai art"})
    assert resp.status_code == 422
    assert "content policy" in resp.json()["detail"][0]["msg"]


@pytest.mark.asyncio
async def test_batch_rejects_nsfw_prompt(client):
    resp = await client.post(
        "/batch",
        json={
            "name": "bad-batch",
            "prompts": [
                {"prompt": "a nice landscape"},
                {"prompt": "pornographic scene"},
            ],
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_safe_prompt_still_works(client):
    """Ensure moderation does not block legitimate prompts."""
    resp = await client.post("/generate", json={"prompt": "a beautiful sunset"})
    assert resp.status_code == 200
