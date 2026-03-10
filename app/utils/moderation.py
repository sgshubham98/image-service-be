"""Prompt content moderation — blocks NSFW and unsafe prompts."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Blocked term categories
# Each entry is matched as a whole-word (case-insensitive) so that
# sub-strings inside innocent words do not trigger false positives.
# ---------------------------------------------------------------------------

_SEXUAL_TERMS: set[str] = {
    "nude",
    "nudes",
    "naked",
    "nudity",
    "nsfw",
    "topless",
    "bottomless",
    "genitalia",
    "pornographic",
    "pornography",
    "xxx",
    "erotic",
    "erotica",
    "hentai",
    "orgasm",
    "sexual intercourse",
    "sexually explicit",
    "explicit content",
}

_VIOLENCE_TERMS: set[str] = {
    "gore",
    "gory",
    "dismemberment",
    "mutilation",
    "decapitation",
    "evisceration",
    "torture porn",
}

_CHILD_SAFETY_TERMS: set[str] = {
    "child abuse",
    "child exploitation",
    "underage sexual",
    "minor sexual",
    "pedophilia",
    "csam",
    "lolicon",
    "shotacon",
}

_HATE_SPEECH_TERMS: set[str] = {
    "white supremacy",
    "ethnic cleansing",
    "racial slur",
    "hate crime",
}

# Combined set used for matching.
BLOCKED_TERMS: frozenset[str] = frozenset(
    _SEXUAL_TERMS | _VIOLENCE_TERMS | _CHILD_SAFETY_TERMS | _HATE_SPEECH_TERMS
)

# Pre-compiled regex: alternation of all blocked phrases wrapped in
# word-boundary anchors so we only match whole words / phrases.
_BLOCKED_RE: re.Pattern[str] = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in sorted(BLOCKED_TERMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def check_prompt(text: str) -> str:
    """Validate *text* against the blocked-term list.

    Returns the original text unchanged when it passes moderation.
    Raises ``ValueError`` with a user-facing message when a blocked
    term is detected — suitable for use inside Pydantic validators.
    """
    match = _BLOCKED_RE.search(text)
    if match:
        raise ValueError(
            "Prompt rejected: your input may violate our content policy. "
            "Please revise the prompt and try again."
        )
    return text
