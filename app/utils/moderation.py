"""Content moderation using Gemma-3-1b-it (local) with HF Inference API fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_SAFETY_SYSTEM_PROMPT = (
    "You are a content safety classifier. "
    "Determine whether the following text prompt for image generation contains "
    "harmful, illegal, sexually explicit, or violent content. "
    "Respond with exactly one word: 'SAFE' if the prompt is acceptable, "
    "or 'UNSAFE' if it should be rejected. Do not include any other text."
)


@dataclass
class ModerationResult:
    is_safe: bool
    reason: str = ""
    source: str = ""


class ModerationEngine:
    """Content moderation engine.

    Primary check: locally-loaded ``google/gemma-3-1b-it`` text-generation pipeline.
    Fallback check: HuggingFace Inference API (agent API) using the same model.
    If both are unavailable the engine fails open and logs a warning.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._model_id: str | None = None

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self, model_id: str = "google/gemma-3-1b-it") -> None:
        """Load *model_id* locally for synchronous inference."""
        try:
            import torch
            from transformers import pipeline

            logger.info("Loading moderation model %s …", model_id)
            self._pipeline = pipeline(
                "text-generation",
                model=model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                max_new_tokens=10,
            )
            self._model_id = model_id
            logger.info("Moderation model %s loaded successfully.", model_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load moderation model %s: %s", model_id, exc)
            self._pipeline = None

    @property
    def is_loaded(self) -> bool:
        """Return True when the local pipeline is ready."""
        return self._pipeline is not None

    # ── Local-model check ────────────────────────────────────────────────────

    def _parse_pipeline_output(self, raw: object) -> str:
        """Extract the assistant reply text from a text-generation pipeline result."""
        # The pipeline returns a list of dicts like:
        # [{"generated_text": [{"role": "system", ...}, {"role": "assistant", "content": "SAFE"}]}]
        if isinstance(raw, list) and raw:
            inner = raw[0].get("generated_text", raw[0])
            if isinstance(inner, list) and inner:
                last = inner[-1]
                if isinstance(last, dict):
                    return last.get("content", "").strip().upper()
            return str(inner).strip().upper()
        return str(raw).strip().upper()

    def check_with_local_model(self, prompt: str) -> ModerationResult:
        """Run the locally-loaded Gemma pipeline and return a :class:`ModerationResult`."""
        if not self.is_loaded:
            raise RuntimeError("Local moderation model is not loaded")
        try:
            messages = [
                {"role": "system", "content": _SAFETY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Prompt: {prompt}"},
            ]
            raw = self._pipeline(messages)
            response = self._parse_pipeline_output(raw)
            is_safe = "UNSAFE" not in response
            return ModerationResult(
                is_safe=is_safe,
                reason="" if is_safe else f"Flagged by local model: {response}",
                source="local",
            )
        except Exception as exc:
            raise RuntimeError(f"Local model inference failed: {exc}") from exc

    # ── Agent-API fallback ───────────────────────────────────────────────────

    def check_with_agent_api(
        self,
        prompt: str,
        *,
        model_id: str | None = None,
        token: str | None = None,
    ) -> ModerationResult:
        """Use the HuggingFace Inference API (agent API) as a fallback safety check."""
        from huggingface_hub import InferenceClient

        _model = model_id or self._model_id or "google/gemma-3-1b-it"
        client = InferenceClient(model=_model, token=token)
        messages = [
            {"role": "system", "content": _SAFETY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Prompt: {prompt}"},
        ]
        result = client.chat_completion(messages=messages, max_tokens=10)
        response = result.choices[0].message.content.strip().upper()
        is_safe = "UNSAFE" not in response
        return ModerationResult(
            is_safe=is_safe,
            reason="" if is_safe else f"Flagged by agent API: {response}",
            source="agent_api",
        )

    # ── Unified entry-point ──────────────────────────────────────────────────

    def check(self, prompt: str, *, hf_token: str | None = None) -> ModerationResult:
        """Check *prompt* safety.

        Tries the local Gemma model first; falls back to the HF Inference API;
        if both are unavailable the call succeeds with a warning (fail-open).
        """
        # 1. Local model
        if self.is_loaded:
            try:
                return self.check_with_local_model(prompt)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Local moderation check failed — falling back to agent API: %s", exc
                )

        # 2. Agent API fallback
        if hf_token or self._model_id:
            try:
                return self.check_with_agent_api(
                    prompt,
                    model_id=self._model_id,
                    token=hf_token,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Agent API moderation check failed — allowing prompt through: %s", exc
                )

        # 3. Both unavailable — fail open
        logger.warning(
            "No moderation backend available; prompt allowed through without safety check."
        )
        return ModerationResult(is_safe=True, reason="Moderation unavailable", source="none")


# ── Module-level singleton ────────────────────────────────────────────────────

_moderation_engine: ModerationEngine = ModerationEngine()


def get_moderation_engine() -> ModerationEngine:
    """Return the global :class:`ModerationEngine` singleton."""
    return _moderation_engine


def check_prompt(prompt: str) -> None:
    """Validate *prompt* against the moderation engine.

    Raises :class:`ValueError` if the prompt is rejected.
    Does nothing when ``MODERATION_ENABLED`` is ``False`` or when no moderation
    backend is configured (fail-open to avoid breaking existing tests).
    """
    from app.config import settings

    if not settings.MODERATION_ENABLED:
        return

    result = _moderation_engine.check(prompt, hf_token=settings.HF_API_TOKEN)
    if not result.is_safe:
        raise ValueError(result.reason or "Prompt rejected by content moderation")
