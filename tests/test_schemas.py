"""Tests for schemas — resolution validation, format handling."""

import pytest
from pydantic import ValidationError

from app.schemas.generate import ImageRequest
from app.schemas.batch import BatchRequest


class TestImageRequest:
    def test_defaults(self):
        req = ImageRequest(prompt="test")
        assert req.width == 1024
        assert req.height == 1024
        assert req.num_steps == 4
        assert req.guidance_scale == 0.0
        assert req.format.value == "png"
        assert req.seed is None
        assert req.num_images == 1

    def test_num_images_up_to_4(self):
        req = ImageRequest(prompt="test", num_images=4)
        assert req.num_images == 4

    def test_num_images_rejects_above_4(self):
        with pytest.raises(ValidationError):
            ImageRequest(prompt="test", num_images=5)

    def test_num_images_rejects_zero(self):
        with pytest.raises(ValidationError):
            ImageRequest(prompt="test", num_images=0)

    def test_width_must_be_multiple_of_8(self):
        with pytest.raises(ValidationError, match="multiple of 8"):
            ImageRequest(prompt="test", width=513)

    def test_height_must_be_multiple_of_8(self):
        with pytest.raises(ValidationError, match="multiple of 8"):
            ImageRequest(prompt="test", height=100 + 3)

    def test_valid_custom_resolution(self):
        req = ImageRequest(prompt="test", width=768, height=1344)
        assert req.width == 768
        assert req.height == 1344

    def test_rejects_empty_prompt(self):
        with pytest.raises(ValidationError):
            ImageRequest(prompt="")

    def test_rejects_oversized_resolution(self):
        with pytest.raises(ValidationError):
            ImageRequest(prompt="test", width=4096)

    def test_seed_bounds(self):
        req = ImageRequest(prompt="test", seed=0)
        assert req.seed == 0

        req = ImageRequest(prompt="test", seed=2**32 - 1)
        assert req.seed == 2**32 - 1

        with pytest.raises(ValidationError):
            ImageRequest(prompt="test", seed=-1)

    def test_all_formats(self):
        for fmt in ("png", "jpeg", "webp"):
            req = ImageRequest(prompt="test", format=fmt)
            assert req.format.value == fmt


class TestBatchRequest:
    def test_valid_batch(self):
        batch = BatchRequest(
            name="test",
            prompts=[ImageRequest(prompt="a"), ImageRequest(prompt="b")],
        )
        assert len(batch.prompts) == 2

    def test_empty_prompts_rejected(self):
        with pytest.raises(ValidationError):
            BatchRequest(name="empty", prompts=[])
