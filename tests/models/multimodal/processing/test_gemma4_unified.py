# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping

import pytest
import torch
from PIL import Image as PILImage

from vllm.model_executor.models.gemma4_mm import Gemma4ImagePixelInputs
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig

from ....conftest import ImageTestAssets
from ...utils import build_model_context

# The Unified model ID for testing purposes
GEMMA4_UNIFIED_MODEL_ID = "google/gemma-4-12B-it"


def test_gemma4_unified_image_schema_accepts_variable_patch_counts():
    Gemma4ImagePixelInputs(
        pixel_values=[
            torch.randn(10080, 768),
            torch.randn(2520, 768),
        ],
        pixel_position_ids=[
            torch.zeros(10080, 2, dtype=torch.long),
            torch.zeros(2520, 2, dtype=torch.long),
        ],
    )


def test_gemma4_unified_image_batching_keeps_variable_patch_counts_unstacked():
    field = MultiModalFieldConfig.batched("image").field
    elems = field.build_elems(
        "image",
        "pixel_values",
        [torch.randn(10080, 768), torch.randn(2520, 768)],
    )

    reduced = field.reduce_data(list(elems))

    assert isinstance(reduced, list)
    assert [tensor.shape for tensor in reduced] == [
        torch.Size([10080, 768]),
        torch.Size([2520, 768]),
    ]


@pytest.mark.parametrize(
    "image_width,image_height,max_soft_tokens",
    [
        (900, 3, 280),
        (3, 900, 280),
        (900, 3, 70),
        (4000, 2, 1120),
    ],
)
@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens(
    model_id: str,
    image_width: int,
    image_height: int,
    max_soft_tokens: int,
):
    """Verify ``_compute_num_soft_tokens`` caps output at ``max_soft_tokens``."""
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={"do_pan_and_scan": True},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    num_soft_tokens = processor.info._compute_num_soft_tokens(
        image_width=image_width,
        image_height=image_height,
        max_soft_tokens=max_soft_tokens,
    )

    assert num_soft_tokens <= max_soft_tokens, (
        f"_compute_num_soft_tokens returned {num_soft_tokens} for "
        f"image_width={image_width}, image_height={image_height}, "
        f"max_soft_tokens={max_soft_tokens} — exceeds the cap."
    )


@pytest.mark.parametrize(
    ("mm_processor_kwargs", "expected_image_tokens"),
    [
        ({}, 280),
        ({"max_soft_tokens": 70}, 70),
        ({"max_soft_tokens": 280}, 280),
        ({"max_soft_tokens": 1120}, 1120),
        ({"images_kwargs": {"max_soft_tokens": 560}}, 560),
        ({"images_kwargs": None}, 280),
        ({"images_kwargs": "not-a-dict"}, 280),
    ],
)
@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens(
    model_id: str,
    mm_processor_kwargs: dict[str, object],
    expected_image_tokens: int,
):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    tokens = processor.info.get_mm_max_tokens_per_item(
        seq_len=ctx.model_config.max_model_len,
        mm_counts={"image": 1, "video": 1},
    )

    assert tokens is not None
    assert tokens["image"] == expected_image_tokens
    assert tokens["video"] == 32 * (70 + 2 + 6)


@pytest.mark.parametrize(
    ("limit_mm_per_prompt", "expected_video_tokens"),
    [
        ({"video": 1}, 32 * (70 + 2 + 6)),
        ({"video": {"count": 1}}, 32 * (70 + 2 + 6)),
        ({"video": {"count": 1, "num_frames": 1}}, 1 * (70 + 2 + 6)),
        ({"video": {"count": 1, "num_frames": 8}}, 8 * (70 + 2 + 6)),
        ({"video": {"count": 1, "num_frames": 32}}, 32 * (70 + 2 + 6)),
        ({"video": {"count": 1, "num_frames": 40}}, 32 * (70 + 2 + 6)),
    ],
)
@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_get_mm_max_tokens_per_item_respects_configured_video_num_frames(
    model_id: str,
    limit_mm_per_prompt: Mapping[str, int | Mapping[str, int]],
    expected_video_tokens: int,
):
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    tokens = processor.info.get_mm_max_tokens_per_item(
        seq_len=ctx.model_config.max_model_len,
        mm_counts={"video": 1},
    )

    assert tokens is not None
    assert tokens["image"] == 280
    assert tokens["video"] == expected_video_tokens


@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_get_prompt_updates_respects_nested_max_soft_tokens(model_id: str):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={"images_kwargs": {"max_soft_tokens": 560}},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    image = PILImage.new("RGB", (1000, 1000), color="white")
    image_size = image.size
    mm_items = processor.info.parse_mm_data({"image": image})

    prompt_update = processor._get_prompt_updates(mm_items, {}, {})[0]
    replacement = prompt_update.resolve(0).content.full
    expected = processor.info.get_image_repl(
        image_width=image_size[0],
        image_height=image_size[1],
        processor=processor.info.get_hf_processor(),
        max_soft_tokens=560,
    ).full

    assert replacement == expected


@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_limit_mm_per_prompt(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that limit_mm_per_prompt restricts multiple images correctly."""
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<image><image>"
    images = [asset.pil_image for asset in image_assets][:2]
    if len(images) < 2:
        images = [images[0], images[0]]

    mm_data = {"image": images}

    with pytest.raises(ValueError, match="At most 1 image"):
        processor(
            prompt,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={},
        )


def test_gemma4_unified_maps_nested_embed_vision_checkpoint_keys():
    """AxionML-style NVFP4 checkpoints nest the vision pipeline under
    model.embed_vision.*: patch/pos weights belong to vision_embedder, and
    the projection lives in multimodal_embedder.* -> embed_vision."""
    pytest.importorskip("transformers.models.gemma4_unified")
    from vllm.model_executor.models.gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration,
    )

    mapper = Gemma4UnifiedForConditionalGeneration.hf_to_vllm_mapper
    checkpoint_keys = {
        "model.embed_audio.embedding_projection.weight": 0,
        "model.embed_vision.multimodal_embedder.embedding_projection.weight": 0,
        "model.embed_vision.patch_dense.weight": 0,
        "model.embed_vision.patch_dense.bias": 0,
        "model.embed_vision.patch_ln1.weight": 0,
        "model.embed_vision.patch_ln2.bias": 0,
        "model.embed_vision.pos_embedding": 0,
        "model.embed_vision.pos_norm.weight": 0,
        "model.language_model.embed_tokens.weight": 0,
    }

    mapped = set(mapper.apply_dict(checkpoint_keys))

    assert mapped == {
        "embed_audio.embedding_projection.weight",
        "embed_vision.embedding_projection.weight",
        "vision_embedder.patch_dense.weight",
        "vision_embedder.patch_dense.bias",
        "vision_embedder.patch_ln1.weight",
        "vision_embedder.patch_ln2.bias",
        "vision_embedder.pos_embedding",
        "vision_embedder.pos_norm.weight",
        "language_model.model.embed_tokens.weight",
    }


def test_gemma4_unified_maps_flat_embed_vision_projection_key():
    """The generic model.embed_vision. rule still routes a directly-nested
    projection (non-multimodal_embedder layout) to embed_vision."""
    pytest.importorskip("transformers.models.gemma4_unified")
    from vllm.model_executor.models.gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration,
    )

    mapper = Gemma4UnifiedForConditionalGeneration.hf_to_vllm_mapper

    assert mapper.apply_list(
        ["model.embed_vision.embedding_projection.weight"]
    ) == ["embed_vision.embedding_projection.weight"]


def test_vision_patch_pixels_derives_from_patch_and_pooling():
    """patch_dense input must be (patch_size * pooling_kernel_size)**2 * 3,
    matching the processor's pooled unit. AxionML/Gemma-4-12B-NVFP4 uses
    patch_size=16, pooling_kernel_size=3 -> 6912; the standard pooling=1 case
    stays at patch_size**2 * 3 -> 768."""
    pytest.importorskip("transformers.models.gemma4_unified")
    from types import SimpleNamespace

    from vllm.model_executor.models.gemma4_unified import _vision_patch_pixels

    axionml = SimpleNamespace(patch_size=16, pooling_kernel_size=3)
    assert _vision_patch_pixels(axionml) == 6912

    standard = SimpleNamespace(patch_size=16, pooling_kernel_size=1)
    assert _vision_patch_pixels(standard) == 768
