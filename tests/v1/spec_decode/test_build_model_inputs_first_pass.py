# SPDX-License-Identifier: Apache-2.0
"""Tests for SpecDecodeBaseProposer.build_model_inputs_first_pass.

Verifies that is_mm_embed is correctly aligned with input_ids when
set_inputs_first_pass has appended next_token_ids per request.
All tests run without GPU.
"""
from unittest import mock

import torch

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


def _make_proposer(
    *,
    supports_mm_inputs: bool,
    max_num_tokens: int = 32,
    inputs_embeds_size: int = 8,
    pass_hidden_states_to_model: bool = False,
) -> SpecDecodeBaseProposer:
    """Build a SpecDecodeBaseProposer with minimal state (no GPU)."""
    p = object.__new__(SpecDecodeBaseProposer)
    p.supports_mm_inputs = supports_mm_inputs
    p.pass_hidden_states_to_model = pass_hidden_states_to_model
    p.input_ids = torch.arange(max_num_tokens, dtype=torch.int32)
    p.inputs_embeds = torch.zeros(
        (max_num_tokens, inputs_embeds_size), dtype=torch.float32
    )
    p.positions = torch.zeros(max_num_tokens, dtype=torch.int32)
    p.hidden_states = torch.zeros(
        (max_num_tokens, inputs_embeds_size), dtype=torch.float32
    )
    p._get_positions = lambda n: p.positions[:n]
    p.uses_mrope = False
    p.uses_xdrope = False
    return p


def test_is_mm_embed_padded_when_shorter_than_num_tokens():
    """When is_mm_embed.shape[0] < num_tokens, it must be padded with False
    so the multimodal model can broadcast correctly inside embed_input_ids."""
    p = _make_proposer(supports_mm_inputs=True)

    num_tokens = 15
    num_input_tokens = 16

    is_mm_embed = torch.zeros(14, dtype=torch.bool)
    mm_embeds: list[torch.Tensor] = []

    captured = {}

    def fake_embed(input_ids, multimodal_embeddings=None, is_multimodal=None):
        captured["input_ids_shape"] = input_ids.shape
        captured["is_multimodal_shape"] = (
            None if is_multimodal is None else is_multimodal.shape
        )
        captured["is_multimodal_dtype"] = (
            None if is_multimodal is None else is_multimodal.dtype
        )
        return torch.zeros(input_ids.shape[0], 8, dtype=torch.float32)

    p.model = mock.MagicMock()
    p.model.embed_input_ids.side_effect = fake_embed

    p.build_model_inputs_first_pass(
        num_tokens=num_tokens,
        num_input_tokens=num_input_tokens,
        mm_embed_inputs=(mm_embeds, is_mm_embed),
    )

    assert captured["input_ids_shape"] == torch.Size([num_tokens])
    assert captured["is_multimodal_shape"] == torch.Size([num_tokens])
    assert captured["is_multimodal_dtype"] == torch.bool


def test_is_mm_embed_padded_values_are_false():
    """Padded positions must be False (next_token_ids are never multimodal)."""
    p = _make_proposer(supports_mm_inputs=True)

    num_tokens = 15
    num_input_tokens = 15

    is_mm_embed = torch.zeros(14, dtype=torch.bool)
    is_mm_embed[0] = True

    captured_mask = {}

    def fake_embed(input_ids, multimodal_embeddings=None, is_multimodal=None):
        captured_mask["value"] = is_multimodal.clone()
        return torch.zeros(input_ids.shape[0], 8, dtype=torch.float32)

    p.model = mock.MagicMock()
    p.model.embed_input_ids.side_effect = fake_embed

    p.build_model_inputs_first_pass(
        num_tokens=num_tokens,
        num_input_tokens=num_input_tokens,
        mm_embed_inputs=([], is_mm_embed),
    )

    mask = captured_mask["value"]
    assert mask.shape == torch.Size([num_tokens])
    assert mask[0].item() is True
    assert mask[14].item() is False


def test_is_mm_embed_not_modified_when_already_aligned():
    """When is_mm_embed already matches num_tokens, do not change it."""
    p = _make_proposer(supports_mm_inputs=True)

    num_tokens = 16
    num_input_tokens = 16

    is_mm_embed = torch.zeros(num_tokens, dtype=torch.bool)
    is_mm_embed[3] = True

    captured = {}

    def fake_embed(input_ids, multimodal_embeddings=None, is_multimodal=None):
        captured["mask"] = is_multimodal.clone()
        return torch.zeros(input_ids.shape[0], 8, dtype=torch.float32)

    p.model = mock.MagicMock()
    p.model.embed_input_ids.side_effect = fake_embed

    p.build_model_inputs_first_pass(
        num_tokens=num_tokens,
        num_input_tokens=num_input_tokens,
        mm_embed_inputs=([], is_mm_embed),
    )

    assert captured["mask"].shape == torch.Size([num_tokens])
    assert captured["mask"][3].item() is True


def test_no_mm_inputs_path_unchanged():
    """When supports_mm_inputs is False, embed_input_ids must not be called
    and input_ids must be returned via model_kwargs."""
    p = _make_proposer(supports_mm_inputs=False)

    p.model = mock.MagicMock()

    model_kwargs, slot_mapping_size = p.build_model_inputs_first_pass(
        num_tokens=10,
        num_input_tokens=10,
        mm_embed_inputs=None,
    )

    p.model.embed_input_ids.assert_not_called()
    assert model_kwargs["input_ids"] is not None
    assert model_kwargs["inputs_embeds"] is None
    assert slot_mapping_size == 10


def test_mm_supports_with_none_inputs_path_unchanged():
    """When supports_mm_inputs=True but mm_embed_inputs is None, both
    mm_embeds and is_mm_embed default to None and no padding logic runs."""
    p = _make_proposer(supports_mm_inputs=True)

    def fake_embed(input_ids, multimodal_embeddings=None, is_multimodal=None):
        assert multimodal_embeddings is None
        assert is_multimodal is None
        return torch.zeros(input_ids.shape[0], 8, dtype=torch.float32)

    p.model = mock.MagicMock()
    p.model.embed_input_ids.side_effect = fake_embed

    p.build_model_inputs_first_pass(
        num_tokens=10,
        num_input_tokens=10,
        mm_embed_inputs=None,
    )
