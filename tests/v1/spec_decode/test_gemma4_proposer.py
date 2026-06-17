# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Gemma4Proposer._create_draft_vllm_config.

Tests run without GPU. We mock VllmConfig and SpeculativeConfig
so no model weights are downloaded.
"""

from types import SimpleNamespace
from unittest import mock

from vllm.config import AttentionConfig, VllmConfig
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer


def _make_fake_proposer(
    *,
    target_quant_config=None,
    draft_quant_config=None,
    target_hidden_size_per_layer_input: int = 0,
    draft_hidden_size_per_layer_input: int = 256,
    attention_backend=None,
) -> Gemma4Proposer:
    """Build a Gemma4Proposer with mocked configs, no GPU required."""

    # --- Target model config (e.g. 26B) ---
    target_hf_text_config = SimpleNamespace(
        hidden_size_per_layer_input=target_hidden_size_per_layer_input,
    )
    target_hf_config = mock.MagicMock()
    target_hf_config.get_text_config.return_value = target_hf_text_config

    target_model_config = mock.MagicMock()
    target_model_config.hf_config = target_hf_config
    target_model_config.quantization = "compressed_tensors"

    # --- Draft model config (e.g. E4B) ---
    draft_hf_text_config = SimpleNamespace(
        hidden_size_per_layer_input=draft_hidden_size_per_layer_input,
    )
    draft_hf_config = mock.MagicMock()
    draft_hf_config.get_text_config.return_value = draft_hf_text_config

    draft_model_config = mock.MagicMock()
    draft_model_config.hf_config = draft_hf_config
    draft_model_config.quantization = "compressed_tensors"

    # --- SpeculativeConfig ---
    speculative_config = mock.MagicMock()
    speculative_config.draft_model_config = draft_model_config
    speculative_config.method = "mtp"
    speculative_config.num_speculative_tokens = 1
    speculative_config.parallel_drafting = False
    speculative_config.attention_backend = None
    speculative_config.moe_backend = None

    # --- VllmConfig ---
    load_config = mock.MagicMock()
    vllm_config = mock.MagicMock(spec=VllmConfig)
    vllm_config.model_config = target_model_config
    vllm_config.quant_config = target_quant_config
    vllm_config.load_config = load_config
    vllm_config.speculative_config = speculative_config
    vllm_config.attention_config = AttentionConfig(backend=attention_backend)
    vllm_config.parallel_config = mock.MagicMock()
    vllm_config.parallel_config.data_parallel_rank = 0
    vllm_config.model_config.dtype = "bfloat16"
    vllm_config.model_config.max_model_len = 8192
    vllm_config.model_config.get_hidden_size.return_value = 256
    vllm_config.model_config.get_inputs_embeds_size.return_value = 256

    # Build proposer bypassing __init__ to avoid GPU/model loading
    proposer = object.__new__(Gemma4Proposer)
    proposer.vllm_config = vllm_config
    proposer.speculative_config = speculative_config
    proposer.draft_model_config = draft_model_config
    proposer.method = "mtp"
    proposer.pass_hidden_states_to_model = True
    proposer.constant_draft_positions = True
    proposer.parallel_drafting = False
    proposer.num_speculative_tokens = 1
    proposer.hidden_size = 256
    proposer.inputs_embeds_size = 256
    proposer.device = "cpu"
    proposer.dtype = "bfloat16"
    proposer.max_model_len = 8192
    proposer.dp_rank = 0
    proposer._per_group_block_tables = {}
    proposer._centroids_sizes = []
    proposer._centroids_graphs = {}
    proposer._centroids_inputs = {}
    proposer._centroids_outputs = {}
    return proposer


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_draft_config_uses_draft_model_config():
    """draft_vllm_config.model_config must be draft_model_config, not target's."""
    target_quant = mock.MagicMock(name="target_8bit_quant")
    draft_quant = mock.MagicMock(name="draft_int4_quant")
    proposer = _make_fake_proposer(
        target_quant_config=target_quant,
        draft_quant_config=draft_quant,
    )

    with mock.patch.object(
        VllmConfig,
        "get_quantization_config",
        return_value=draft_quant,
    ):
        draft_cfg = proposer._create_draft_vllm_config()

    assert draft_cfg.model_config is proposer.speculative_config.draft_model_config, (
        "draft_vllm_config.model_config should be draft_model_config, "
        "not target model_config"
    )


def test_draft_config_uses_draft_quant_config():
    """draft_vllm_config.quant_config must be derived from draft_model_config."""
    target_quant = mock.MagicMock(name="target_8bit_quant")
    draft_quant = mock.MagicMock(name="draft_int4_quant")
    proposer = _make_fake_proposer(
        target_quant_config=target_quant,
        draft_quant_config=draft_quant,
    )

    with mock.patch.object(
        VllmConfig,
        "get_quantization_config",
        return_value=draft_quant,
    ) as mock_get_quant:
        draft_cfg = proposer._create_draft_vllm_config()

        # Verify get_quantization_config was called with draft model config
        mock_get_quant.assert_called_once_with(
            proposer.speculative_config.draft_model_config,
            proposer.vllm_config.load_config,
        )

    assert draft_cfg.quant_config is draft_quant, (
        "draft_vllm_config.quant_config should be derived from draft_model_config "
        "(INT4), not copied from target quant_config (8-bit)"
    )
    assert draft_cfg.quant_config is not target_quant, (
        "draft_vllm_config.quant_config must not be target's quant_config"
    )


def test_draft_config_preserves_target_attention_backend():
    """attention_config.backend from target must be carried through."""
    proposer = _make_fake_proposer(attention_backend="TRITON_ATTN")

    with mock.patch.object(VllmConfig, "get_quantization_config", return_value=None):
        draft_cfg = proposer._create_draft_vllm_config()

    assert draft_cfg.attention_config.backend == "TRITON_ATTN", (
        "Target's forced TRITON_ATTN backend must be preserved in draft config"
    )
