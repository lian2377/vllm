# SPDX-License-Identifier: Apache-2.0
"""Tests for SpecDecodeBaseProposer._build_layer_spec_mapping and
initialize_attn_backend.  All tests run without GPU."""

from unittest import mock

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_uniform_spec(layer_specs: dict):
    """Build a mock UniformTypeKVCacheSpecs with the given per-layer specs."""
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    us = mock.MagicMock(spec=UniformTypeKVCacheSpecs)
    us.kv_cache_specs = layer_specs
    return us


def _make_kv_cache_group(layer_names: list[str], kv_spec):
    g = mock.MagicMock()
    g.layer_names = layer_names
    g.kv_cache_spec = kv_spec
    return g


def _make_kv_cache_config(groups):
    cfg = mock.MagicMock()
    cfg.kv_cache_groups = groups
    return cfg


def _make_proposer(draft_attn_layer_names: set[str]) -> SpecDecodeBaseProposer:
    """Build a SpecDecodeBaseProposer with minimal state (no GPU)."""
    p = object.__new__(SpecDecodeBaseProposer)
    p._draft_attn_layer_names = draft_attn_layer_names
    p.vllm_config = mock.MagicMock()
    p.device = "cpu"
    p.draft_attn_groups = []
    p.kv_cache_gid = -1
    p.block_size = -1
    return p


# ---------------------------------------------------------------------------
# _build_layer_spec_mapping — base class
# ---------------------------------------------------------------------------


def test_build_layer_spec_mapping_single_group():
    """Single homogeneous group → all layers get same (gid=0, spec)."""
    spec_a = mock.MagicMock(name="spec_a")
    group = _make_kv_cache_group(["layer.0.attn", "layer.1.attn"], spec_a)
    cfg = _make_kv_cache_config([group])
    all_attn = {"layer.0.attn": mock.MagicMock(), "layer.1.attn": mock.MagicMock()}

    p = _make_proposer({"layer.0.attn", "layer.1.attn"})
    gid_map, spec_map = p._build_layer_spec_mapping(cfg, all_attn)

    assert gid_map == {"layer.0.attn": 0, "layer.1.attn": 0}
    assert spec_map["layer.0.attn"] is spec_a
    assert spec_map["layer.1.attn"] is spec_a


def test_build_layer_spec_mapping_multi_group():
    """Two groups with different specs → each layer maps to its own (gid, spec)."""
    spec_slide = mock.MagicMock(name="spec_slide")
    spec_full = mock.MagicMock(name="spec_full")
    group0 = _make_kv_cache_group(["layer.0.attn", "layer.1.attn"], spec_slide)
    group1 = _make_kv_cache_group(["layer.2.attn"], spec_full)
    cfg = _make_kv_cache_config([group0, group1])
    all_attn = {
        "layer.0.attn": mock.MagicMock(),
        "layer.1.attn": mock.MagicMock(),
        "layer.2.attn": mock.MagicMock(),
    }

    p = _make_proposer({"layer.0.attn", "layer.1.attn", "layer.2.attn"})
    gid_map, spec_map = p._build_layer_spec_mapping(cfg, all_attn)

    assert gid_map["layer.0.attn"] == 0
    assert gid_map["layer.1.attn"] == 0
    assert gid_map["layer.2.attn"] == 1
    assert spec_map["layer.0.attn"] is spec_slide
    assert spec_map["layer.1.attn"] is spec_slide
    assert spec_map["layer.2.attn"] is spec_full


def test_build_layer_spec_mapping_uniform_specs():
    """UniformTypeKVCacheSpecs → each layer gets its individual spec."""
    per_layer = {
        "layer.0.attn": mock.MagicMock(name="spec_0"),
        "layer.1.attn": mock.MagicMock(name="spec_1"),
    }
    us = _make_uniform_spec(per_layer)
    group = _make_kv_cache_group(["layer.0.attn", "layer.1.attn"], us)
    cfg = _make_kv_cache_config([group])
    all_attn = {"layer.0.attn": mock.MagicMock(), "layer.1.attn": mock.MagicMock()}

    p = _make_proposer({"layer.0.attn", "layer.1.attn"})
    gid_map, spec_map = p._build_layer_spec_mapping(cfg, all_attn)

    assert spec_map["layer.0.attn"] is per_layer["layer.0.attn"]
    assert spec_map["layer.1.attn"] is per_layer["layer.1.attn"]


def test_build_layer_spec_mapping_uniform_missing_key_fallback():
    """Layer not in UniformTypeKVCacheSpecs.kv_cache_specs → falls back."""
    per_layer = {"layer.0.attn": mock.MagicMock(name="spec_0")}
    us = _make_uniform_spec(per_layer)
    group = _make_kv_cache_group(["layer.0.attn", "layer.1.attn"], us)
    cfg = _make_kv_cache_config([group])
    all_attn = {"layer.0.attn": mock.MagicMock(), "layer.1.attn": mock.MagicMock()}

    p = _make_proposer({"layer.0.attn", "layer.1.attn"})
    gid_map, spec_map = p._build_layer_spec_mapping(cfg, all_attn)

    assert spec_map["layer.1.attn"] is us


# ---------------------------------------------------------------------------
# initialize_attn_backend — single group backward-compat
# ---------------------------------------------------------------------------


def test_initialize_attn_backend_single_group_creates_one_group():
    """Single group → one AttentionGroup created."""
    spec = mock.MagicMock(name="spec")
    spec.block_size = 16
    group = _make_kv_cache_group(["draft_model.model.layers.0.self_attn.attn"], spec)
    cfg = _make_kv_cache_config([group])

    backend = mock.MagicMock()
    backend.full_cls_name.return_value = "FakeBackend"
    attn_layer = mock.MagicMock()
    attn_layer.get_attn_backend.return_value = backend

    all_attn = {"draft_model.model.layers.0.self_attn.attn": attn_layer}

    p = _make_proposer({"draft_model.model.layers.0.self_attn.attn"})

    mock_attn_group = mock.MagicMock()
    mock_attn_group.get_metadata_builder.return_value.kv_cache_spec.block_size = 16
    mock_attn_group.kv_cache_group_id = 0

    with (
        mock.patch(
            "vllm.v1.spec_decode.llm_base_proposer.get_layers_from_vllm_config",
            return_value=all_attn,
        ),
        mock.patch(
            "vllm.v1.spec_decode.llm_base_proposer.AttentionGroup",
            return_value=mock_attn_group,
        ),
    ):
        p.initialize_attn_backend(cfg)

    assert len(p.draft_attn_groups) == 1
    assert p.kv_cache_gid == 0
    assert p.block_size == 16


def test_initialize_attn_backend_multi_group_creates_two_groups():
    """Two groups (sliding + full) → two AttentionGroups, each with correct gid."""
    spec_slide = mock.MagicMock(name="spec_slide")
    spec_slide.block_size = 16
    spec_full = mock.MagicMock(name="spec_full")
    spec_full.block_size = 32

    group0 = _make_kv_cache_group(
        ["draft_model.model.layers.0.self_attn.attn"], spec_slide
    )
    group1 = _make_kv_cache_group(
        ["draft_model.model.layers.1.self_attn.attn"], spec_full
    )
    cfg = _make_kv_cache_config([group0, group1])

    backend = mock.MagicMock()
    backend.full_cls_name.return_value = "FakeBackend"
    attn_layer = mock.MagicMock()
    attn_layer.get_attn_backend.return_value = backend

    all_attn = {
        "draft_model.model.layers.0.self_attn.attn": attn_layer,
        "draft_model.model.layers.1.self_attn.attn": attn_layer,
    }

    p = _make_proposer(
        {
            "draft_model.model.layers.0.self_attn.attn",
            "draft_model.model.layers.1.self_attn.attn",
        }
    )

    created_groups = []

    def make_group(**kwargs):
        g = mock.MagicMock()
        g.kv_cache_group_id = kwargs["kv_cache_group_id"]
        g.get_metadata_builder.return_value.kv_cache_spec.block_size = kwargs[
            "kv_cache_spec"
        ].block_size
        created_groups.append(g)
        return g

    with (
        mock.patch(
            "vllm.v1.spec_decode.llm_base_proposer.get_layers_from_vllm_config",
            return_value=all_attn,
        ),
        mock.patch(
            "vllm.v1.spec_decode.llm_base_proposer.AttentionGroup",
            side_effect=make_group,
        ),
    ):
        p.initialize_attn_backend(cfg)

    assert len(p.draft_attn_groups) == 2
    gids = {g.kv_cache_group_id for g in p.draft_attn_groups}
    assert gids == {0, 1}


# ---------------------------------------------------------------------------
# validate_same_kv_cache_group — must be no-op (no assert)
# ---------------------------------------------------------------------------


def test_validate_same_kv_cache_group_no_longer_raises():
    """Base class validate_same_kv_cache_group must not raise for multi-group."""
    spec_a = mock.MagicMock()
    spec_b = mock.MagicMock()
    group0 = _make_kv_cache_group(["layer.0.attn"], spec_a)
    group1 = _make_kv_cache_group(["layer.1.attn"], spec_b)
    cfg = _make_kv_cache_config([group0, group1])

    p = _make_proposer({"layer.0.attn", "layer.1.attn"})
    p.validate_same_kv_cache_group(cfg)


# ---------------------------------------------------------------------------
# Gemma4Proposer._build_layer_spec_mapping — KV-sharing fallback
# ---------------------------------------------------------------------------


def test_gemma4_build_layer_spec_mapping_kv_sharing_fallback():
    """Layer without own spec in UniformTypeKVCacheSpecs inherits target's spec."""
    from vllm.v1.spec_decode.gemma4 import Gemma4Proposer

    tgt_spec = mock.MagicMock(name="tgt_spec")
    per_layer = {"layer.0.self_attn.attn": tgt_spec}
    us = _make_uniform_spec(per_layer)
    # layer.1 is a KV-sharing layer that doesn't have its own spec
    group = _make_kv_cache_group(
        ["layer.0.self_attn.attn", "layer.1.self_attn.attn"], us
    )
    cfg = _make_kv_cache_config([group])

    attn_layer_0 = mock.MagicMock(spec=[])  # no kv_sharing_target_layer_name
    attn_layer_1 = mock.MagicMock(spec=["kv_sharing_target_layer_name"])
    attn_layer_1.kv_sharing_target_layer_name = "layer.0.self_attn.attn"
    all_attn = {
        "layer.0.self_attn.attn": attn_layer_0,
        "layer.1.self_attn.attn": attn_layer_1,
    }

    p = object.__new__(Gemma4Proposer)
    p._draft_attn_layer_names = {"layer.0.self_attn.attn", "layer.1.self_attn.attn"}
    p.vllm_config = mock.MagicMock()
    p.device = "cpu"
    p.draft_attn_groups = []
    p.kv_cache_gid = -1
    p.block_size = -1

    gid_map, spec_map = p._build_layer_spec_mapping(cfg, all_attn)

    assert spec_map["layer.0.self_attn.attn"] is tgt_spec
    assert spec_map["layer.1.self_attn.attn"] is tgt_spec


def test_gemma4_build_layer_spec_mapping_no_kv_sharing():
    """Layer without kv_sharing_target_layer_name falls back to group spec."""
    from vllm.v1.spec_decode.gemma4 import Gemma4Proposer

    per_layer = {"layer.0.self_attn.attn": mock.MagicMock(name="spec_0")}
    us = _make_uniform_spec(per_layer)
    group = _make_kv_cache_group(
        ["layer.0.self_attn.attn", "layer.1.self_attn.attn"], us
    )
    cfg = _make_kv_cache_config([group])

    attn_layer = mock.MagicMock(spec=[])  # no attributes → getattr returns default
    all_attn = {
        "layer.0.self_attn.attn": attn_layer,
        "layer.1.self_attn.attn": attn_layer,
    }

    p = object.__new__(Gemma4Proposer)
    p._draft_attn_layer_names = {"layer.0.self_attn.attn", "layer.1.self_attn.attn"}
    p.vllm_config = mock.MagicMock()
    p.device = "cpu"
    p.draft_attn_groups = []
    p.kv_cache_gid = -1
    p.block_size = -1

    gid_map, spec_map = p._build_layer_spec_mapping(cfg, all_attn)

    # layer.1 has no kv_sharing_target_layer_name → falls back to group spec
    assert spec_map["layer.1.self_attn.attn"] is us
