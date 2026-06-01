# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ModelOpt NVFP4_AWQ quantization support.

NVFP4_AWQ checkpoints (e.g. AEON-7/Gemma-4-E4B-it-Uncensored-NVFP4) store
standard NVFP4 weights plus a per-layer `pre_quant_scale` 1-D bf16 tensor
that smooths activations. These tests verify:
1. NVFP4_AWQ passes config validation (does not raise on QUANT_ALGOS check).
2. NVFP4_AWQ dispatches to ModelOptNvFp4LinearMethod (no Marlin fallback).
3. Linear create_weights registers pre_quant_scale parameter when enabled.
4. Linear create_weights does not register pre_quant_scale for plain NVFP4.
5. MoE method raises NotImplementedError to avoid silent accuracy loss.
6. Linear apply() multiplies activations by pre_quant_scale before GEMM.
7. Linear apply() does not change input when pre_quant_scale is absent.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.modelopt import (
    QUANT_ALGOS,
    ModelOptNvFp4Config,
    ModelOptNvFp4LinearMethod,
)


def _build_nvfp4_awq_config() -> ModelOptNvFp4Config:
    return ModelOptNvFp4Config._from_config(
        quant_method="NVFP4_AWQ",
        kv_cache_quant_method=None,
        exclude_modules=[],
        group_size=16,
        original_config={
            "quantization": {
                "quant_algo": "NVFP4_AWQ",
                "kv_cache_quant_algo": None,
                "group_size": 16,
                "exclude_modules": [],
                "pre_quant_scale": True,
                "has_zero_point": False,
            }
        },
    )


def test_nvfp4_awq_in_quant_algos_whitelist():
    """Validation step must accept NVFP4_AWQ without raising."""
    assert "NVFP4_AWQ" in QUANT_ALGOS


def test_nvfp4_awq_config_dispatches_to_nvfp4_linear_method():
    """NVFP4_AWQ must reuse the W4A4 Linear path (not Marlin, not raise)."""
    cfg = _build_nvfp4_awq_config()
    assert cfg.LinearMethodCls is ModelOptNvFp4LinearMethod
    assert cfg.has_pre_quant_scale is True


class _StubLinearLayer(torch.nn.Module):
    """Minimal stand-in for a vLLM Linear layer to exercise create_weights."""

    def __init__(self):
        super().__init__()


def _create_weights_helper(has_pre_quant_scale: bool) -> _StubLinearLayer:
    cfg = ModelOptNvFp4Config(
        quant_method="NVFP4_AWQ" if has_pre_quant_scale else "NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        group_size=16,
        has_pre_quant_scale=has_pre_quant_scale,
    )
    method = ModelOptNvFp4LinearMethod(cfg)
    layer = _StubLinearLayer()
    method.create_weights(
        layer=layer,
        input_size_per_partition=128,
        output_partition_sizes=[64],
        input_size=128,
        output_size=64,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **kw: None,
    )
    return layer


def test_pre_quant_scale_registered_when_enabled():
    """NVFP4_AWQ must register a 1-D bf16 pre_quant_scale param of input width."""
    layer = _create_weights_helper(has_pre_quant_scale=True)
    assert hasattr(layer, "pre_quant_scale"), (
        "create_weights must register `pre_quant_scale` for NVFP4_AWQ."
    )
    p = layer.pre_quant_scale
    assert p.dtype == torch.bfloat16, f"expected bf16, got {p.dtype}"
    assert tuple(p.shape) == (128,), f"expected (128,), got {tuple(p.shape)}"


def test_pre_quant_scale_absent_for_plain_nvfp4():
    """Plain NVFP4 (no AWQ) must not register pre_quant_scale (avoid wasted memory)."""
    layer = _create_weights_helper(has_pre_quant_scale=False)
    assert not hasattr(layer, "pre_quant_scale"), (
        "Plain NVFP4 must not register `pre_quant_scale`."
    )


def test_moe_method_rejects_nvfp4_awq():
    """MoE NVFP4_AWQ is not implemented yet; instantiation must raise clearly."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE,
    )

    cfg = ModelOptNvFp4Config(
        quant_method="NVFP4_AWQ",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        group_size=16,
        has_pre_quant_scale=True,
    )

    class _StubMoEConfig:
        pass

    with pytest.raises(NotImplementedError, match="NVFP4_AWQ.*MoE"):
        ModelOptNvFp4FusedMoE(quant_config=cfg, moe_config=_StubMoEConfig())


def test_apply_multiplies_input_by_pre_quant_scale():
    """apply() must scale x by pre_quant_scale before delegating to the kernel."""
    cfg = ModelOptNvFp4Config(
        quant_method="NVFP4_AWQ",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        group_size=16,
        has_pre_quant_scale=True,
    )
    method = ModelOptNvFp4LinearMethod(cfg)

    class _StubLayer(torch.nn.Module):
        pre_quant_scale: torch.Tensor

    layer = _StubLayer()
    layer.pre_quant_scale = torch.tensor([2.0, 0.5, 4.0, 0.25], dtype=torch.bfloat16)

    captured = {}

    def fake_apply_weights(layer, x, bias=None):
        captured["x"] = x.clone()
        return torch.zeros_like(x[..., :2])

    method.kernel = type("K", (), {"apply_weights": staticmethod(fake_apply_weights)})()

    x_in = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.bfloat16)
    method.apply(layer=layer, x=x_in, bias=None)

    expected = (x_in * layer.pre_quant_scale).to(torch.bfloat16)
    assert torch.allclose(captured["x"].float(), expected.float(), atol=1e-3), (
        f"apply() did not multiply x by pre_quant_scale: got {captured['x']}, "
        f"expected {expected}"
    )


def test_apply_unchanged_when_pre_quant_scale_absent():
    """Plain NVFP4 path: apply() must not touch x (no layer.pre_quant_scale)."""
    cfg = ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        group_size=16,
        has_pre_quant_scale=False,
    )
    method = ModelOptNvFp4LinearMethod(cfg)

    class _StubLayer(torch.nn.Module):
        pass

    layer = _StubLayer()
    # Note: no pre_quant_scale attribute set.

    captured = {}

    def fake_apply_weights(layer, x, bias=None):
        captured["x"] = x.clone()
        return torch.zeros_like(x[..., :2])

    method.kernel = type("K", (), {"apply_weights": staticmethod(fake_apply_weights)})()

    x_in = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    method.apply(layer=layer, x=x_in, bias=None)
    assert torch.equal(captured["x"], x_in), (
        "apply() must pass x through unchanged when pre_quant_scale is absent."
    )
