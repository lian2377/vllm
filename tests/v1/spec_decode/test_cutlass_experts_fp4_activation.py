# SPDX-License-Identifier: Apache-2.0
"""Tests for CutlassExpertsFp4._supports_activation.

Gemma 4 (and similar Transformer models) use GELU_TANH as the MLP
activation function.  CutlassExpertsFp4 was missing GELU_TANH /
GELU_TANH_NO_MUL from its supported activation list, causing NVFP4
MoE deployments of these models to fall back to MARLIN.

These tests run without GPU; we only exercise the staticmethod whitelist.
"""
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    CutlassExpertsFp4,
)

# ---------------------------------------------------------------------------
# Activations that MUST be supported (the change of this PR)
# ---------------------------------------------------------------------------


def test_gelu_tanh_is_supported():
    """GELU_TANH must be accepted so Gemma 4 NVFP4 MoE uses VLLM_CUTLASS."""
    assert CutlassExpertsFp4._supports_activation(MoEActivation.GELU_TANH)


def test_gelu_tanh_no_mul_is_supported():
    """Non-gated GELU_TANH variant must also be accepted for completeness
    (mirrors how GELU_NO_MUL is already supported alongside GELU)."""
    assert CutlassExpertsFp4._supports_activation(MoEActivation.GELU_TANH_NO_MUL)


# ---------------------------------------------------------------------------
# Activations that MUST stay supported (regression guard)
# ---------------------------------------------------------------------------


def test_previously_supported_activations_still_supported():
    """No regression for activations that already worked before this change."""
    for activation in [
        MoEActivation.SILU,
        MoEActivation.GELU,
        MoEActivation.SWIGLUOAI,
        MoEActivation.SWIGLUSTEP,
        MoEActivation.SILU_NO_MUL,
        MoEActivation.GELU_NO_MUL,
        MoEActivation.RELU2_NO_MUL,
    ]:
        assert CutlassExpertsFp4._supports_activation(activation), (
            f"Previously-supported activation {activation} must remain supported."
        )
