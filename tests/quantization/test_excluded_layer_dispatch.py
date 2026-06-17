# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for excluded quant-method dispatch contracts."""

import pytest
import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase


class _DummyQuantMethod(QuantizeMethodBase):
    def create_weights(self, layer, *weight_args, **extra_weight_attrs):
        raise AssertionError("should not be called")

    def apply(self, layer, *args, **kwargs):
        raise AssertionError("should not be called")


def test_base_embedding_remains_explicitly_unsupported():
    method = _DummyQuantMethod()
    layer = torch.nn.Module()
    layer.weight = torch.eye(4, dtype=torch.float32)

    with pytest.raises(NotImplementedError):
        method.embedding(layer, torch.tensor([0], dtype=torch.long))


def test_make_excluded_quant_method_lm_head_returns_none():
    from vllm.model_executor.layers.quantization.utils.excluded import (
        make_excluded_quant_method,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    layer = ParallelLMHead.__new__(ParallelLMHead)

    assert make_excluded_quant_method(layer) is None


def test_make_excluded_quant_method_linear_returns_unquantized_linear():
    from vllm.model_executor.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.excluded import (
        make_excluded_quant_method,
    )

    layer = LinearBase.__new__(LinearBase)

    assert isinstance(make_excluded_quant_method(layer), UnquantizedLinearMethod)


def test_make_excluded_quant_method_other_returns_none():
    from vllm.model_executor.layers.quantization.utils.excluded import (
        make_excluded_quant_method,
    )

    assert make_excluded_quant_method(torch.nn.Identity()) is None
