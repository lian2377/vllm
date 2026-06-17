# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for excluded-layer quant method dispatch."""

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase


def make_excluded_quant_method(
    layer: torch.nn.Module,
) -> QuantizeMethodBase | None:
    """Return the unquantized method for an excluded layer.

    `ParallelLMHead` intentionally returns `None` so
    `VocabParallelEmbedding.__init__` installs `UnquantizedEmbeddingMethod`.
    Regular linear layers receive `UnquantizedLinearMethod`.
    """
    from vllm.model_executor.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    if isinstance(layer, ParallelLMHead):
        return None
    if isinstance(layer, LinearBase):
        return UnquantizedLinearMethod()
    return None
