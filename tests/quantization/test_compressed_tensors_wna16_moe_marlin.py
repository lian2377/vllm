# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import WNA16MoEBackend
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_marlin import (  # noqa: E501
    CompressedTensorsWNA16MarlinMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8StaticGroupScale,
    scalar_types,
)


@pytest.fixture()
def should_do_global_cleanup_after_test():
    return False


def _int8_group_weight_quant(group_size: int = 128) -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
        symmetric=True,
        dynamic=False,
    )


def _fake_moe_layer() -> SimpleNamespace:
    moe_config = SimpleNamespace(
        intermediate_size_per_partition_unpadded=4096,
    )
    return SimpleNamespace(
        hidden_size=8192,
        moe_config=moe_config,
        apply_router_weight_on_input=False,
    )


class _FakeCompressedTensorsConfig:
    def __init__(self, weight_quant: QuantizationArgs):
        self.weight_quant = weight_quant

    def _add_fused_moe_to_target_scheme_map(self) -> None:
        pass

    def get_scheme_dict(self, layer, name):
        return {
            "weights": self.weight_quant,
            "input_activations": None,
            "format": CompressionFormat.pack_quantized.value,
        }

    def _is_mxfp4(self, weight_quant) -> bool:
        return False

    def _is_mxfp8(self, weight_quant) -> bool:
        return False

    def _is_wNa16_group_channel(self, weight_quant, input_quant) -> bool:
        return True


@pytest.fixture
def marlin_backend(monkeypatch):
    captured = {}

    def fake_select_wna16_moe_backend(config, weight_key):
        captured["weight_key"] = weight_key
        return WNA16MoEBackend.MARLIN, MarlinExperts

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_wna16_marlin."
        "select_wna16_moe_backend",
        fake_select_wna16_moe_backend,
    )
    return captured


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_ct_wna16_int8_groupwise_dispatches_to_marlin(
    marlin_backend,
    group_size,
):
    method = CompressedTensorsMoEMethod.get_moe_method(
        _FakeCompressedTensorsConfig(_int8_group_weight_quant(group_size=group_size)),
        _fake_moe_layer(),
        "model.layers.0.moe.experts",
    )

    assert isinstance(method, CompressedTensorsWNA16MarlinMoEMethod)
    assert method.group_size == group_size


def test_ct_wna16_int8_groupwise_marlin_uses_int8_group_quant_key(
    marlin_backend,
):
    method = CompressedTensorsWNA16MarlinMoEMethod(
        _int8_group_weight_quant(group_size=128),
        input_quant=None,
        moe=SimpleNamespace(),
    )

    assert method.group_size == 128
    assert marlin_backend["weight_key"] == QuantKey(
        scalar_types.uint8b128,
        kInt8StaticGroupScale,
        symmetric=True,
    )


def test_ct_wna16_int8_asymmetric_marlin_rejects_with_clear_error(
    marlin_backend,
):
    weight_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=128,
        symmetric=False,
        dynamic=False,
    )

    with pytest.raises(
        ValueError,
        match="does not support asymmetric int8 MoE weights",
    ):
        CompressedTensorsWNA16MarlinMoEMethod(
            weight_quant,
            input_quant=None,
            moe=SimpleNamespace(),
        )


def test_ct_wna16_int8_unsupported_group_size_rejects_with_clear_error(
    marlin_backend,
):
    with pytest.raises(
        ValueError,
        match="only supports int8 group_size",
    ):
        CompressedTensorsWNA16MarlinMoEMethod(
            _int8_group_weight_quant(group_size=16),
            input_quant=None,
            moe=SimpleNamespace(),
        )
