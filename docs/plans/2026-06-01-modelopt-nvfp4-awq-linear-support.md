# ModelOpt NVFP4_AWQ Linear Support Implementation Plan

> **For implementers:** Execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks are sequential — each one depends on the previous task's commit. All work happens on branch `jetson`. Per repo rules, only `.venv/bin/python` is used.

**Goal:** Make vLLM load and run ModelOpt-quantized checkpoints whose `hf_quant_config.json` has `quant_algo: NVFP4_AWQ` (NVFP4 weights + per-input-channel `pre_quant_scale` activation smoothing), for **dense Linear layers only** — Gemma 4 E4B (`AEON-7/Gemma-4-E4B-it-Uncensored-NVFP4`) being the primary target.

**Architecture:** NVFP4_AWQ checkpoints store standard NVFP4 weights (`has_zero_point: false`, identical on-disk layout to plain NVFP4) plus a per-layer `pre_quant_scale` 1-D bf16 tensor along the input dimension. We thread that tensor through the existing `ModelOptNvFp4LinearMethod`: register it as a row-sharded parameter at `create_weights`, keep it on the layer after loading, multiply it into the activation input inside `apply()` — leaving the CUTLASS NVFP4 GEMM kernel untouched. The `NVFP4_AWQ` algo is accepted in the ModelOpt config whitelist and dispatched to the same Linear method as `NVFP4`. MoE (`ModelOptNvFp4FusedMoE`) is **explicitly out of scope for v1**: we raise a clear `NotImplementedError` when NVFP4_AWQ is detected on a MoE model so users see the failure mode immediately.

**Tech Stack:**

- vLLM ModelOpt quantization module (`vllm/model_executor/layers/quantization/modelopt.py`)
- vLLM parameter system (`RowvLLMParameter`, `BasevLLMParameter`)
- pytest, repo's `.venv/bin/python` workflow

**Scope opt-out:** none. Code change → unit tests required. End-to-end Thor verification is a manual step at the end (no GPU available locally).

**Sign-convention assumption to verify before Task 4:** Plan assumes `y = w @ (x * pre_quant_scale)` (the "scale is the reciprocal of the smoothing factor" convention ModelOpt typically emits). If the captured `pre_quant_scale` values are predominantly < 1, this is correct. If predominantly > 1, swap `*` → `/` in Task 4 step 3 before committing. Verification command is in Task 0.

---

## File Map

| File | Responsibility | Action |
|---|---|---|
| `vllm/model_executor/layers/quantization/modelopt.py` | ModelOpt config validation, NVFP4 Linear & MoE methods | Modify (lines 103-118, 1005-1100, 1103-1232, 1381-1400) |
| `tests/quantization/test_modelopt_nvfp4_awq.py` | Unit tests: config acceptance, param registration, apply() math, MoE guard | Create |

No new files in the quantization module — all changes live alongside the existing NVFP4 implementation to keep code together.

---

## Task 0: Verify sign convention from checkpoint

**Files:** none (verification only)

- [ ] **Step 1: On Thor, capture min/max/mean of a sample `pre_quant_scale`**

Run on Thor (inside the vllm container):

```bash
sudo docker exec -it llm-service-pack-vllm-1 python3 -c "
from safetensors import safe_open
import glob
f = sorted(glob.glob('/models/AEON-7/Gemma-4-E4B-it-Uncensored-NVFP4/*.safetensors'))[0]
with safe_open(f, 'pt') as h:
    t = h.get_tensor('model.language_model.layers.0.mlp.down_proj.pre_quant_scale')
    print('shape:', tuple(t.shape), 'dtype:', t.dtype)
    print('min/max/mean:', t.min().item(), t.max().item(), t.float().mean().item())
    print('count<1:', (t.float() < 1.0).sum().item(), '/', t.numel())
"
```

Expected: prints shape `(10240,)`, dtype `torch.bfloat16`, and value statistics.

- [ ] **Step 2: Decide sign convention and record decision in this plan**

If `count<1` ≥ 50% of `numel`: keep `x * pre_quant_scale` (Task 4 default).
If `count<1` < 50% of `numel`: edit Task 4 Step 3 to use `x / pre_quant_scale` and add a one-line note "Verified divide convention from Task 0" before the implementation.

No commit for Task 0; it informs Task 4 only.

---

## Task 1: Whitelist NVFP4_AWQ in `QUANT_ALGOS` and dispatch to NVFP4 Linear

**Files:**

- Modify: `vllm/model_executor/layers/quantization/modelopt.py:103-118`, `:1035-1043`
- Test: `tests/quantization/test_modelopt_nvfp4_awq.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/quantization/test_modelopt_nvfp4_awq.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Tests for ModelOpt NVFP4_AWQ quantization support.

NVFP4_AWQ checkpoints (e.g. AEON-7/Gemma-4-E4B-it-Uncensored-NVFP4) store
standard NVFP4 weights plus a per-layer `pre_quant_scale` 1-D bf16 tensor
that smooths activations. These tests verify:
1. NVFP4_AWQ passes config validation (does not raise on QUANT_ALGOS check).
2. NVFP4_AWQ dispatches to ModelOptNvFp4LinearMethod (no Marlin fallback).
3. Linear apply() multiplies activations by pre_quant_scale before GEMM.
4. MoE method raises NotImplementedError to avoid silent accuracy loss.
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
```

- [ ] **Step 2: Run the new tests and confirm they fail**

```bash
.venv/bin/python -m pytest tests/quantization/test_modelopt_nvfp4_awq.py::test_nvfp4_awq_in_quant_algos_whitelist tests/quantization/test_modelopt_nvfp4_awq.py::test_nvfp4_awq_config_dispatches_to_nvfp4_linear_method -v
```

Expected: both FAIL. The first fails with `AssertionError` ("NVFP4_AWQ" not in QUANT_ALGOS). The second fails inside `_from_config` with `ValueError: Unsupported ModelOpt NVFP4 quant_algo: NVFP4_AWQ`.

- [ ] **Step 3: Add `NVFP4_AWQ` to the whitelist**

In `vllm/model_executor/layers/quantization/modelopt.py`, replace lines 103-118:

```python
QUANT_ALGOS = [
    # FP8 (per-tensor weight + optional static activation scale).
    "FP8",
    # FP8 per-channel weight scale + per-token activation scale.
    "FP8_PER_CHANNEL_PER_TOKEN",
    # FP8 per-block weight-only (ModelOpt may emit this as lowercase).
    "FP8_PB_WO",
    # NVFP4 W4A4 (4-bit float weights AND 4-bit float activations).
    "NVFP4",
    # NVFP4 W4A4 with AWQ-calibrated per-input-channel activation smoothing.
    # On-disk weight layout is identical to plain NVFP4 (has_zero_point: false);
    # checkpoint additionally carries per-layer `pre_quant_scale` (1-D bf16).
    "NVFP4_AWQ",
    # W4A16 NVFP4 (4-bit float weights, fp16/bf16 activations).
    "W4A16_NVFP4",
    # MXFP8
    "MXFP8",
    # MIXED_PRECISION,
    "MIXED_PRECISION",
]
```

- [ ] **Step 4: Extend `ModelOptNvFp4Config` to carry the `pre_quant_scale` flag and accept NVFP4_AWQ**

In the same file, replace `ModelOptNvFp4Config.__init__` parameter list and dispatch block at lines 1005-1043:

```python
class ModelOptNvFp4Config(ModelOptQuantConfigBase):
    """Config class for ModelOpt FP4."""

    def __init__(
        self,
        quant_method: str = "NVFP4",
        is_checkpoint_nvfp4_serialized: bool = False,
        kv_cache_quant_algo: str | None = None,
        exclude_modules: list[str] | None = None,
        group_size: int = 16,
        has_pre_quant_scale: bool = False,
    ) -> None:
        if exclude_modules is None:
            exclude_modules = []
        super().__init__(exclude_modules)
        self.quant_method = quant_method
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        self.has_pre_quant_scale = has_pre_quant_scale
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected ModelOpt NVFP4 checkpoint (quant_algo=%s). Please "
                "note that the format is experimental and could change in "
                "future.",
                quant_method,
            )

            self.group_size = group_size
            self.kv_cache_quant_algo = kv_cache_quant_algo

        # Select LinearMethod implementation based on quant_algo (FP8 pattern).
        # NVFP4 / NVFP4_AWQ -> W4A4: cutlass NVFP4 GEMM with input quantization.
        #   NVFP4_AWQ additionally applies a per-input-channel `pre_quant_scale`
        #   to activations before the GEMM (AWQ smoothing). Weight layout is
        #   identical to plain NVFP4.
        # W4A16_NVFP4       -> W4A16: FP4 Marlin GEMM with bf16/fp16 activations
        if quant_method in ("NVFP4", "NVFP4_AWQ"):
            self.LinearMethodCls = ModelOptNvFp4LinearMethod
        elif quant_method == "W4A16_NVFP4":
            self.LinearMethodCls = ModelOptNvFp4W4A16LinearMethod
        else:
            raise ValueError(
                f"Unsupported ModelOpt NVFP4 quant_algo: {quant_method}. "
                "Supported: NVFP4 / NVFP4_AWQ / W4A16_NVFP4."
            )
```

- [ ] **Step 5: Plumb `has_pre_quant_scale` through `_from_config`**

In the same file, replace `_from_config` body (lines 1064-1100):

```python
    @classmethod
    def _from_config(
        cls,
        *,
        quant_method: str,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        original_config: dict[str, Any],
        group_size: int | None,
        **kwargs: Any,
    ) -> "ModelOptNvFp4Config":
        is_checkpoint_nvfp4_serialized = "NVFP4" in quant_method

        if group_size is None:
            group_size = 16  # Default value

        # For FP4, these fields are required
        has_pre_quant_scale = False
        if is_checkpoint_nvfp4_serialized and "quantization" in original_config:
            # Check if required fields are present in the quantization config
            quant_config = original_config["quantization"]
            required_fields = ["group_size", "kv_cache_quant_algo", "exclude_modules"]
            missing_fields = [
                field for field in required_fields if field not in quant_config
            ]
            if missing_fields:
                raise ValueError(
                    f"NVFP4 quantization requires the following fields in "
                    f"hf_quant_config.json: {missing_fields}"
                )
            has_pre_quant_scale = bool(quant_config.get("pre_quant_scale", False))

        return cls(
            quant_method,
            is_checkpoint_nvfp4_serialized,
            kv_cache_quant_method,
            exclude_modules,
            group_size,
            has_pre_quant_scale,
        )
```

- [ ] **Step 6: Re-run the new tests; confirm they pass**

```bash
.venv/bin/python -m pytest tests/quantization/test_modelopt_nvfp4_awq.py::test_nvfp4_awq_in_quant_algos_whitelist tests/quantization/test_modelopt_nvfp4_awq.py::test_nvfp4_awq_config_dispatches_to_nvfp4_linear_method -v
```

Expected: both PASS.

- [ ] **Step 7: Run the existing modelopt test file to confirm no regression**

```bash
.venv/bin/python -m pytest tests/quantization/ -k "modelopt" -v 2>&1 | tail -20
```

Expected: existing tests still PASS (no NameError, no behavior change for `NVFP4` / `W4A16_NVFP4` / FP8 paths).

- [ ] **Step 8: Commit**

```bash
git add vllm/model_executor/layers/quantization/modelopt.py tests/quantization/test_modelopt_nvfp4_awq.py
git commit -m "$(cat <<'EOF'
feat(modelopt): accept NVFP4_AWQ algo in config dispatch

NVFP4_AWQ is NVFP4 weights + per-input-channel pre_quant_scale (AWQ
activation smoothing).  Weight layout is identical to plain NVFP4
(has_zero_point: false), so the same ModelOptNvFp4LinearMethod is
reused; the scale is plumbed via has_pre_quant_scale and consumed in
later commits.

This commit only unblocks config-time validation. It does not yet
register or apply pre_quant_scale; loading an NVFP4_AWQ checkpoint
after this commit alone would either ignore the scale (silent accuracy
loss) or fail later at load_weights when no destination parameter
exists for the on-disk pre_quant_scale tensor.

Co-authored-by: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Register `pre_quant_scale` parameter in `ModelOptNvFp4LinearMethod.create_weights`

**Files:**

- Modify: `vllm/model_executor/layers/quantization/modelopt.py:1119-1191`
- Test: `tests/quantization/test_modelopt_nvfp4_awq.py` (append)

- [ ] **Step 1: Add failing tests for parameter registration**

Append to `tests/quantization/test_modelopt_nvfp4_awq.py`:

```python
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
```

- [ ] **Step 2: Run the new tests; confirm they fail**

```bash
.venv/bin/python -m pytest tests/quantization/test_modelopt_nvfp4_awq.py::test_pre_quant_scale_registered_when_enabled tests/quantization/test_modelopt_nvfp4_awq.py::test_pre_quant_scale_absent_for_plain_nvfp4 -v
```

Expected: `test_pre_quant_scale_registered_when_enabled` FAILS with `AssertionError: create_weights must register pre_quant_scale ...`. `test_pre_quant_scale_absent_for_plain_nvfp4` PASSES (no change yet).

- [ ] **Step 3: Register the parameter inside `create_weights`**

In `vllm/model_executor/layers/quantization/modelopt.py`, add `RowvLLMParameter` to the import block at the top of the file (find the import line that already pulls `ModelWeightParameter, PerTensorScaleParameter` from `vllm.model_executor.parameter` and add `RowvLLMParameter` to the same import list).

Then in `ModelOptNvFp4LinearMethod.create_weights`, immediately after the existing `layer.register_parameter("weight_scale", weight_scale)` line (currently line 1191), append:

```python
        # AWQ pre-quant scale: per-input-channel multiplier applied to
        # activations before NVFP4 input quantization. Only present when the
        # checkpoint was produced by ModelOpt with `pre_quant_scale: true`
        # (quant_algo NVFP4_AWQ). Row-parallel sharding: sliced along dim 0
        # by input_size_per_partition. For ReplicatedLinear / non-sharded
        # callers the row-parallel loader is a no-op slice.
        if self.quant_config.has_pre_quant_scale:
            pre_quant_scale = RowvLLMParameter(
                data=torch.empty(input_size_per_partition, dtype=torch.bfloat16),
                input_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("pre_quant_scale", pre_quant_scale)
```

- [ ] **Step 4: Re-run the new tests; confirm both pass**

```bash
.venv/bin/python -m pytest tests/quantization/test_modelopt_nvfp4_awq.py -v
```

Expected: all four tests written so far PASS.

- [ ] **Step 5: Re-run the broader modelopt test suite for regressions**

```bash
.venv/bin/python -m pytest tests/quantization/ -k "modelopt" -v 2>&1 | tail -20
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add vllm/model_executor/layers/quantization/modelopt.py tests/quantization/test_modelopt_nvfp4_awq.py
git commit -m "$(cat <<'EOF'
feat(modelopt): register pre_quant_scale param for NVFP4_AWQ

Add conditional registration of a 1-D bf16 `pre_quant_scale` parameter
inside ModelOptNvFp4LinearMethod.create_weights, gated on
quant_config.has_pre_quant_scale.  Uses RowvLLMParameter(input_dim=0)
so RowParallelLinear gets a per-rank slice and ReplicatedLinear gets
the full vector; column-parallel callers never see this code path
because their checkpoints have no pre_quant_scale tensor.

apply() still ignores the new parameter; runtime use lands in the
next commit.

Co-authored-by: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Detect NVFP4_AWQ on MoE models and fail loudly

**Files:**

- Modify: `vllm/model_executor/layers/quantization/modelopt.py:1388-1404` (`ModelOptNvFp4FusedMoE.__init__`)
- Test: `tests/quantization/test_modelopt_nvfp4_awq.py` (append)

Reason: MoE support requires either modifying the modular MoE kernel or folding `pre_quant_scale` into per-group weight scales (precision loss). Out of scope for v1. Without this guard, a user loading a hypothetical NVFP4_AWQ MoE checkpoint would silently get wrong outputs.

- [ ] **Step 1: Add failing test**

Append to `tests/quantization/test_modelopt_nvfp4_awq.py`:

```python
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
```

- [ ] **Step 2: Run; confirm it fails**

```bash
.venv/bin/python -m pytest tests/quantization/test_modelopt_nvfp4_awq.py::test_moe_method_rejects_nvfp4_awq -v
```

Expected: FAIL — either `NotImplementedError` not raised, or a different error (e.g. AttributeError accessing stub) happens first.

- [ ] **Step 3: Add the guard at the top of `ModelOptNvFp4FusedMoE.__init__`**

In `vllm/model_executor/layers/quantization/modelopt.py`, modify `ModelOptNvFp4FusedMoE.__init__` (starting at line 1388). Insert the guard immediately after the `super().__init__(moe_config)` call:

```python
    def __init__(
        self,
        quant_config: ModelOptNvFp4Config,
        moe_config: FusedMoEConfig,
    ) -> None:
        super().__init__(moe_config)
        if getattr(quant_config, "has_pre_quant_scale", False):
            raise NotImplementedError(
                "NVFP4_AWQ (pre_quant_scale) on MoE layers is not yet "
                "implemented in vLLM. The modular NVFP4 MoE kernel has no "
                "hook for per-input-channel activation smoothing, and "
                "folding pre_quant_scale into per-group fp8 weight scales "
                "is lossy. Use the dense NVFP4_AWQ path (Linear only) or a "
                "plain NVFP4 / W4A16_NVFP4 checkpoint for MoE models."
            )
        self.quant_config = quant_config
        # Select experts implementation.
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=kNvfp4Dynamic,
        )

        self.use_global_sf = is_global_sf_supported_for_nvfp4_backend(
            self.nvfp4_backend
        )
```

- [ ] **Step 4: Run the MoE test; confirm it passes**

```bash
.venv/bin/python -m pytest tests/quantization/test_modelopt_nvfp4_awq.py::test_moe_method_rejects_nvfp4_awq -v
```

Expected: PASS.

- [ ] **Step 5: Run the full new-test file to confirm no other regression**

```bash
.venv/bin/python -m pytest tests/quantization/test_modelopt_nvfp4_awq.py -v
```

Expected: all five tests PASS.

- [ ] **Step 6: Commit**

```bash
git add vllm/model_executor/layers/quantization/modelopt.py tests/quantization/test_modelopt_nvfp4_awq.py
git commit -m "$(cat <<'EOF'
feat(modelopt): reject NVFP4_AWQ on MoE with clear NotImplementedError

The modular NVFP4 MoE kernel has no hook for per-input-channel activation
smoothing, and folding pre_quant_scale into per-group fp8 weight scales
would lose precision (scale varies within a 16-elem group).  Raise at
__init__ so users see the failure mode immediately rather than silently
getting wrong outputs.

Dense Linear path (next commit) is unaffected.

Co-authored-by: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Apply `pre_quant_scale` to activations inside `ModelOptNvFp4LinearMethod.apply`

**Files:**

- Modify: `vllm/model_executor/layers/quantization/modelopt.py:1193-1232` (`process_weights_after_loading` and `apply`)
- Test: `tests/quantization/test_modelopt_nvfp4_awq.py` (append)

Numerical-equivalence strategy: we cannot run the real NVFP4 CUTLASS kernel on CPU, but we can verify the `apply()` wrapper performs `x = x * pre_quant_scale` before delegating, by stubbing `self.kernel.apply_weights` and capturing its `x` argument.

Sign convention: defaults to multiply (`x * pre_quant_scale`). Task 0 may override this — if Task 0 captured values predominantly > 1, change `*` → `/` in Step 3 below.

- [ ] **Step 1: Add failing test for apply() scaling**

Append to `tests/quantization/test_modelopt_nvfp4_awq.py`:

```python
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

    class _StubLayer:
        pass

    layer = _StubLayer()
    layer.pre_quant_scale = torch.tensor(
        [2.0, 0.5, 4.0, 0.25], dtype=torch.bfloat16
    )

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

    class _StubLayer:
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
```

- [ ] **Step 2: Run; confirm both new tests fail**

```bash
.venv/bin/python -m pytest tests/quantization/test_modelopt_nvfp4_awq.py::test_apply_multiplies_input_by_pre_quant_scale tests/quantization/test_modelopt_nvfp4_awq.py::test_apply_unchanged_when_pre_quant_scale_absent -v
```

Expected: `test_apply_multiplies_input_by_pre_quant_scale` FAILS (x not scaled). `test_apply_unchanged_when_pre_quant_scale_absent` PASSES (current apply() already passes x through).

- [ ] **Step 3: Modify `apply()` to multiply input by `pre_quant_scale` when present**

In `vllm/model_executor/layers/quantization/modelopt.py`, replace the existing `ModelOptNvFp4LinearMethod.apply` (lines 1226-1232) with:

```python
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # AWQ activation smoothing: applied per-input-channel before
        # NVFP4 input quantization.  Only NVFP4_AWQ checkpoints register
        # this parameter (see create_weights); plain NVFP4 layers skip
        # the multiply entirely.
        pre_quant_scale = getattr(layer, "pre_quant_scale", None)
        if pre_quant_scale is not None:
            x = x * pre_quant_scale.to(x.dtype)
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)
```

If Task 0 indicated divide convention, use `x = x / pre_quant_scale.to(x.dtype)` instead.

- [ ] **Step 4: Modify `process_weights_after_loading` to preserve `pre_quant_scale` through kernel format conversion**

The existing implementation calls `self.kernel.process_weights_after_loading(layer)` which may rebind or rename parameters. We must ensure `layer.pre_quant_scale` survives. Inspect the end of the existing method (around line 1224 `self.kernel.process_weights_after_loading(layer)`) and replace the trailing block with:

```python
        # Stash pre_quant_scale across kernel format conversion so apply()
        # can still find it on the layer afterwards.
        pre_quant_scale = getattr(layer, "pre_quant_scale", None)

        # Convert layer to NVFP4 linear kernel format
        self.kernel.process_weights_after_loading(layer)

        if pre_quant_scale is not None and not hasattr(layer, "pre_quant_scale"):
            layer.pre_quant_scale = Parameter(
                pre_quant_scale.data.detach(), requires_grad=False
            )
```

(`Parameter` is already imported at the top of the file alongside `torch`.)

- [ ] **Step 5: Re-run the apply() tests; confirm both pass**

```bash
.venv/bin/python -m pytest tests/quantization/test_modelopt_nvfp4_awq.py -v
```

Expected: all seven tests in this file PASS.

- [ ] **Step 6: Run the broader modelopt suite for regressions**

```bash
.venv/bin/python -m pytest tests/quantization/ -k "modelopt" -v 2>&1 | tail -20
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add vllm/model_executor/layers/quantization/modelopt.py tests/quantization/test_modelopt_nvfp4_awq.py
git commit -m "$(cat <<'EOF'
feat(modelopt): apply pre_quant_scale to activations in NVFP4_AWQ linear

Multiply x by layer.pre_quant_scale (cast to x.dtype) inside
ModelOptNvFp4LinearMethod.apply, immediately before delegating to the
NVFP4 CUTLASS kernel.  Plain NVFP4 layers (no pre_quant_scale attr)
skip the multiply, so the hot path for non-AWQ models is unchanged.

process_weights_after_loading now re-attaches pre_quant_scale on the
layer if the kernel's own post-load step drops or renames params, so
apply() always finds it.

This completes the dense Linear NVFP4_AWQ support.  MoE is still
rejected at __init__ (previous commit).

Co-authored-by: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Lint pass and full test sweep on the touched module

**Files:** none modified (verification only)

- [ ] **Step 1: Run ruff on the modified file**

```bash
pre-commit run ruff-check --files vllm/model_executor/layers/quantization/modelopt.py tests/quantization/test_modelopt_nvfp4_awq.py
```

Expected: PASS (no ruff issues). If issues are reported, fix them with `pre-commit run ruff-format --files <files>` and re-run; do not commit unless lint is clean.

- [ ] **Step 2: Run the full new-test file plus the existing FP4 backend test**

```bash
.venv/bin/python -m pytest tests/quantization/test_modelopt_nvfp4_awq.py tests/v1/spec_decode/test_cutlass_experts_fp4_activation.py -v
```

Expected: all PASS.

- [ ] **Step 3: If lint required edits, commit them**

```bash
git status
# If there are unstaged formatting fixes:
git add vllm/model_executor/layers/quantization/modelopt.py tests/quantization/test_modelopt_nvfp4_awq.py
git commit -m "style(modelopt): ruff format pass for NVFP4_AWQ additions

Co-authored-by: Claude <noreply@anthropic.com>"
```

Otherwise skip the commit.

---

## Task 6: Thor end-to-end verification

**Files:** none (manual verification only — requires Thor GPU + container)

The unit tests stub the NVFP4 CUTLASS kernel; only an on-device load can prove the on-disk checkpoint actually loads and produces sane outputs.

- [ ] **Step 1: Push the commits to `jetson` and rebuild the Thor vLLM image**

On the dev box:

```bash
git push origin jetson
```

Then on Thor, rebuild the `vllm-jetson:thor` image per the project's normal rebuild procedure (out of scope for this plan — follow the team's vLLM Jetson build runbook).

- [ ] **Step 2: Flip the AEON-7 config back to `experimental`**

This step lives in the `llm_service_pack` repo, not vLLM. In that repo, edit `config/vllm/thor-aeon7-gemma-4-E4B-it-uncensored-NVFP4.yaml`:

- Change `status: invalid` → `status: experimental`
- Replace the long "無法啟動" block in `notes:` with a one-line "Requires vLLM with NVFP4_AWQ support (jetson branch ≥ <commit-sha>)."

Also flip the row in `docs/config-status.md` from `invalid` back to `experimental`.

(No vLLM-side change; included here for completeness so the implementer doesn't forget. Commit in the llm_service_pack repo.)

- [ ] **Step 3: Start the container and watch logs for successful load**

```bash
sudo docker compose up -d vllm
sudo docker logs llm-service-pack-vllm-1 -f
```

Expected log markers (PASS criteria):

- No `pydantic_core.ValidationError: ... ModelOpt currently only supports`.
- A line `Detected ModelOpt NVFP4 checkpoint (quant_algo=NVFP4_AWQ).`
- Model finishes loading (`Started vLLM server` or equivalent uvicorn ready line).
- No `KeyError` or `unexpected key pre_quant_scale` during weight load.

FAIL criteria (any of):

- Pydantic ValidationError reappears → Task 1 not picked up; check image build.
- `KeyError: ... pre_quant_scale` → Task 2 didn't run on this checkpoint variant; capture the offending key name and revisit `create_weights` shape.
- `NotImplementedError: NVFP4_AWQ ... MoE` → unexpected (E4B should be dense); confirm checkpoint architecture.

- [ ] **Step 4: Smoke-test generation quality**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "AEON-7/Gemma-4-E4B-it-Uncensored-NVFP4",
    "messages": [{"role": "user", "content": "Write one short sentence about the sea."}],
    "max_tokens": 64
  }' | head -50
```

Expected: a coherent English sentence about the sea. If the output is garbage tokens / repeated punctuation / pure noise:

- Most likely cause: sign convention wrong (multiply vs divide). Re-check Task 0's verification; flip the operator in `apply()` and re-test.
- Secondary cause: `pre_quant_scale` not sharded correctly across TP (irrelevant on Thor single-GPU but verify `tensor_parallel_size` in the yaml).

- [ ] **Step 5: Record the verification result in the llm_service_pack config**

Once Step 4 passes, in `llm_service_pack/config/vllm/thor-aeon7-gemma-4-E4B-it-uncensored-NVFP4.yaml`, change `status: experimental` → `status: verified`. Commit in that repo.

---

## Done When

- All seven unit tests in `tests/quantization/test_modelopt_nvfp4_awq.py` pass on the dev box (no GPU needed).
- `pre-commit run ruff-check --files <modified files>` is clean.
- Thor smoke test (Task 6 Step 4) returns a coherent generation from `AEON-7/Gemma-4-E4B-it-Uncensored-NVFP4`.
- vLLM commits exist on `origin/jetson`; `llm_service_pack` config flipped to `verified` and committed.

## Out of Scope (intentional)

- MoE support for NVFP4_AWQ (e.g. a hypothetical NVFP4_AWQ port of Gemma 4 26B-A4B). Guarded with a clear `NotImplementedError` in Task 3; revisit as a separate plan if/when such a checkpoint appears.
- W4A16_NVFP4_AWQ (Marlin path with pre_quant_scale). No such ModelOpt variant in the wild today; revisit when one appears.
- Upstreaming to `vllm-project/vllm`. Follow `AGENTS.md` duplicate-work checks and PR rules separately if/when we choose to upstream.
