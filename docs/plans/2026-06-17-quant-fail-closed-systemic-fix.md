# Quantization Fail-Fast Canary and ModelOpt Dispatch Implementation Plan

> **For implementers:** Execute this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking. All Python commands use `.venv/bin/python`
> per repository rules. This plan is repository-local; do not run production,
> deployment, or migration commands.

**Goal:** Make the quantization failure path more predictable by fixing the
actual ModelOpt excluded-`ParallelLMHead` dispatch inconsistency, pinning the
embedding/tie-weight architecture with tests, and adding a local preflight
canary for model-load regressions.

**Architecture:** `VocabParallelEmbedding` owns the embedding fallback contract:
when `quant_config.get_quant_method(...)` returns `None`, it installs
`UnquantizedEmbeddingMethod`. `ParallelLMHead` is a `VocabParallelEmbedding`
subclass, not a `LinearBase`, so excluded lm heads should return `None` when the
intended result is an unquantized embedding-style head. Build-system and
checkpoint-validation failures cannot be made safe by permissive Python
fallbacks, so they are detected with an explicit preflight smoke test.

**Tech Stack:**

- vLLM quantization configs in `vllm/model_executor/layers/quantization/`
- `VocabParallelEmbedding`, `ParallelLMHead`, and `UnquantizedEmbeddingMethod`
- `pytest` for targeted unit coverage
- `scripts/preflight.py` using `vllm.LLM.generate(...)`

---

## Confirmed Facts

### Existing architecture

- `VocabParallelEmbedding.__init__` calls
  `quant_config.get_quant_method(self, prefix=prefix)` when a quant config is
  present. If the result is `None`, it installs `UnquantizedEmbeddingMethod`.
- A real `VocabParallelEmbedding` requires its quant method class to override
  `embedding()`. This is enforced by `method_has_implemented_embedding(...)`
  before the layer can be used.
- `ParallelLMHead` inherits from `VocabParallelEmbedding`. It does not execute
  the embedding forward path, but tied-embedding models call
  `ParallelLMHead.tie_weights(...)`, which delegates to the current
  `quant_method.tie_weights(...)`.
- `QuantizeMethodBase.tie_weights(...)` already provides a safe default:
  swap `layer.quant_method` to `UnquantizedEmbeddingMethod` and delegate.
- `QuantizeMethodBase.embedding(...)` still raises `NotImplementedError`. That
  is currently consistent with the explicit embedding-support check and should
  not be changed in this plan.

### Confirmed issues

1. **Stale ModelOpt test expectation:** `tests/quantization/test_modelopt.py`
   still expects excluded ModelOpt NVFP4 `ParallelLMHead` to return
   `UnquantizedLinearMethod`, but current code returns `None`.
2. **ModelOpt mixed-precision inconsistency:** `ModelOptMixedPrecisionConfig`
   still returns `UnquantizedLinearMethod()` for an excluded `ParallelLMHead`.
   This may not crash after the current `tie_weights` base fallback, but it
   relies on a linear method for an embedding-style layer and diverges from the
   safer ModelOpt NVFP4 behavior.
3. **AWQ/FP8 are not the same bug:** current AWQ and FP8 configs only enter the
   skipped-layer branch for `LinearBase`; `ParallelLMHead` is not a
   `LinearBase`. Do not change AWQ/FP8 in this plan unless a new failing test
   proves an actual bug.
4. **Validation/build failures need detection:** unknown quantization values,
   missing compiled symbols, and runtime kernel arch guards cannot be safely
   solved by relaxing validation. They need an explicit canary before deploy or
   model onboarding.

---

## File Map

| File | Responsibility | Action |
|---|---|---|
| `vllm/model_executor/layers/quantization/utils/excluded.py` | Shared helper for excluded layer dispatch | Create |
| `vllm/model_executor/layers/quantization/modelopt.py` | ModelOpt NVFP4 and mixed-precision dispatch | Modify only ModelOpt excluded branches |
| `tests/quantization/test_modelopt.py` | Existing ModelOpt dispatch tests | Update stale NVFP4 expectation; add mixed-precision regression |
| `tests/quantization/test_excluded_layer_dispatch.py` | Direct helper and base-contract tests | Create |
| `scripts/preflight.py` | Local model-load smoke test | Create |
| `docs/preflight.md` | Operator/developer preflight usage | Create |

---

## Task 1: Pin the Existing Embedding and Excluded-Layer Contracts

**Files:**

- Modify: `tests/quantization/test_modelopt.py`
- Create: `tests/quantization/test_excluded_layer_dispatch.py`

- [ ] **Step 1: Update the stale ModelOpt NVFP4 test expectation**

In `tests/quantization/test_modelopt.py`, update the excluded lm-head test so
it documents the current and desired architecture:

```python
def test_modelopt_nvfp4_leaves_excluded_parallel_lm_head_unquantized():
    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=["lm_head"],
    )

    method = config.get_quant_method(_mock_lm_head(), prefix="lm_head")

    assert method is None
```

- [ ] **Step 2: Add a mixed-precision regression test**

Append this test near the existing ModelOpt lm-head tests in
`tests/quantization/test_modelopt.py`:

```python
def test_modelopt_mixed_precision_leaves_excluded_parallel_lm_head_unquantized():
    config = _mixed_precision_config(
        {"model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"}}
    )
    config.exclude_modules = ["lm_head"]

    method = config.get_quant_method(_mock_lm_head(), prefix="lm_head")

    assert method is None
```

- [ ] **Step 3: Add direct helper and base-contract tests**

Create `tests/quantization/test_excluded_layer_dispatch.py`:

```python
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
```

- [ ] **Step 4: Run tests and confirm the new mixed-precision/helper checks fail**

Run:

```bash
.venv/bin/python -m pytest \
  tests/quantization/test_modelopt.py::test_modelopt_nvfp4_leaves_excluded_parallel_lm_head_unquantized \
  tests/quantization/test_modelopt.py::test_modelopt_mixed_precision_leaves_excluded_parallel_lm_head_unquantized \
  tests/quantization/test_excluded_layer_dispatch.py \
  -v
```

Expected:

- NVFP4 test passes after Step 1.
- Mixed-precision test fails because it currently returns
  `UnquantizedLinearMethod()`.
- Helper tests fail with `ModuleNotFoundError` until Task 2 creates the helper.

---

## Task 2: Centralize ModelOpt Excluded-Layer Dispatch

**Files:**

- Create: `vllm/model_executor/layers/quantization/utils/excluded.py`
- Modify: `vllm/model_executor/layers/quantization/modelopt.py`
- Test: `tests/quantization/test_modelopt.py`
- Test: `tests/quantization/test_excluded_layer_dispatch.py`

- [ ] **Step 1: Create the shared helper**

Create `vllm/model_executor/layers/quantization/utils/excluded.py`:

```python
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
```

- [ ] **Step 2: Use the helper in `ModelOptQuantConfigBase.get_quant_method`**

In `vllm/model_executor/layers/quantization/modelopt.py`, replace only the
excluded branch in `ModelOptQuantConfigBase.get_quant_method` with:

```python
        # handle exclusion
        if self.is_layer_excluded(prefix):
            from vllm.model_executor.layers.quantization.utils.excluded import (
                make_excluded_quant_method,
            )

            return make_excluded_quant_method(layer)
```

- [ ] **Step 3: Use the helper in `ModelOptMixedPrecisionConfig.get_quant_method`**

In the mixed-precision `get_quant_method`, replace the excluded branch with:

```python
        # Excluded layers
        if self.is_layer_excluded(prefix):
            from vllm.model_executor.layers.quantization.utils.excluded import (
                make_excluded_quant_method,
            )

            return make_excluded_quant_method(layer)
```

- [ ] **Step 4: Run the targeted dispatch tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/quantization/test_modelopt.py::test_modelopt_nvfp4_leaves_excluded_parallel_lm_head_unquantized \
  tests/quantization/test_modelopt.py::test_modelopt_mixed_precision_leaves_excluded_parallel_lm_head_unquantized \
  tests/quantization/test_excluded_layer_dispatch.py \
  -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Run the broader ModelOpt quantization tests**

Run:

```bash
.venv/bin/python -m pytest tests/quantization/test_modelopt.py -v
```

Expected: all tests in `test_modelopt.py` pass, except tests skipped due to
missing local checkpoints or unsupported local hardware.

---

## Task 3: Add a Local Preflight Canary

**Files:**

- Create: `scripts/preflight.py`
- Create: `docs/preflight.md`

- [ ] **Step 1: Create `scripts/preflight.py`**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a local model-load smoke test."""

from __future__ import annotations

import argparse
import sys
import traceback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local model directory or model id")
    parser.add_argument(
        "--dummy-weights",
        action="store_true",
        help="Use load_format=dummy to skip real checkpoint weight loading.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model/tokenizer code when the checkpoint requires it.",
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from vllm import LLM, SamplingParams

    try:
        llm = LLM(
            model=args.model,
            enforce_eager=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            load_format="dummy" if args.dummy_weights else "auto",
            max_model_len=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
        )
        outputs = llm.generate(
            ["preflight test"],
            SamplingParams(max_tokens=args.max_tokens),
        )
        text = outputs[0].outputs[0].text
        print(f"OK: generated {len(text)} chars: {text!r}")
        return 0
    except Exception:
        traceback.print_exc()
        print("FAIL: preflight model load failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Create `docs/preflight.md`**

```markdown
# Preflight smoke test

`scripts/preflight.py` loads a model through vLLM and generates a few tokens.
Use it before deploying a rebuilt image or before accepting a new checkpoint.

## Build or image canary

Use dummy weights when the goal is to catch import, model construction,
quant-method setup, tied-weight setup, and obvious forward-path failures without
loading a full checkpoint:

```bash
.venv/bin/python scripts/preflight.py /models/<canary-model> --dummy-weights
```

`--dummy-weights` does not validate real checkpoint tensor names, tensor shapes,
or checkpoint-specific quantization scales.

## New model onboarding

Use real weights when validating a new model:

```bash
.venv/bin/python scripts/preflight.py /models/<new-model>
```

If the checkpoint requires custom remote code, pass:

```bash
.venv/bin/python scripts/preflight.py /models/<new-model> --trust-remote-code
```

## What it can catch

- Quantization config validation failures.
- Missing compiled symbols during import or first use.
- Model initialization failures, including tied-weight setup.
- Runtime kernel guards that raise during a short forward pass.

## What it cannot catch

- Accuracy regressions.
- Long-context KV-cache issues.
- Multi-GPU tensor or pipeline parallel issues.
- Checkpoint-specific problems when `--dummy-weights` is used.
- Silent wrong-output bugs that do not raise.

## Runtime healthcheck caution

Do not add this blindly as a healthcheck to a live serving container. Starting a
second `LLM` inside the same container can compete for GPU memory and cause
restart loops. Prefer running it as a post-build or pre-deploy canary. For a live
server healthcheck, use the server health endpoint instead.

```

- [ ] **Step 3: Sanity-check the CLI**

Run:

```bash
.venv/bin/python scripts/preflight.py --help
```

Expected: argparse help text and exit code 0.

- [ ] **Step 4: Run a real canary when hardware/model access is available**

Run one or more of these on the matching Thor/Orin environment, not on a
production service:

```bash
.venv/bin/python scripts/preflight.py /models/AEON-7/Gemma-4-E4B-it-Uncensored-NVFP4
.venv/bin/python scripts/preflight.py /models/AxionML/Gemma-4-12B-NVFP4
.venv/bin/python scripts/preflight.py /models/Intel/gemma-4-12B-it-int8-AutoRound
```

Expected: exit code 0 and an `OK: generated ...` line. If a command fails,
capture the traceback and fix the specific validation, import, init, or forward
failure before using that build/model.

---

## Validation Summary

- Unit tests:

```bash
.venv/bin/python -m pytest \
  tests/quantization/test_modelopt.py::test_modelopt_nvfp4_leaves_excluded_parallel_lm_head_unquantized \
  tests/quantization/test_modelopt.py::test_modelopt_mixed_precision_leaves_excluded_parallel_lm_head_unquantized \
  tests/quantization/test_excluded_layer_dispatch.py \
  -v
```

- Broader ModelOpt regression:

```bash
.venv/bin/python -m pytest tests/quantization/test_modelopt.py -v
```

- CLI sanity:

```bash
.venv/bin/python scripts/preflight.py --help
```

- Optional full-environment canary:

```bash
.venv/bin/python scripts/preflight.py /models/<model-path>
```

---

## Done When

- Existing ModelOpt NVFP4 excluded-lm-head test expects `None`.
- ModelOpt mixed-precision excluded `ParallelLMHead` returns `None`.
- The helper returns `None` for `ParallelLMHead`,
  `UnquantizedLinearMethod` for `LinearBase`, and `None` for unrelated modules.
- `QuantizeMethodBase.embedding()` remains fail-fast and covered by a test.
- Targeted tests pass in the local development environment.
- `scripts/preflight.py --help` exits 0.
- Real-model preflight is run on Thor/Orin before deployment or model
  onboarding when that hardware/model access is available.

## Explicitly Out of Scope

- Changing `QuantizeMethodBase.embedding()` to an unquantized fallback.
- Changing AWQ or FP8 excluded-layer behavior without a new failing test.
- Relaxing pydantic/config validation for unknown quantization algorithms.
- CMake/kernel build-system changes.
- Production deployment, image push, or production healthcheck changes.
- Accuracy validation beyond the smoke test.
