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
