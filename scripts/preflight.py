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
