# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.config.utils import replace
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


class DraftModelProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()

    def _raise_if_vocab_size_mismatch(self):
        self.speculative_config.verify_equal_vocab_size_if_draft_model()

    def _raise_if_draft_tp_mismatch(self):
        # Note(Tomas Ruiz) If we run the target model with TP > 1 and
        # the draft model with TP = 1, then the different TP ranks collide.
        # Specifically when all ranks compile the draft model on rank 0
        # (because TP=1), then the torch compile cache is overwritten and corrupted.
        # We need a mechanism like this: https://github.com/vllm-project/vllm/pull/5414
        # To prevent this error, we assert that both TP sizes must be the same.
        spec_cfg = self.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                f"Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' "
                f"must be the same. Got {draft_tp} and {tgt_tp}. "
                "Please pass 'draft_tensor_parallel_size' in the speculative_config."
            )

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        spec = self.speculative_config

        return replace(
            base,
            quant_config=None,
            parallel_config=replace(
                spec.draft_parallel_config,
                rank=self.vllm_config.parallel_config.rank,
            ),
            model_config=spec.draft_model_config,
        )

    @override
    def _get_model(self) -> nn.Module:
        import re as _re

        from vllm.compilation.backends import set_model_tag
        from vllm.model_executor.models import ModelRegistry

        draft_vllm_config = self._create_draft_vllm_config()

        # compressed-tensors uses exact string matching for its ignore list.
        # Because get_model is called with prefix="draft_model", every layer's
        # vLLM prefix begins with "draft_model." — but _maybe_apply_model_mapping
        # (called in __new__) only remaps HF→vLLM names without adding that
        # prefix, so should_ignore_layer returns False for layers that should
        # be ignored (e.g. per_layer_input_gate).  Those layers are then
        # initialised as quantised (qweight/scales instead of .weight), while
        # the checkpoint stores them as plain float — causing KeyError in
        # load_weights.
        #
        # Fix: pre-apply the model mapper here, then convert every exact-path
        # ignore entry to a prefix-agnostic regex so it matches both standalone
        # and "draft_model."-prefixed layer names.
        quant_config = draft_vllm_config.quant_config
        if quant_config is not None and getattr(quant_config, "ignore", None):
            try:
                model_cls, _ = ModelRegistry.resolve_model_cls(
                    draft_vllm_config.model_config.hf_config.architectures,
                    draft_vllm_config.model_config,
                )
                mapper = getattr(model_cls, "hf_to_vllm_mapper", None)
                if mapper is not None:
                    quant_config.apply_vllm_mapper(mapper)
            except Exception:
                pass

            quant_config.ignore = [
                f"re:(?:.*\\.)?{_re.escape(entry)}$"
                if "." in entry and not entry.startswith("re:")
                else entry
                for entry in quant_config.ignore
            ]

        with set_model_tag("draft_model"):
            model = get_model(
                vllm_config=draft_vllm_config,
                prefix="draft_model",
            )
        return model

    @override
    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        # Draft models don't share embeddings with the target model
        pass

    @override
    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        # Draft models don't share lm_head with the target model
        pass
