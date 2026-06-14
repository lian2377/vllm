# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 MTP (Multi-Token Prediction) proposer for speculative decoding.

The Gemma4 assistant model runs all decoder layers per draft step
(producing one token), and all its attention layers share KV cache
with the target model via cross-model KV sharing.
"""

from collections import defaultdict
from copy import copy

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config, replace
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


class Gemma4Proposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        # All draft steps predict from the same position (the last
        # target-model position), so positions and seq_lens must not
        # advance between steps.
        self.constant_draft_positions = True

        # Per-group block tables for multi-group KV cache models.
        # Populated by gpu_model_runner during _prepare_inputs.
        self._per_group_block_tables: dict[int, torch.Tensor] = {}

        # Centroids CUDA graphs — populated in load_model if centroids
        # masking is active. _centroids_sizes is pre-sorted for fast
        # lookup in _greedy_sample.
        self._centroids_sizes: list[int] = []
        self._centroids_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._centroids_inputs: dict[int, torch.Tensor] = {}
        self._centroids_outputs: dict[int, torch.Tensor] = {}

    def set_per_group_block_table(self, gid: int, block_table: torch.Tensor) -> None:
        self._per_group_block_tables[gid] = block_table

    def model_returns_tuple(self) -> bool:
        # forward() returns (draft_hidden_states, backbone_hidden_states).
        # The proposer uses draft_hidden_states for compute_logits and
        # backbone_hidden_states for the hidden-state feedback buffer.
        return True

    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        """Build attention metadata using the correct block table per group.

        Gemma4 has multiple KV cache groups (sliding vs full attention)
        with different block tables.  The base class receives a single
        common_attn_metadata whose block_table belongs to one group.
        We swap in the correct block table for each draft attention group.
        """
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        batch_size = common_attn_metadata.batch_size()
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid in self._per_group_block_tables:
                cm = copy(common_attn_metadata)
                # Slice to actual batch size to match cu_seqlens_q dimension.
                # The stored block tables may be padded (num_reqs_padded) from
                # the target forward pass, but the drafter operates on the
                # unpadded batch.
                cm.block_table_tensor = self._per_group_block_tables[gid][:batch_size]
            else:
                cm = common_attn_metadata
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cm, draft_index=draft_index
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._centroids_sizes:
            T = hidden_states.shape[0]
            for size in self._centroids_sizes:
                if size >= T:
                    self._centroids_inputs[size][:T].copy_(hidden_states)
                    self._centroids_graphs[size].replay()
                    return self._centroids_outputs[size][:T].clone()
            return self.model.get_top_tokens(hidden_states)
        return super()._greedy_sample(hidden_states)

    def _setup_centroids_cuda_graphs(self) -> None:
        """Capture CUDA graphs for centroids get_top_tokens at key sizes."""
        masked_emb = self.model.masked_embedding
        lm_head_weight = self.model._get_full_lm_head_weight()

        for size in [1, 2, 4, 8, 16, 32, 64]:
            static_input = torch.zeros(
                size,
                masked_emb.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(3):
                masked_emb.get_top_tokens(static_input, lm_head_weight)
            torch.accelerator.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_output = masked_emb.get_top_tokens(
                    static_input,
                    lm_head_weight,
                )
            self._centroids_graphs[size] = g
            self._centroids_inputs[size] = static_input
            self._centroids_outputs[size] = static_output

        self._centroids_sizes = sorted(self._centroids_graphs)
        logger.info(
            "Gemma4 MTP: captured centroids CUDA graphs for sizes %s.",
            self._centroids_sizes,
        )

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Preserve the target's forced TRITON_ATTN backend for draft layers,
        and replace model_config/quant_config with the draft model's own values.

        Gemma4 forces TRITON_ATTN due to heterogeneous head dimensions
        (head_dim=256 sliding, global_head_dim=512 full). The base class
        resets attention_config.backend to None for draft models, causing
        sliding layers to fall back to FLASH_ATTN which cannot handle
        KV-shared cache. Override to carry the target's backend through.

        Also replaces model_config and quant_config with draft model's own
        values. SpecDecodeBaseProposer._get_model() passes vllm_config (built
        from the target's config) to the model constructor, but
        Gemma4ForConditionalGeneration.__init__ reads
        vllm_config.model_config.hf_config to determine
        hidden_size_per_layer_input. Without this replacement the target's value
        (0 for 26B) is used instead of the draft's (256 for E4B), so
        per_layer_input_gate initialises to None and load_weights crashes with
        KeyError on layers.*.per_layer_input_gate.weight.
        """
        base = super()._create_draft_vllm_config()

        # Carry target's attention backend through (original logic unchanged)
        target_backend = self.vllm_config.attention_config.backend
        if target_backend is not None:
            base = replace(
                base,
                attention_config=replace(
                    base.attention_config,
                    backend=target_backend,
                ),
            )

        # Replace model_config and quant_config with draft model's own values.
        # VllmConfig.quant_config is an independent field; replace() on model_config
        # alone does not re-derive it, so both must be updated explicitly.
        draft_model_config = self.speculative_config.draft_model_config
        draft_quant_config = VllmConfig.get_quantization_config(
            draft_model_config, self.vllm_config.load_config
        )
        return replace(
            base,
            model_config=draft_model_config,
            quant_config=draft_quant_config,
        )

    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        """Gemma4 MTP always keeps its own draft-dim lm_head.

        The draft model's lm_head operates in draft hidden_size (e.g. 256),
        which differs from the target's backbone hidden_size (e.g. 1536).
        Sharing would break compute_logits (and centroids masking when
        use_ordered_embeddings is enabled).
        """
        logger.info(
            "Gemma4 MTP: keeping draft model's own lm_head (draft_dim != backbone_dim)."
        )

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        super().load_model(target_model)

        self._setup_gemma4_kv_sharing(target_attn_layer_names)

        if getattr(self.model, "masked_embedding", None) is not None:
            self._setup_centroids_cuda_graphs()

    def _build_layer_spec_mapping(
        self,
        kv_cache_config: KVCacheConfig,
        all_attn_layers: dict,
    ) -> tuple[dict[str, int], dict[str, KVCacheSpec]]:
        """Extend base mapping with KV-sharing fallback.

        Gemma4 draft layers that share KV cache with a target layer are
        listed in the kv_cache_group but have no entry of their own in
        UniformTypeKVCacheSpecs.kv_cache_specs.  For those layers we fall
        back to the spec of their KV-sharing target.
        """
        layer_to_gid, layer_to_spec = super()._build_layer_spec_mapping(
            kv_cache_config, all_attn_layers
        )
        for group in kv_cache_config.kv_cache_groups:
            if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
                continue
            specs = group.kv_cache_spec.kv_cache_specs
            for ln in group.layer_names:
                if ln in specs:
                    continue
                tgt = getattr(
                    all_attn_layers.get(ln),
                    "kv_sharing_target_layer_name",
                    None,
                )
                if tgt and tgt in specs:
                    layer_to_spec[ln] = specs[tgt]
        return layer_to_gid, layer_to_spec

    def _setup_gemma4_kv_sharing(
        self,
        target_attn_layer_names: set[str],
    ) -> None:
        """Wire draft layers to share KV with the target model.

        Each draft decoder layer is mapped to the last non-KV-shared
        target layer of the same attention type (sliding or full).
        """
        draft_config = self.speculative_config.draft_model_config.hf_config
        draft_text_config = draft_config.get_text_config()
        target_config = self.vllm_config.model_config.hf_config
        target_text_config = target_config.get_text_config()
        target_layer_types = getattr(target_text_config, "layer_types", [])

        if not (hasattr(self.model, "model") and hasattr(self.model.model, "layers")):
            return

        target_num_kv_shared = getattr(target_text_config, "num_kv_shared_layers", 0)
        num_non_shared = len(target_layer_types) - target_num_kv_shared
        type_to_target_indices: dict[str, list[int]] = defaultdict(list)
        for idx, lt in enumerate(target_layer_types[:num_non_shared]):
            type_to_target_indices[lt].append(idx)

        target_prefix = "model.layers"
        for name in target_attn_layer_names:
            if ".layers." in name:
                target_prefix = name.split(".layers.")[0] + ".layers"
                break

        draft_layer_types = getattr(draft_text_config, "layer_types", [])
        for draft_idx, layer in enumerate(self.model.model.layers):
            if not hasattr(layer, "self_attn"):
                continue
            attn = getattr(layer.self_attn, "attn", None)
            if attn is None:
                continue

            draft_layer_type = (
                draft_layer_types[draft_idx]
                if draft_idx < len(draft_layer_types)
                else "full_attention"
            )
            candidates = type_to_target_indices.get(draft_layer_type, [])
            if not candidates:
                logger.warning(
                    "No target layer of type '%s' for draft layer %d",
                    draft_layer_type,
                    draft_idx,
                )
                continue

            target_idx = candidates[-1]
            target_layer_name = f"{target_prefix}.{target_idx}.self_attn.attn"
            attn.kv_sharing_target_layer_name = target_layer_name
            logger.info(
                "Gemma4 MTP: draft layer %d (%s) -> %s",
                draft_idx,
                draft_layer_type,
                target_layer_name,
            )
