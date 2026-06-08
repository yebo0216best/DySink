"""
Memory-augmented Causal Inference Pipeline for LongLive with PE-Core Vision Encoder

This pipeline extends the base CausalInferencePipeline with a memory bank
mechanism that uses PE-Core VisionTransformer for visual feature extraction.

Key Features:
1. Auto-regressive block generation with memory retrieval
2. **Block-level encoding** (matching training pipeline):
   - Each block = num_frame_per_block latent frames
   - Decoded to num_frame_per_block * 4 = 12 real frames via VAE
   - Visual feature: 12 real frames encoded by PE-Core VisionTransformer
   - Visual-only retrieval (no text encoder)
3. Block-level KV cache storage and retrieval
4. Visual-only cosine similarity for retrieval
5. Bridges train-test gap: same retrieval granularity as training
6. Stack-like dedup: if new block similarity > threshold, skip it (keep older, higher-quality blocks)

SPDX-License-Identifier: Apache-2.0
"""

from typing import List, Optional, Tuple, Dict, Any, Union
import torch
import torch.nn.functional as F
from einops import rearrange

from pipeline.causal_inference import CausalInferencePipeline
from utils.memory_bank import MemoryBank, extract_kv_cache_for_frame
from utils.memory_bank_eb import PECoreMemoryEncoder
from utils.memory import gpu, get_cuda_free_memory_gb, log_gpu_memory

import torch.distributed as dist


class MemoryCausalInferencePipelineEB(CausalInferencePipeline):
    """
    Causal Inference Pipeline with Memory Bank using PE-Core Vision Encoder.
    
    This extends the base pipeline by:
    1. Decoding each generated block to pixel space
    2. Encoding entire block (num_frame_per_block*4 real frames) as single visual feature
    3. Storing block feature + block KV cache in memory bank
    4. Retrieving similar blocks' KV cache for future generation
    
    Block-level retrieval (matching training pipeline):
    - Retrieval unit = 1 block = num_frame_per_block latent frames
    - Each block → num_frame_per_block * 4 real frames → single visual feature via PE-Core
    - Visual-only cosine similarity for retrieval
    """
    
    def __init__(
        self,
        args,
        device,
        generator=None,
        text_encoder=None,
        vae=None,
        memory_encoder_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Memory-augmented Causal Inference Pipeline with PE-Core.
        
        Args:
            args: Configuration arguments
            device: Target device
            generator: Optional pre-initialized generator
            text_encoder: Optional pre-initialized text encoder
            vae: Optional pre-initialized VAE
            memory_encoder_config: Configuration for the PE-Core memory encoder
                - pe_config: PE-Core model config name (e.g. "PE-Core-B16-224")
                - checkpoint_path: Path to PE-Core checkpoint file
        """
        super().__init__(args, device, generator, text_encoder, vae)
        
        # Memory bank configuration
        self.memory_top_k = getattr(args, "memory_top_k", 3)
        self.memory_max_entries = getattr(args, "memory_max_entries", None)
        
        # Memory deduplication threshold
        # Frames with similarity > threshold will be deduplicated
        self.memory_dedup_threshold = getattr(args, "memory_dedup_threshold", 0.95)
        
        # Initialize memory encoder (lazy initialization)
        self._memory_encoder = None
        self._memory_encoder_config = memory_encoder_config or {}
        
        # Memory bank will be initialized per-video (per-sample)
        self.memory_bank = None
        
        # Frame sequence length for KV cache extraction
        self.frame_seq_length = 1560
        
    @property
    def memory_encoder(self) -> PECoreMemoryEncoder:
        """Lazy initialization of PE-Core memory encoder (visual-only)."""
        if self._memory_encoder is None:
            self._memory_encoder = PECoreMemoryEncoder(
                pe_config=self._memory_encoder_config.get("pe_config", "PE-Core-B16-224"),
                checkpoint_path=self._memory_encoder_config.get("checkpoint_path", None),
                device=self.generator.model.patch_embedding.weight.device,
                dtype=torch.float32,
            )
        return self._memory_encoder
    
    def _initialize_memory_bank(self, batch_size: int = 1):
        """Initialize fresh memory banks for a new video generation (one per sample)."""
        feature_dim = self.memory_encoder.feature_dim
        self.memory_bank = [
            MemoryBank(
                feature_dim=feature_dim,
                top_k=self.memory_top_k,
                max_entries=self.memory_max_entries,
                device=torch.device("cpu")  # Store on CPU for memory efficiency
            )
            for _ in range(batch_size)
        ]

    def _get_memory_bank(self, sample_idx: int = 0) -> Optional[MemoryBank]:
        """Get memory bank for a sample (handles list or single)."""
        if self.memory_bank is None:
            return None
        if isinstance(self.memory_bank, list):
            return self.memory_bank[sample_idx]
        return self.memory_bank

    def _memory_bank_entry_count(self) -> int:
        """Return the minimum memory-bank size across samples."""
        if self.memory_bank is None:
            return 0
        if isinstance(self.memory_bank, list):
            if len(self.memory_bank) == 0:
                return 0
            return min(len(bank) for bank in self.memory_bank)
        return len(self.memory_bank)

    def _memory_adjusted_local_attn_size(self, memory_entries: Optional[int] = None) -> int:
        """
        Temporarily expand local attention while memory has fewer than top-k blocks.

        The shortfall is counted in memory blocks and converted to latent frames,
        because each retrieved memory entry stores one generated block.
        """
        if self.local_attn_size == -1:
            return -1

        entries = self._memory_bank_entry_count() if memory_entries is None else memory_entries
        missing_blocks = max(0, int(self.memory_top_k) - int(entries))
        return int(self.local_attn_size) + missing_blocks * int(self.num_frame_per_block)

    def _log_temp_attention_window(
        self,
        *,
        enabled: bool,
        effective_local_attn: int,
        memory_entries: Optional[int] = None,
    ) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        base_local_attn = int(self.local_attn_size)
        entries = (
            self._memory_bank_entry_count()
            if memory_entries is None and enabled
            else (0 if memory_entries is None else int(memory_entries))
        )
        missing_blocks = (
            max(0, int(self.memory_top_k) - entries)
            if enabled and base_local_attn != -1
            else 0
        )
        key = (bool(enabled), int(effective_local_attn), missing_blocks)
        if getattr(self, "_last_temp_attn_log_key", None) == key:
            return
        self._last_temp_attn_log_key = key

        print(
            "[TempAttn][Inference] "
            f"enabled={enabled}, per_forward=True, "
            f"base_local_attn={base_local_attn}, "
            f"effective_local_attn={int(effective_local_attn)}, "
            f"memory_entries={entries}, missing_blocks={missing_blocks}, "
            f"top_k={int(self.memory_top_k)}, "
            f"block_size={int(self.num_frame_per_block)}"
        )

    def _apply_memory_adjusted_attention_window(self, memory_entries: Optional[int] = None) -> int:
        """Return the per-forward memory-shortfall local attention window."""
        effective_local_attn = self._memory_adjusted_local_attn_size(memory_entries)
        self._log_temp_attention_window(
            enabled=True,
            effective_local_attn=effective_local_attn,
            memory_entries=memory_entries,
        )
        return effective_local_attn

    def _slice_kv_cache_for_sample(
        self,
        kv_cache: List[Dict[str, torch.Tensor]],
        sample_idx: int
    ) -> List[Dict[str, torch.Tensor]]:
        """Slice KV cache to a single sample (keeps batch dim = 1)."""
        sliced_cache = []
        for block_cache in kv_cache:
            block_sliced = {}
            for key, value in block_cache.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    if value.shape[0] == 1:
                        block_sliced[key] = value.clone()
                    else:
                        block_sliced[key] = value[sample_idx:sample_idx + 1].clone()
                else:
                    block_sliced[key] = value
            sliced_cache.append(block_sliced)
        return sliced_cache

    def _merge_retrieval_kv_cache_by_sample(
        self,
        per_sample_kv: List[Optional[List[Dict[str, torch.Tensor]]]],
        batch_size: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Merge per-sample retrieval KV cache into a per-block KV cache.

        Two output formats are produced depending on whether the retrieval is
        uniform across the batch:

        * **Batched fast path** – when every sample retrieved the *same* number
          of entries with identical per-entry sequence lengths, the K/V tensors
          are concatenated along the batch dimension exactly as before. The
          downstream model code receives ``{'k', 'v', 'entry_seq_lens',
          'entry_frame_indices'}`` and runs a single batched attention call.

        * **Per-sample correctness path** – when samples retrieved a different
          number of entries (e.g. ``top_k > 1`` and only some samples saturated
          their bank), batching along ``dim=0`` is impossible because the
          sequence dimension differs. Instead we keep one tensor per sample and
          mark the dict with ``'per_sample_k'`` / ``'per_sample_v'`` lists plus
          per-sample ``entry_seq_lens`` / ``entry_frame_indices``. The model
          self-attention detects this format and loops per sample.
        """
        if all(kv is None for kv in per_sample_kv):
            return None

        ref_kv = next(kv for kv in per_sample_kv if kv is not None)
        num_blocks = len(ref_kv)

        merged_cache: List[Dict[str, Any]] = []
        for block_idx in range(num_blocks):
            per_sample_k: List[Optional[torch.Tensor]] = []
            per_sample_v: List[Optional[torch.Tensor]] = []
            per_sample_entry_seq_lens: List[List[int]] = []
            per_sample_entry_frame_indices: List[List[int]] = []

            for kv in per_sample_kv:
                if kv is None:
                    per_sample_k.append(None)
                    per_sample_v.append(None)
                    per_sample_entry_seq_lens.append([])
                    per_sample_entry_frame_indices.append([])
                    continue

                block_cache = kv[block_idx]
                k_t = block_cache['k']
                v_t = block_cache['v']
                if k_t.shape[0] != 1:
                    k_t = k_t[:1]
                if v_t.shape[0] != 1:
                    v_t = v_t[:1]
                per_sample_k.append(k_t)
                per_sample_v.append(v_t)
                # entry_seq_lens / entry_frame_indices may be absent when only
                # a single entry was retrieved — synthesise a single-entry list
                # from tensor shape so per-sample gating still has metadata.
                esl = block_cache.get('entry_seq_lens', [int(k_t.shape[1])])
                efi = block_cache.get('entry_frame_indices', [-1])
                per_sample_entry_seq_lens.append(list(esl))
                per_sample_entry_frame_indices.append(list(efi))

            # Fast path requires every sample to have a tensor of identical
            # sequence length AND identical per-entry layout.
            all_present = all(t is not None for t in per_sample_k)
            shapes = [t.shape[1] for t in per_sample_k if t is not None]
            same_shape = all_present and len(set(shapes)) == 1
            same_entries = (
                all_present
                and len({tuple(e) for e in per_sample_entry_seq_lens}) == 1
            )

            if same_shape and same_entries:
                merged_block: Dict[str, Any] = {
                    'k': torch.cat(per_sample_k, dim=0),
                    'v': torch.cat(per_sample_v, dim=0),
                }
                ref_esl = per_sample_entry_seq_lens[0]
                ref_efi = per_sample_entry_frame_indices[0]
                if len(ref_esl) > 1:
                    merged_block['entry_seq_lens'] = ref_esl
                    merged_block['entry_frame_indices'] = ref_efi
                merged_cache.append(merged_block)
            else:
                merged_cache.append({
                    'per_sample_k': per_sample_k,
                    'per_sample_v': per_sample_v,
                    'per_sample_entry_seq_lens': per_sample_entry_seq_lens,
                    'per_sample_entry_frame_indices': per_sample_entry_frame_indices,
                })

        return merged_cache
    
    def _decode_latent_frame(
        self,
        latent: torch.Tensor,
        frame_idx: int
    ) -> torch.Tensor:
        """
        Decode a single latent frame to pixel space.
        
        Args:
            latent: Latent tensor for frames to decode, shape [B, num_frames, C, H, W]
            frame_idx: Starting frame index
            
        Returns:
            Decoded frames, shape [B, 4*num_latent_frames, C, H, W] with values in [0, 1]
        """
        # Decode using VAE
        # Note: VAE has 4x temporal compression, so 1 latent frame -> 4 pixel frames
        video = self.vae.decode_to_pixel(latent, use_cache=False)
        
        # Normalize to [0, 1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        
        return video
    
    def _encode_block_features(
        self,
        decoded_frames: torch.Tensor,
        text_prompt: Optional[str] = None
    ) -> torch.Tensor:
        """
        Encode entire decoded block to a single visual feature using PE-Core.
        
        This encodes ALL decoded frames from a block (num_frame_per_block * 4 = 12 real
        frames) into a single block-level visual feature via PE-Core VisionTransformer.
        text_prompt parameter is kept for API compatibility but is ignored.
        
        Args:
            decoded_frames: Decoded video tensor for entire block,
                           shape [B, num_frame_per_block*4, C, H, W]
            text_prompt: Ignored (PE-Core is visual-only)
            
        Returns:
            Block-level visual feature, shape [B, feature_dim]
        """
        return self.memory_encoder.encode_frames(
            frames=decoded_frames,
            normalize=True
        )
    
    def _store_block_in_memory(
        self,
        block_start_frame: int,
        num_frames: int,
        feature: torch.Tensor,
        kv_cache: List[Dict[str, torch.Tensor]],
        sample_idx: int,
        latent: Optional[torch.Tensor] = None,
    ):
        """
        Store a block's feature and KV cache in the memory bank.
        
        Extracts KV cache for the entire block (num_frames latent frames)
        and stores it with the block feature. This matches the training pipeline's
        block-level storage.
        
        Args:
            block_start_frame: Starting frame index of this block
            num_frames: Number of latent frames in the block
            feature: Block-level visual feature
            kv_cache: Full KV cache from which to extract
            sample_idx: Sample index in the batch
            latent: Optional block latent frames used to recache this memory entry
        """
        # Extract KV cache for the entire block
        extracted_kv = extract_kv_cache_for_frame(
            kv_cache,
            frame_idx=block_start_frame,
            frame_seq_length=self.frame_seq_length,
            num_frames_to_extract=num_frames,
        )

        # Slice KV cache for the current sample
        extracted_kv = self._slice_kv_cache_for_sample(extracted_kv, sample_idx)
        latent_s = None
        if latent is not None:
            latent_s = latent[:1] if latent.shape[0] == 1 else latent[sample_idx:sample_idx + 1]

        # Store in memory bank (visual-only, no text feature)
        self.memory_bank[sample_idx].add_entry(
            frame_idx=block_start_frame,
            feature=feature.squeeze(0) if feature.dim() > 1 else feature,
            kv_cache=extracted_kv,
            text_feature=None,
            latent=latent_s,
        )

    def _build_empty_kv_cache(
        self,
        batch_size: int,
        kv_cache_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        return [
            {
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            }
            for _ in range(self.num_transformer_blocks)
        ]

    def _build_empty_crossattn_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        return [
            {
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False,
            }
            for _ in range(self.num_transformer_blocks)
        ]

    @staticmethod
    def _slice_conditional_dict_for_sample(
        conditional_dict: Dict[str, torch.Tensor],
        sample_idx: int,
    ) -> Dict[str, torch.Tensor]:
        sliced = {}
        for key, value in conditional_dict.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] > 1:
                sliced[key] = value[sample_idx:sample_idx + 1]
            else:
                sliced[key] = value
        return sliced

    @staticmethod
    def _copy_kv_cache_to_cpu(
        kv_cache: List[Dict[str, torch.Tensor]],
        valid_tokens: int,
    ) -> List[Dict[str, torch.Tensor]]:
        copied = []
        for block_cache in kv_cache:
            block_copied = {}
            for key, value in block_cache.items():
                if isinstance(value, torch.Tensor):
                    tensor = value[:, :valid_tokens] if key in ("k", "v") else value
                    block_copied[key] = tensor.detach().cpu().clone()
                else:
                    block_copied[key] = value
            copied.append(block_copied)
        return copied

    def _recache_memory_entry_for_new_prompt(
        self,
        entry,
        new_conditional_dict: Dict[str, torch.Tensor],
        sample_idx: int = 0,
        latent_override: Optional[torch.Tensor] = None,
        entry_frame_idx_override: int = 0,
    ) -> bool:
        latent = None if entry is None else entry.latent
        should_update_entry = entry is not None and latent is not None
        if latent is None:
            latent = latent_override
        if latent is None:
            return False

        target_device = self.generator.model.patch_embedding.weight.device
        target_dtype = (
            self.kv_cache1[0]["k"].dtype
            if self.kv_cache1 is not None
            else self.generator.model.patch_embedding.weight.dtype
        )
        if latent.dim() == 4:
            latent = latent.unsqueeze(0)
        latent = latent.to(device=target_device, dtype=target_dtype)

        batch_size, num_frames = latent.shape[:2]
        num_tokens = num_frames * self.frame_seq_length
        entry_frame_idx = (
            int(entry.frame_idx) if entry is not None else int(entry_frame_idx_override)
        )
        entry_start_token = entry_frame_idx * self.frame_seq_length

        temp_kv_cache = self._build_empty_kv_cache(
            batch_size=batch_size,
            kv_cache_size=num_tokens,
            dtype=target_dtype,
            device=target_device,
        )
        for block_cache in temp_kv_cache:
            block_cache["global_end_index"].fill_(entry_start_token)
            block_cache["local_end_index"].zero_()

        temp_crossattn_cache = self._build_empty_crossattn_cache(
            batch_size=batch_size,
            dtype=target_dtype,
            device=target_device,
        )
        conditional_dict = (
            self._slice_conditional_dict_for_sample(new_conditional_dict, sample_idx)
            if batch_size == 1
            else new_conditional_dict
        )
        local_attn_size = -1 if self.local_attn_size == -1 else num_frames
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=target_device,
            num_frames=num_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=local_attn_size,
        )
        context_timestep = torch.ones(
            [batch_size, num_frames],
            device=target_device,
            dtype=torch.int64,
        ) * self.args.context_noise

        self.generator.model.block_mask = block_mask
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=latent,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=temp_kv_cache,
                crossattn_cache=temp_crossattn_cache,
                current_start=entry_start_token,
                sink_recache_after_switch=True,
                local_attn_size=local_attn_size,
            )

        if should_update_entry:
            entry.kv_cache = self._copy_kv_cache_to_cpu(temp_kv_cache, num_tokens)
            return True
        return False

    def _memory_entries_for_recache(self):
        if self.memory_bank is None:
            return []

        banks = self.memory_bank if isinstance(self.memory_bank, list) else [self.memory_bank]
        entries = []
        for sample_idx, bank in enumerate(banks):
            for entry in bank.entries:
                entries.append((sample_idx, entry))
        return entries

    def _make_dummy_recache_latent(
        self,
        latent: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return a single-sample latent block for FSDP recache alignment."""
        if latent is not None:
            if latent.dim() == 4:
                latent = latent.unsqueeze(0)
            if latent.shape[1] > 0:
                block_frames = min(self.num_frame_per_block, latent.shape[1])
                dummy = latent[:1, -block_frames:].detach()
                return dummy

        for _, entry in self._memory_entries_for_recache():
            if entry.latent is not None:
                dummy = entry.latent
                if dummy.dim() == 4:
                    dummy = dummy.unsqueeze(0)
                return dummy[:1].detach()
        return None

    def _recache_memory_bank_for_new_prompt(
        self,
        new_conditional_dict: Dict[str, torch.Tensor],
        dummy_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int]:
        """
        Recompute stored memory-bank KV blocks under a new prompt.

        The visual features remain unchanged; only each entry's stored KV cache
        is replaced with the result of running its stored latent block through
        the generator with ``new_conditional_dict``.
        """
        if self.memory_bank is None:
            return 0, 0

        local_entries = self._memory_entries_for_recache()
        local_count = len(local_entries)
        max_count = local_count
        if dist.is_initialized():
            device = self.generator.model.patch_embedding.weight.device
            count_tensor = torch.tensor([local_count], device=device, dtype=torch.long)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.MAX)
            max_count = int(count_tensor.item())
        self._last_memory_recache_aligned_steps = max_count

        if max_count == 0:
            return 0, 0

        dummy_latent = self._make_dummy_recache_latent(dummy_latent)
        needs_dummy_forward = (
            local_count < max_count
            or any(entry.latent is None for _, entry in local_entries)
        )
        if dummy_latent is None and needs_dummy_forward:
            raise RuntimeError(
                "Need a dummy latent to keep FSDP recache forwards aligned across ranks, "
                "but no generated output or memory latent is available."
            )

        recached_entries = 0
        for entry_idx in range(max_count):
            if entry_idx < local_count:
                sample_idx, entry = local_entries[entry_idx]
                latent_override = dummy_latent if entry.latent is None else None
            else:
                sample_idx, entry = 0, None
                latent_override = dummy_latent

            if self._recache_memory_entry_for_new_prompt(
                entry,
                new_conditional_dict,
                sample_idx=sample_idx,
                latent_override=latent_override,
                entry_frame_idx_override=0,
            ):
                recached_entries += 1
        return recached_entries, local_count
    
    def _retrieve_kv_cache(
        self,
        query_feature: torch.Tensor,
        current_frame_idx: int,
        sample_idx: int,
        return_info: bool = False,
        text_feature: Optional[torch.Tensor] = None,
        return_score_breakdown: bool = False,
        query_features: Optional[List[torch.Tensor]] = None,
    ):
        """
        Retrieve KV cache from memory bank based on visual similarity (PE-Core).

        Supports **multi-query** retrieval via *query_features*.  When a list
        of recent block features is provided, similarity is averaged across
        queries before ranking.

        Args:
            query_feature: Single query visual feature (fallback).
            current_frame_idx: Current frame index being generated.
            return_info: If True, also return retrieved frame indices and similarities.
            text_feature: Ignored (PE-Core is visual-only, kept for API compatibility).
            return_score_breakdown: If True, also return visual / tt score breakdown.
            query_features: Optional list of recent block features for multi-query.

        Returns:
            Retrieved KV cache or None if no retrievable entries.
        """
        if self.memory_bank is None or len(self.memory_bank[sample_idx]) == 0:
            if return_info:
                if return_score_breakdown:
                    return None, [], [], [], []
                return None, [], []
            return None

        sink_size = getattr(self.args.model_kwargs, "sink_size", 0)
        effective_local_attn_size = self._memory_adjusted_local_attn_size(
            len(self.memory_bank[sample_idx])
        )
        local_attn_size = effective_local_attn_size if effective_local_attn_size > 0 else 0

        exclude_frame_indices = set()
        for frame_idx in range(sink_size):
            exclude_frame_indices.add(frame_idx)
        if local_attn_size > 0:
            sliding_window_start = max(0, current_frame_idx - local_attn_size)
            for frame_idx in range(sliding_window_start, current_frame_idx):
                exclude_frame_indices.add(frame_idx)
        exclude_frame_indices = list(exclude_frame_indices)

        return self.memory_bank[sample_idx].get_retrieval_kv_cache(
            query_feature=query_feature,
            target_device=self.generator.model.patch_embedding.weight.device,
            target_dtype=torch.bfloat16,
            exclude_frame_indices=exclude_frame_indices,
            return_info=return_info,
            return_score_breakdown=return_score_breakdown,
            query_features=query_features,
        )
    
    def inference_with_memory(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        use_memory_retrieval: bool = True
    ) -> torch.Tensor:
        """
        Perform inference with memory-augmented KV cache retrieval.
        
        This method generates video frame-by-frame (or chunk-by-chunk) while:
        1. Decoding each generated frame to pixel space
        2. Encoding visual features using PE-Core VisionTransformer
        3. Storing features and KV cache in the memory bank
        4. Retrieving similar historical KV caches using visual-only cosine similarity
        
        Args:
            noise: Input noise tensor, shape [B, num_output_frames, C, H, W]
            text_prompts: List of text prompts
            return_latents: Whether to return latents
            profile: Whether to profile the inference
            low_memory: Whether to use low memory mode
            use_memory_retrieval: Whether to use memory retrieval
            
        Returns:
            Generated video tensor, shape [B, T, C, H, W] in [0, 1] range
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block
        
        # Get text prompt(s) for retrieval and frame encoding
        current_text_prompt = text_prompts[0] if isinstance(text_prompts, list) else text_prompts

        if use_memory_retrieval:
            print(f"[MemoryEB] Visual-only retrieval (PE-Core), batch size: {batch_size}")
            print(f"[MemoryEB] Block-level retrieval (block size: {self.num_frame_per_block} latent frames = {self.num_frame_per_block * 4} real frames)")
        
        # Get text embeddings
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        
        if low_memory:
            from utils.memory import move_model_to_device_with_memory_preservation
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation
            )
        
        # Output device
        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )
        
        # Initialize memory bank for this video
        if use_memory_retrieval:
            self._initialize_memory_bank(batch_size)
        
        # Initialize KV cache
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            max_local_attn_cfg = (
                self._memory_adjusted_local_attn_size(memory_entries=0)
                if use_memory_retrieval
                else int(local_attn_cfg)
            )
            kv_cache_size = max_local_attn_cfg * self.frame_seq_length
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
        
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )
        
        current_start_frame = 0
        if use_memory_retrieval:
            effective_local_attn_size = self._apply_memory_adjusted_attention_window(memory_entries=0)
        else:
            effective_local_attn_size = int(self.local_attn_size)
            self.generator.model.local_attn_size = self.local_attn_size
            self._set_all_modules_max_attention_size(self.local_attn_size)
            self._log_temp_attention_window(
                enabled=False,
                effective_local_attn=effective_local_attn_size,
                memory_entries=0,
            )
        
        # Recent block features for multi-query retrieval (matching training pipeline).
        # Stores up to (local_attn_size // num_frame_per_block) features.
        from collections import deque
        local_query_blocks = max(1, self.local_attn_size // self.num_frame_per_block)
        recent_block_features: deque = deque(maxlen=local_query_blocks)
        prev_block_feature = None  # kept for per-sample fallback

        # Store retrieval KV cache to use during generation
        retrieval_kv_cache = None
        
        # Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        
        for block_idx, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames]

            if use_memory_retrieval:
                effective_local_attn_size = self._apply_memory_adjusted_attention_window()
            
            # Reset retrieval KV cache for each block
            retrieval_kv_cache = None
            retrieved_frame_ids = [[] for _ in range(batch_size)]
            retrieved_similarities = [[] for _ in range(batch_size)]
            retrieved_mm_scores = [[] for _ in range(batch_size)]
            retrieved_tt_scores = [[] for _ in range(batch_size)]
            
            # Retrieve KV cache using multi-query (all blocks in local window).
            if use_memory_retrieval and len(recent_block_features) > 0:
                per_sample_kv = []
                for sample_idx in range(batch_size):
                    if len(self.memory_bank[sample_idx]) > 0:
                        # Build per-sample query list from recent block features
                        multi_q = [bf[sample_idx] for bf in recent_block_features]
                        kv_cache_s, block_ids_s, sims_s, mm_s, tt_s = self._retrieve_kv_cache(
                            query_feature=multi_q[-1],
                            current_frame_idx=current_start_frame,
                            sample_idx=sample_idx,
                            return_info=True,
                            return_score_breakdown=True,
                            query_features=multi_q if len(multi_q) > 1 else None,
                        )
                        per_sample_kv.append(kv_cache_s)
                        retrieved_frame_ids[sample_idx] = block_ids_s
                        retrieved_similarities[sample_idx] = sims_s
                        retrieved_mm_scores[sample_idx] = mm_s
                        retrieved_tt_scores[sample_idx] = tt_s
                    else:
                        per_sample_kv.append(None)
                retrieval_kv_cache = self._merge_retrieval_kv_cache_by_sample(
                    per_sample_kv, batch_size
                )
            
            # Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64
                ) * current_timestep
                
                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        retrieval_kv_cache=retrieval_kv_cache,
                        local_attn_size=effective_local_attn_size,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames],
                            device=noise.device,
                            dtype=torch.long
                        )
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        retrieval_kv_cache=retrieval_kv_cache,
                        local_attn_size=effective_local_attn_size,
                    )
            
            # Store output
            output[:, current_start_frame:current_start_frame + current_num_frames] = \
                denoised_pred.to(output.device)
            
            # Rerun with clean context to update KV cache
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                local_attn_size=effective_local_attn_size,
            )
            
            # Memory bank operations
            if use_memory_retrieval:
                # Decode generated frames to pixel space
                decoded_frames = self._decode_latent_frame(
                    denoised_pred.to(noise.device),
                    frame_idx=current_start_frame
                )
                
                # Log retrieval status
                # Note: memory bank stores block_start_frame as frame_idx,
                # convert to block index for clearer logging
                if block_idx > 0 and retrieval_kv_cache is not None:
                    for sample_idx in range(batch_size):
                        if retrieved_frame_ids[sample_idx]:
                            sim_str = ", ".join([f"{s:.3f}" for s in retrieved_similarities[sample_idx]])
                            visual_str = ", ".join([f"{s:.3f}" for s in retrieved_mm_scores[sample_idx]])
                            retrieved_block_indices = [fid // self.num_frame_per_block for fid in retrieved_frame_ids[sample_idx]]
                            print(f"[MemoryEB] Block {block_idx} (frame {current_start_frame}) Sample {sample_idx}: "
                                  f"Retrieved block KV cache (visual-only). "
                                  f"Retrieved blocks: {retrieved_block_indices}, "
                                  f"Visual: [{visual_str}]")
                        else:
                            print(f"[MemoryEB] Block {block_idx} Sample {sample_idx}: No retrievable block found "
                                  f"(memory bank size: {len(self.memory_bank[sample_idx])})")
                elif block_idx > 0:
                    for sample_idx in range(batch_size):
                        print(f"[MemoryEB] Block {block_idx} Sample {sample_idx}: No retrievable block found "
                              f"(memory bank size: {len(self.memory_bank[sample_idx])})")
                
                # Print sink anomaly gate stats for this block
                gate_summary = self.generator.model.get_gate_stats_summary()
                if gate_summary is not None:
                    gated_layers = [l for l, c, g, r in gate_summary["per_layer"] if g > 0]
                    entry_info = ""
                    if gate_summary["entries_total"] > 0:
                        entry_info = (f", entries {gate_summary['entries_gated']}/"
                                      f"{gate_summary['entries_total']} gated")
                    print(f"[Rectify] Block {block_idx}: "
                          f"{gate_summary['total_gated']}/{gate_summary['total_calls']} gated "
                          f"({gate_summary['gated_pct']:.1f}%), "
                          f"mean anomaly ratio={gate_summary['mean_anomaly_ratio']:.3f}"
                          + entry_info
                          + (f", gated layers={gated_layers}" if gated_layers else ""))
                    if gate_summary.get("per_entry"):
                        nfpb = self.num_frame_per_block
                        parts = []
                        for fid, info in gate_summary["per_entry"].items():
                            bid = fid // nfpb
                            parts.append(f"mem_blk{bid}(gated {info['gated_layers']}/"
                                         f"{info['total_layers']} layers, "
                                         f"ratio={info['mean_ratio']:.3f})")
                        print(f"[Rectify] Block {block_idx} per-entry: {', '.join(parts)}")
                
                # ============================================================
                # Block-level encoding and storage (matching training pipeline)
                # Encode entire block as single visual feature via PE-Core
                # ============================================================
                block_feature = self._encode_block_features(decoded_frames)
                
                # Stack-like dedup: check similarity, skip if too similar (keep old blocks)
                # Early blocks tend to have higher quality, so we prefer keeping them.
                # Visual-only cosine similarity (PE-Core, no text features).
                skipped = [False for _ in range(batch_size)]
                for sample_idx in range(batch_size):
                    block_feature_s = block_feature[sample_idx]
                    
                    # Check max visual similarity with existing entries
                    max_sim = self.memory_bank[sample_idx].check_similarity(
                        block_feature_s,
                    )
                    
                    if len(self.memory_bank[sample_idx]) > 0 and max_sim > self.memory_dedup_threshold:
                        # New block is too similar — skip (keep old, higher-quality entries)
                        skipped[sample_idx] = True
                        print(f"[MemoryEB] Block {block_idx} Sample {sample_idx}: "
                              f"Skipped (combined_sim={max_sim:.4f} > threshold={self.memory_dedup_threshold}). "
                              f"Bank size: {len(self.memory_bank[sample_idx])}")
                    else:
                        # Store block feature + block KV cache
                        self._store_block_in_memory(
                            block_start_frame=current_start_frame,
                            num_frames=current_num_frames,
                            feature=block_feature_s,
                            kv_cache=self.kv_cache1,
                            sample_idx=sample_idx,
                            latent=denoised_pred,
                        )
                
                # Log memory bank status
                if block_idx % 10 == 0 or any(s for s in skipped):
                    for sample_idx in range(batch_size):
                        if not skipped[sample_idx]:
                            print(f"[MemoryEB] Block {block_idx} Sample {sample_idx}: Memory bank has "
                                  f"{len(self.memory_bank[sample_idx])} block entries")
                
                # Append to recent block features for multi-query retrieval
                recent_block_features.append(block_feature)
                prev_block_feature = block_feature
            
            # Update frame index
            current_start_frame += current_num_frames
        
        # Final decode
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        
        # Clear memory bank after generation
        if use_memory_retrieval:
            if self.memory_bank is not None:
                for bank in self.memory_bank:
                    bank.clear()
        
        if return_latents:
            return video, output.to(noise.device)
        else:
            return video
    
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Override base inference to use memory-augmented version with PE-Core.
        
        Set use_memory_retrieval based on args configuration.
        """
        use_memory = getattr(self.args, "use_memory_retrieval", True)
        
        return self.inference_with_memory(
            noise=noise,
            text_prompts=text_prompts,
            return_latents=return_latents,
            profile=profile,
            low_memory=low_memory,
            use_memory_retrieval=use_memory
        )
