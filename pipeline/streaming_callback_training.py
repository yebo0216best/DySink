# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0

"""
Streaming Callback Training Pipeline — Bridging the Train-Test Gap

This pipeline extends StreamingSwitchTrainingPipeline to support three-segment
prompt training with a callback mechanism AND real-time memory retrieval,
matching the inference pipeline's behaviour to bridge the train-test gap.

Key Features:
1. Three-segment prompt support: prompt1 -> prompt2 -> prompt3 (callback to prompt1)
2. Two switch points for transitioning between prompts
3. **Real-time memory retrieval during training** (same as inference):
   - After each block: encode block feature + extract KV cache → store in memory bank
   - Before each block: retrieve similar blocks from memory bank → inject KV cache
   - Generator receives `retrieval_kv_cache` during training, just like inference
4. Block-level encoding for memory retrieval:
   - Each block = num_frame_per_block latent frames
   - Decoded to num_frame_per_block * 4 = 12 real frames via VAE
   - Visual feature: 12 real frames encoded into a single feature via PE-Core VisionTransformer
   - Visual-only cosine similarity for retrieval (no text encoder)
5. Stack-like dedup: if new block similarity > threshold, skip it (keep older, higher-quality blocks)
7. **On prompt switch**: memory bank entries' KV caches are recached under the
   new prompt, and the sink/local sliding-window cache is refreshed too.

This bridges the train-test gap: the model sees retrieval KV cache during training,
so it learns to properly utilise retrieved context at inference time.
"""

from collections import deque

from pipeline.streaming_switch_training import StreamingSwitchTrainingPipeline
from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.distributed as dist
from utils.debug_option import DEBUG, LOG_GPU_MEMORY
from utils.memory import log_gpu_memory
from utils.memory_bank import extract_kv_cache_for_frame


class StreamingCallbackTrainingPipeline(StreamingSwitchTrainingPipeline):
    """Training pipeline supporting three-segment prompt switching with callback.

    Use case: In a single roll-out:
    - First segment uses prompt-1 (initial scene/topic)
    - At first switch frame, transition to prompt-2 (different scene/topic)  
    - At callback switch frame, transition to prompt-3 (callback referencing prompt-1)
    
    **Bridging train-test gap**: During training, this pipeline performs the SAME
    memory retrieval as the inference pipeline:
    - Encode block features + store KV cache in memory bank (after each block)
    - Retrieve similar blocks' KV cache from memory bank (before each block)
    - Pass retrieval_kv_cache to generator (same parameter as inference)
    
    Memory encoding uses block-level approach:
    - Retrieval unit = 1 block = num_frame_per_block latent frames
    - Each block is decoded to num_frame_per_block * 4 = 12 real frames
    - Visual feature: 12 frames → single feature via PE-Core VisionTransformer
    - Retrieval: visual-only cosine similarity (no text encoder)
    
    Stack-like deduplication:
    - If new block max_sim > threshold with existing entries → skip (keep old)
    - Early-generated blocks tend to have higher quality, so we prefer keeping them
    """

    def __init__(
        self,
        *args,
        use_memory_in_training: bool = False,
        memory_top_k: int = 3,
        memory_dedup_threshold: float = 0.95,
        memory_encoder_config: Optional[Dict[str, Any]] = None,
        vae=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Keep retrieval exclusion policy aligned with inference:
        # exclude sink frames + local sliding window frames.
        self.sink_size = int(kwargs.get("sink_size", 0))
        
        # Memory-aware training settings (matching inference pipeline)
        self.use_memory_in_training = use_memory_in_training
        self.memory_top_k = memory_top_k
        self.memory_dedup_threshold = memory_dedup_threshold
        if self.use_memory_in_training and self.local_attn_size != -1:
            max_memory_adjusted_frames = (
                int(self.local_attn_size)
                + int(self.memory_top_k) * int(self.num_frame_per_block)
            )
            self.kv_cache_size = max(
                self.kv_cache_size,
                max_memory_adjusted_frames * self.frame_seq_length,
            )
        
        # Memory encoder configuration (PE-Core Vision Encoder)
        self._memory_encoder = None
        self._memory_encoder_config = memory_encoder_config or {}
        
        # VAE for decoding latents to pixel space (needed for memory encoding)
        self.vae = vae
        
        # Memory bank for storing block features + KV cache during training
        self._training_memory_bank = None
        
        # Recent block features for multi-query retrieval.
        # Stores up to (local_attn_size // num_frame_per_block) features so that
        # all blocks in the local attention window contribute to the query.
        self._local_query_blocks = max(1, self.local_attn_size // self.num_frame_per_block)
        self._recent_block_features: deque = deque(maxlen=self._local_query_blocks)
        
        # Frame sequence length for KV cache extraction (matching inference pipeline)
        self.frame_seq_length = 1560
        
        # Persistent segment tracking across chunk calls.
        # Updated when a switch/callback occurs inside generate_chunk_with_cache
        # and also settable from the outer training loop via set_current_segment().
        self.current_segment = 0  # 0: prompt1, 1: prompt2, 2: prompt3 (callback)
        
        # Current prompts per segment (used as guard to check segment presence)
        self._current_prompts = {}  # segment_idx -> prompt string
        self._last_recache_latent = None
        
        if use_memory_in_training and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingCallbackTraining] Memory-aware training with RETRIEVAL (bridging train-test gap)")
            print(f"[StreamingCallbackTraining] Block size: {self.num_frame_per_block} latent frames "
                  f"= {self.num_frame_per_block * 4} real frames per block feature")
            print(f"[StreamingCallbackTraining] Memory: top_k={memory_top_k}, "
                  f"dedup={memory_dedup_threshold}")
    
    @property
    def memory_encoder(self):
        """Lazy initialization of PE-Core memory encoder (visual-only)."""
        if self._memory_encoder is None and self.use_memory_in_training:
            from utils.memory_bank_eb import PECoreMemoryEncoder
            self._memory_encoder = PECoreMemoryEncoder(
                pe_config=self._memory_encoder_config.get("pe_config", "PE-Core-B16-224"),
                checkpoint_path=self._memory_encoder_config.get("checkpoint_path", None),
                device=self.generator.model.patch_embedding.weight.device,
                dtype=torch.float32,
            )
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[StreamingCallbackTraining] Initialized PE-Core memory encoder: "
                      f"{self._memory_encoder_config.get('pe_config', 'PE-Core-B16-224')}")
        return self._memory_encoder
    
    def _initialize_training_memory_bank(self):
        """Initialize memory bank for training."""
        if not self.use_memory_in_training:
            return
        
        from utils.memory_bank import MemoryBank
        feature_dim = self.memory_encoder.feature_dim
        self._training_memory_bank = MemoryBank(
            feature_dim=feature_dim,
            top_k=self.memory_top_k,
            max_entries=None,
            device=torch.device("cpu")
        )
        
    def set_current_segment(self, segment: int):
        """Set the current segment index from the outer training loop.

        This must be called before each chunk generation so that memory
        retrieval and storage use the correct text features when the
        prompt switch already happened in a previous chunk.
        """
        self.current_segment = segment

    def clear_kv_cache(self):
        """
        Override parent's clear_kv_cache to also reset memory bank state.
        Called by reset_state() at the start of each new training video.
        """
        super().clear_kv_cache()
        
        # Clear memory bank so it's freshly initialized for the next video
        if self._training_memory_bank is not None:
            self._training_memory_bank.clear()
            self._training_memory_bank = None
        self._recent_block_features.clear()
        self._last_recache_latent = None
        self.current_segment = 0

    def _training_memory_bank_entry_count(self) -> int:
        if self._training_memory_bank is None:
            return 0
        return len(self._training_memory_bank)

    def _memory_adjusted_local_attn_size(self, memory_entries: Optional[int] = None) -> int:
        """
        Temporarily expand local attention while memory has fewer than top-k blocks.

        The shortfall is counted in memory blocks and converted to latent frames,
        matching the block-level KV retrieval unit.
        """
        if self.local_attn_size == -1:
            return -1

        entries = (
            self._training_memory_bank_entry_count()
            if memory_entries is None
            else memory_entries
        )
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
            self._training_memory_bank_entry_count()
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
            "[TempAttn][Train] "
            f"enabled={enabled}, per_forward=True, "
            f"base_local_attn={base_local_attn}, "
            f"effective_local_attn={int(effective_local_attn)}, "
            f"memory_entries={entries}, missing_blocks={missing_blocks}, "
            f"top_k={int(self.memory_top_k)}, "
            f"block_size={int(self.num_frame_per_block)}"
        )

    def _apply_memory_adjusted_attention_window(self, memory_entries: Optional[int] = None) -> int:
        # Keep the effective attention window as a per-forward value.  Mutating
        # model attributes here breaks activation checkpoint recomputation when
        # the memory bank changes between forward and backward.
        effective_local_attn = self._memory_adjusted_local_attn_size(memory_entries)
        self._log_temp_attention_window(
            enabled=self.use_memory_in_training,
            effective_local_attn=effective_local_attn,
            memory_entries=memory_entries,
        )
        return effective_local_attn
    
    def _encode_block_features_for_training(
        self,
        latent_frames: torch.Tensor,
        block_start_idx: int,
        num_frames_in_block: int,
        text_prompt: Optional[str] = None
    ) -> torch.Tensor:
        """
        Encode a whole block of latent frames to a single visual feature using PE-Core.
        
        Instead of encoding each latent frame separately (1 latent → 4 real frames → feature),
        this method encodes an entire block at once:
            num_frame_per_block latent frames → num_frame_per_block * 4 real frames → single feature
        
        The block-level feature captures richer temporal context than per-frame features,
        leading to more coherent retrieval and matching results.
        
        Args:
            latent_frames: Latent frames from generator, shape [B, T, C, H, W]
            block_start_idx: Starting latent frame index of the block
            num_frames_in_block: Number of latent frames in this block
            text_prompt: Text prompt for joint encoding
            
        Returns:
            Block-level visual feature, shape [B, feature_dim]
        """
        if self.vae is None:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingCallbackTraining] VAE not available, skipping block encoding")
            return None
        
        # Decode the entire block of latent frames to pixel space
        block_end_idx = min(block_start_idx + num_frames_in_block, latent_frames.shape[1])
        latent_to_decode = latent_frames[:, block_start_idx:block_end_idx]
        
        with torch.no_grad():
            decoded_frames = self.vae.decode_to_pixel(latent_to_decode, use_cache=False)
            decoded_frames = (decoded_frames * 0.5 + 0.5).clamp(0, 1)
        
        # decoded_frames: [B, num_frames_in_block * 4, C, H, W]
        # Encode all decoded frames + text prompt as a single block-level feature
        features = self.memory_encoder.encode_frames_with_prompt(
            frames=decoded_frames,
            text_prompt=text_prompt,
            normalize=True
        )
        
        return features
    
    def set_current_prompts(self, prompts: Dict[int, str]):
        """
        Set the current prompts for each segment.
        
        Note: PE-Core is visual-only, so text features are NOT encoded.
        Text features dict is kept empty for compatibility.
        
        Args:
            prompts: Dict mapping segment_idx to prompt string
                     {0: prompt1, 1: prompt2, 2: prompt3}
        """
        self._current_prompts = prompts
        
        # PE-Core is visual-only: no text encoding needed
        self._text_features = {}
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingCallbackTraining] Set prompts for {len(prompts)} segments (visual-only retrieval)")

    def generate_chunk_with_cache(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        *,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        # First switch: prompt1 -> prompt2
        switch_frame_index: Optional[int] = None,
        switch_conditional_dict: Optional[dict] = None,
        switch_recache_frames: Optional[torch.Tensor] = None,
        # Callback switch: prompt2 -> prompt3
        callback_frame_index: Optional[int] = None,
        callback_conditional_dict: Optional[dict] = None,
        callback_recache_frames: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """
        Chunk generation method for three-segment callback training.

        Args:
            noise: noise of a single chunk [batch_size, chunk_frames, C, H, W]
            conditional_dict: initial conditional information (prompt 1)
            current_start_frame: start frame index of the chunk in the full sequence
            requires_grad: whether gradients are required
            switch_frame_index: first switch frame index (prompt1 -> prompt2)
            switch_conditional_dict: conditional info for prompt 2
            switch_recache_frames: frames used to recache during first switch
            callback_frame_index: callback switch frame index (prompt2 -> prompt3)
            callback_conditional_dict: conditional info for prompt 3 (callback)
            callback_recache_frames: frames used to recache during callback switch
            return_sim_step: whether to return simulation step info

        Returns:
            output: generated chunk [batch_size, chunk_frames, C, H, W]
            denoised_timestep_from: starting denoise timestep
            denoised_timestep_to: ending denoise timestep
        """

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Callback] generate_chunk_with_cache called")
            print(f"[SeqTrain-Callback] switch_frame_index={switch_frame_index}, "
                  f"callback_frame_index={callback_frame_index}")
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"SeqTrain-Callback: Before callback chunk generation", 
                          device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        # Keep callback pipeline logic active for all chunks so memory retrieval
        # starts from the beginning of the video. Callback switching is simply
        # skipped when callback info is not provided for this chunk.
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Callback] First switch at frame {switch_frame_index}, "
                  f"callback at frame {callback_frame_index}")
        
        batch_size, chunk_frames, num_channels, height, width = noise.shape
        assert chunk_frames % self.num_frame_per_block == 0
        num_blocks = chunk_frames // self.num_frame_per_block
        all_num_frames = [self.num_frame_per_block] * num_blocks

        # Prepare output
        output = torch.zeros_like(noise)
        
        # Randomly select denoising steps (synced across ranks)
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        
        # Determine the gradient-enabled range
        # For callback training, we enable gradients for the callback segment (after callback_frame_index)
        if not requires_grad:
            start_gradient_frame_index = chunk_frames  # Out of range: no gradients anywhere
        else:
            # Enable gradients after the callback switch for better learning of consistency
            # Use explicit None checks to handle frame index 0 correctly
            if callback_frame_index is not None:
                start_gradient_frame_index = callback_frame_index
            elif switch_frame_index is not None:
                start_gradient_frame_index = switch_frame_index
            else:
                start_gradient_frame_index = 0  # Default: enable gradients from the beginning
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Callback] start_gradient_frame_index={start_gradient_frame_index}")
        
        local_start_frame = 0

        def resolve_local_attn_size() -> int:
            if self.use_memory_in_training:
                memory_entries = self._training_memory_bank_entry_count()
                effective_local_attn = self._memory_adjusted_local_attn_size(memory_entries)
                self._log_temp_attention_window(
                    enabled=True,
                    effective_local_attn=effective_local_attn,
                    memory_entries=memory_entries,
                )
                return effective_local_attn

            effective_local_attn = int(self.local_attn_size)
            self._log_temp_attention_window(
                enabled=False,
                effective_local_attn=effective_local_attn,
                memory_entries=0,
            )
            return effective_local_attn

        # Initialize memory bank ONCE per video (persist across chunks to match inference)
        # Only create a new memory bank if one doesn't exist yet (first chunk of new video)
        if self.use_memory_in_training:
            if self._training_memory_bank is None:
                self._initialize_training_memory_bank()
                self._recent_block_features.clear()
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"[SeqTrain-Memory] Initialized memory bank for new video "
                          f"(block_size={self.num_frame_per_block}, "
                          f"dedup={self.memory_dedup_threshold}, top_k={self.memory_top_k})")

        # Derive local switch flags from the persistent segment state so that
        # chunks starting *after* a switch already happened still use the
        # correct segment index for memory retrieval / storage.
        using_second = self.current_segment >= 1
        using_callback = self.current_segment >= 2
        cond_in_use = conditional_dict
        
        for block_index, current_num_frames in enumerate(all_num_frames):
            # Check for first switch (prompt1 -> prompt2)
            if (not using_second) and (not using_callback) and \
               switch_frame_index is not None and (local_start_frame >= switch_frame_index):
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Callback] First switch at local_frame={local_start_frame}")
                
                if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                    log_gpu_memory(f"SeqTrain-Callback: Before first switch recache", 
                                  device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
                
                self._recache_memory_bank_for_new_prompt(
                    full_output=output[:, :local_start_frame, ...],
                    current_start_frame=current_start_frame + local_start_frame,
                    new_conditional_dict=switch_conditional_dict,
                    local_start_frame=local_start_frame,
                    recache_frames=switch_recache_frames,
                )
                
                cond_in_use = switch_conditional_dict
                using_second = True
                self.current_segment = 1
                
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Callback] First switch completed, now using prompt 2")
            
            # Check for callback switch (prompt2 -> prompt3)
            if using_second and (not using_callback) and \
               callback_frame_index is not None and (local_start_frame >= callback_frame_index):
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Callback] Callback switch at local_frame={local_start_frame}")
                
                if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                    log_gpu_memory(f"SeqTrain-Callback: Before callback switch recache", 
                                  device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
                
                # For callback, we may want to incorporate memory from the first segment
                # This is the key difference - we recache with awareness of the first segment
                self._recache_for_callback(
                    output[:, :local_start_frame, ...],
                    current_start_frame + local_start_frame,
                    callback_conditional_dict,
                    local_start_frame,
                    callback_recache_frames,
                    # Pass first segment info for memory-aware recaching
                    first_segment_end=switch_frame_index,
                    first_conditional_dict=conditional_dict
                )
                
                cond_in_use = callback_conditional_dict
                using_callback = True
                self.current_segment = 2
                
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Callback] Callback switch completed, now using prompt 3")
            
            # Derive segment index from the local switch flags (which were
            # initialised from self.current_segment and updated on switch).
            current_segment = 2 if using_callback else (1 if using_second else 0)
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                segment_name = "callback (prompt 3)" if using_callback else ("second (prompt 2)" if using_second else "first (prompt 1)")
                print(f"[SeqTrain-Callback] Processing block {block_index}: frames {local_start_frame}-{local_start_frame + current_num_frames}, using {segment_name}")
            
            noisy_input = noise[:, local_start_frame:local_start_frame + current_num_frames]

            effective_local_attn_size = resolve_local_attn_size()
            
            # ============================================================
            # RETRIEVAL: Query memory bank using previous block's feature
            # (Bridges train-test gap: same as inference pipeline)
            # ============================================================
            retrieval_kv_cache = None
            absolute_block_start = current_start_frame + local_start_frame
            if self.use_memory_in_training and len(self._recent_block_features) > 0:
                retrieval_kv_cache, _, _ = self._retrieve_block_from_memory(
                    query_feature=self._recent_block_features[-1],
                    query_features=list(self._recent_block_features),
                    current_block_start=absolute_block_start,
                    segment_idx=current_segment,
                )
            
            # Spatial denoising loop
            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                exit_flag = (
                    step_idx == exit_flags[0]
                    if self.same_step_across_blocks
                    else step_idx == exit_flags[block_index]
                )
                
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64
                ) * current_timestep
                
                if not exit_flag:
                    # Intermediate steps: no gradients
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                            retrieval_kv_cache=retrieval_kv_cache,  # Inject retrieved KV cache
                            local_attn_size=effective_local_attn_size,
                        )
                        
                        # Add noise for the next step
                        if step_idx < len(self.denoising_step_list) - 1:
                            next_timestep = self.denoising_step_list[step_idx + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                                ),
                            ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Final step may require gradients
                    enable_grad = local_start_frame >= start_gradient_frame_index
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Callback] Block {block_index} final step: enable_grad={enable_grad}")
                    
                    context_manager = torch.enable_grad() if enable_grad else torch.no_grad()
                    with context_manager:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                            retrieval_kv_cache=retrieval_kv_cache,  # Inject retrieved KV cache
                            local_attn_size=effective_local_attn_size,
                        )
                    break
            
            # Record output
            output[:, local_start_frame:local_start_frame + current_num_frames] = denoised_pred
            self._last_recache_latent = denoised_pred.detach()
            
            # Update cache using context noise
            context_timestep = torch.ones_like(timestep) * self.context_noise
            context_noisy = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, denoised_pred.shape[:2])
            
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=context_noisy,
                    conditional_dict=cond_in_use,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                    local_attn_size=effective_local_attn_size,
                )
            
            # ============================================================
            # STORE: Encode block feature + extract KV cache → memory bank
            # (Bridges train-test gap: same as inference pipeline)
            # ============================================================
            if self.use_memory_in_training:
                block_feature = self.store_block_features_with_dedup(
                    output=output,
                    block_start_frame=local_start_frame,
                    num_frames=current_num_frames,
                    segment_idx=current_segment,
                    current_start_frame=current_start_frame,
                )
                # Append to recent block features for multi-query retrieval
                if block_feature is not None:
                    self._recent_block_features.append(block_feature.detach())
            
            # Print sink anomaly gate stats for this block (training)
            if not dist.is_initialized() or dist.get_rank() == 0:
                gate_summary = self.generator.model.get_gate_stats_summary()
                if gate_summary is not None:
                    absolute_block_idx = (current_start_frame + local_start_frame) // self.num_frame_per_block
                    gated_layers = [l for l, c, g, r in gate_summary["per_layer"] if g > 0]
                    entry_info = ""
                    if gate_summary["entries_total"] > 0:
                        entry_info = (f", entries {gate_summary['entries_gated']}/"
                                      f"{gate_summary['entries_total']} gated")
                    print(f"[Rectify] Block {absolute_block_idx}: "
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
                        print(f"[Rectify] Block {absolute_block_idx} per-entry: {', '.join(parts)}")
            
            local_start_frame += current_num_frames
        
        # Compute returned timestep information
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]]).abs(), dim=0
            ).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1]).abs(), dim=0
            ).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]]).abs(), dim=0
            ).item()
        
        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1
        
        return output, denoised_timestep_from, denoised_timestep_to

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
    def _slice_conditional_dict_for_batch(
        conditional_dict: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        sliced = {}
        for key, value in conditional_dict.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] != batch_size:
                sliced[key] = value[:batch_size]
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

    def _recache_training_memory_entry_for_new_prompt(
        self,
        entry,
        new_conditional_dict: dict,
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
        entry_frame_idx = int(entry.frame_idx) if entry is not None else int(entry_frame_idx_override)
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
        conditional_dict = self._slice_conditional_dict_for_batch(
            new_conditional_dict,
            batch_size,
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
        ) * self.context_noise

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

    def _make_dummy_recache_latent(
        self,
        full_output: Optional[torch.Tensor] = None,
        recache_frames: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if recache_frames is not None and recache_frames.shape[1] > 0:
            return recache_frames[:, -min(self.num_frame_per_block, recache_frames.shape[1]):].detach()
        if full_output is not None and full_output.shape[1] > 0:
            return full_output[:, -min(self.num_frame_per_block, full_output.shape[1]):].detach()
        if self._last_recache_latent is not None:
            return self._last_recache_latent.detach()
        if self._training_memory_bank is not None:
            for entry in self._training_memory_bank.entries:
                if entry.latent is not None:
                    return entry.latent.detach()
        return None

    def _recache_training_memory_entries_for_new_prompt(
        self,
        new_conditional_dict: dict,
        full_output: Optional[torch.Tensor] = None,
        recache_frames: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int]:
        local_entries = (
            []
            if self._training_memory_bank is None
            else list(self._training_memory_bank.entries)
        )
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

        dummy_latent = self._make_dummy_recache_latent(
            full_output=full_output,
            recache_frames=recache_frames,
        )
        needs_dummy_forward = (
            local_count < max_count
            or any(entry.latent is None for entry in local_entries)
        )
        if dummy_latent is None and needs_dummy_forward:
            raise RuntimeError(
                "Need a dummy latent to keep FSDP recache forwards aligned across ranks, "
                "but no recache_frames/full_output/memory latent is available."
            )

        recached_entries = 0
        for entry_idx in range(max_count):
            entry = local_entries[entry_idx] if entry_idx < local_count else None
            if self._recache_training_memory_entry_for_new_prompt(
                entry,
                new_conditional_dict,
                latent_override=dummy_latent,
                entry_frame_idx_override=0,
            ):
                recached_entries += 1
        return recached_entries, local_count

    def _recache_memory_bank_for_new_prompt(
        self,
        full_output: torch.Tensor,
        current_start_frame: int,
        new_conditional_dict: dict,
        local_start_frame: int,
        recache_frames: Optional[torch.Tensor] = None,
    ):
        """
        Handle KV recache on prompt switch when memory bank is active.

        Memory bank entries' KV caches are recached under the new prompt,
        then the main generation KV cache (sink + local window) is refreshed
        with the same prompt.
        """
        num_entries = 0
        if self._training_memory_bank is not None:
            num_entries = len(self._training_memory_bank)

        effective_local_attn = self._memory_adjusted_local_attn_size(num_entries)
        recached_entries, total_entries = self._recache_training_memory_entries_for_new_prompt(
            new_conditional_dict,
            full_output=full_output,
            recache_frames=recache_frames,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            aligned_steps = getattr(self, "_last_memory_recache_aligned_steps", total_entries)
            print(f"[SeqTrain-MemoryRecache] Prompt switch: memory bank entries "
                  f"recached {recached_entries}/{total_entries}. "
                  f"Aligned recache forwards={aligned_steps}. "
                  f"Also recaching sink + local_attn_size sliding-window frames "
                  f"(effective local_attn_size={effective_local_attn}).")

        # Recache sink + local_attn_size frames with the new prompt
        self._recache_after_switch(
            full_output, current_start_frame, new_conditional_dict,
            local_start_frame, recache_frames,
            local_attn_size_override=effective_local_attn,
        )

    def _recache_for_callback(
        self,
        output: torch.Tensor,
        current_start_frame: int,
        new_conditional_dict: dict,
        local_start_frame: Optional[int] = None,
        callback_recache_frames: Optional[torch.Tensor] = None,
        first_segment_end: Optional[int] = None,
        first_conditional_dict: Optional[dict] = None
    ):
        """
        Recache KV cache for callback transition with memory-aware mechanism.
        
        Delegates to ``_recache_memory_bank_for_new_prompt`` when the memory bank
        is active so that stored KV caches are re-computed under the new prompt,
        bridging the train-test gap.  Falls back to the lightweight sliding-window
        recache otherwise.
        """
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Callback] _recache_for_callback: current_start_frame={current_start_frame}, "
                  f"first_segment_end={first_segment_end}")

        if self.use_memory_in_training and self._training_memory_bank is not None and len(self._training_memory_bank) > 0:
            self._recache_memory_bank_for_new_prompt(
                full_output=output,
                current_start_frame=current_start_frame,
                new_conditional_dict=new_conditional_dict,
                local_start_frame=local_start_frame,
                recache_frames=callback_recache_frames,
            )
            return

        if not self.global_sink:
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()

        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        if current_start_frame == 0:
            return

        base_sw = (
            self._memory_adjusted_local_attn_size()
            if self.use_memory_in_training
            else self.local_attn_size
        )
        sw = base_sw if base_sw > 0 else current_start_frame
        if callback_recache_frames is not None:
            frames_to_recache = torch.cat([callback_recache_frames, output], dim=1)[:, -sw:, ...]
            num_recache_frames = frames_to_recache.shape[1]
        else:
            if local_start_frame is not None:
                num_recache_frames = min(local_start_frame, sw)
                frames_to_recache = output[:, -num_recache_frames:]
            else:
                num_recache_frames = min(current_start_frame, sw)
                frames_to_recache = output[:, -num_recache_frames:]

        if num_recache_frames <= 0:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print("[SeqTrain-Callback] Skip callback sliding-window recache (num_recache_frames=0)")
            return

        batch_size, num_recache_frames, c, h, w = frames_to_recache.shape

        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=sw,
        )

        context_timestep = torch.ones(
            [batch_size, num_recache_frames],
            device=device, dtype=torch.int64,
        ) * self.context_noise

        self.generator.model.block_mask = block_mask

        recache_start_frame = current_start_frame - num_recache_frames
        recache_start_token = recache_start_frame * self.frame_seq_length
        for blk in self.kv_cache1:
            if "global_end_index" in blk:
                blk["global_end_index"].fill_(recache_start_token)
            if "local_end_index" in blk:
                blk["local_end_index"].fill_(0)

        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_token,
                sink_recache_after_switch=not self.global_sink,
                local_attn_size=sw,
            )

        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
    
    
    def store_block_features_with_dedup(
        self,
        output: torch.Tensor,
        block_start_frame: int,
        num_frames: int,
        segment_idx: int,
        current_start_frame: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Encode block feature, extract block KV cache, and store in memory bank.
        
        This method bridges the train-test gap by storing BOTH the block feature
        AND the block's KV cache in the memory bank, exactly matching the inference
        pipeline's behaviour. Stored KV caches will be retrieved and injected into
        the generator for subsequent blocks.
        
        Strategy (stack-like deduplication — prefer early, higher-quality blocks):
        1. Encode the block visual feature (num_frame_per_block latent → 12 real frames → feature)
           Note: visual feature is frames-only; text prompt is encoded separately for TT scoring
        2. Check max similarity with existing memory bank entries
        3. If max_sim > dedup_threshold: SKIP this block (keep old entries, early frames are better)
        4. If max_sim <= dedup_threshold: extract KV cache + store feature/KV/latent in memory bank
        
        Args:
            output: Generated output latents, shape [B, T, C, H, W]
            block_start_frame: Starting frame index of this block (local)
            num_frames: Number of latent frames in this block (= num_frame_per_block)
            segment_idx: Current segment index (0, 1, or 2)
            current_start_frame: Absolute start frame of the chunk (for KV cache positioning)
            
        Returns:
            Block feature tensor [B, feature_dim] or None if encoding fails.
            Used as prev_block_feature for the next block's retrieval query.
        """
        if not self.use_memory_in_training or self._training_memory_bank is None:
            return None
        
        if self.vae is None or self.memory_encoder is None:
            return None
        
        try:
            # Get prompt for this segment
            prompt = self._current_prompts.get(segment_idx, None)
            if prompt is None:
                return None
            
            # Encode the entire block as a single visual feature (frames only, no text prompt)
            # Text prompt is encoded separately and stored alongside for TT similarity scoring
            block_feature = self._encode_block_features_for_training(
                latent_frames=output,
                block_start_idx=block_start_frame,
                num_frames_in_block=num_frames,
                text_prompt=None  # Visual-only encoding: 12 frames without text prompt
            )
            
            if block_feature is None:
                return None
            
            absolute_frame_idx = current_start_frame + block_start_frame
            
            # Stack-like deduplication: early blocks have higher quality, so we keep old entries.
            # If the new block is too similar to any existing entry (> threshold), skip adding it.
            # Only add the new block if it brings sufficiently new information (sim <= threshold).
            # Visual-only cosine similarity (PE-Core, no text features).
            max_sim = self._training_memory_bank.check_similarity(
                block_feature,
            )
            block_idx = absolute_frame_idx // self.num_frame_per_block
            
            if len(self._training_memory_bank) > 0 and max_sim > self.memory_dedup_threshold:
                # New block is too similar to existing entries — skip (keep old, higher-quality ones)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"[SeqTrain-Memory] Block {block_idx} (frame {absolute_frame_idx}): "
                          f"Skipped (combined_sim={max_sim:.4f} > threshold={self.memory_dedup_threshold}). "
                          f"Bank size: {len(self._training_memory_bank)}")
                return block_feature
            
            # Extract block's KV cache from the full cache buffer
            # This is the key difference from the old version: we now store actual KV cache
            # so it can be retrieved and injected during subsequent blocks (bridging train-test gap)
            block_kv_cache = extract_kv_cache_for_frame(
                self.kv_cache1,
                frame_idx=current_start_frame + block_start_frame,
                frame_seq_length=self.frame_seq_length,
                num_frames_to_extract=num_frames,
            )
            
            # Store block feature WITH KV cache AND latent (bridging train-test gap)
            # Use ABSOLUTE frame index as the identifier so blocks from different chunks
            # are properly distinguished in the memory bank.
            # The latent is stored so that KV recache on prompt switch can re-run the
            # generator even when the original chunk output is no longer available.
            block_latent = output[:, block_start_frame:block_start_frame + num_frames].detach()
            self._training_memory_bank.add_entry(
                frame_idx=absolute_frame_idx,
                feature=block_feature,
                kv_cache=block_kv_cache,
                text_feature=None,  # PE-Core is visual-only
                latent=block_latent,
            )
            
            if not dist.is_initialized() or dist.get_rank() == 0:
                if block_idx % 10 == 0:
                    print(f"[SeqTrain-Memory] Block {block_idx} (frame {absolute_frame_idx}): "
                          f"Stored in memory bank (max_sim={max_sim:.4f}). "
                          f"Bank size: {len(self._training_memory_bank)}")
            
            return block_feature
        
        except Exception as e:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Callback] store_block_features_with_dedup failed: {e}")
            return None
    
    def _retrieve_block_from_memory(
        self,
        query_feature: torch.Tensor,
        current_block_start: int,
        segment_idx: int,
        query_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Optional[list], list, list]:
        """
        Retrieve similar block's KV cache from memory bank for injection during training.

        Supports **multi-query** retrieval: when *query_features* (list of recent
        block features) is provided, similarity is averaged across all queries.

        Args:
            query_feature: Fallback single query feature [B, feature_dim] or [feature_dim].
            current_block_start: Starting frame of the current block being generated.
            segment_idx: Current segment index (for text feature selection).
            query_features: Optional list of recent block features for multi-query.

        Returns:
            Tuple of (retrieval_kv_cache, retrieved_frame_indices, retrieved_similarities).
            retrieval_kv_cache is None if no retrievable entries are found.
        """
        if self._training_memory_bank is None or len(self._training_memory_bank) == 0:
            return None, [], []
        
        try:
            # Prepare query feature(s) — take first sample if batched
            query = query_feature
            if query.dim() > 1:
                query = query[0]  # [feature_dim]

            multi_queries: Optional[List[torch.Tensor]] = None
            if query_features is not None and len(query_features) > 1:
                multi_queries = [
                    qf[0] if qf.dim() > 1 else qf for qf in query_features
                ]

            # Align with inference retrieval policy:
            # 1) Exclude sink frames [0, sink_size)
            # 2) Exclude sliding window frames [current-local_attn_size, current)
            sink_size = max(0, int(self.sink_size))
            effective_local_attn = self._memory_adjusted_local_attn_size()
            local_attn = effective_local_attn if effective_local_attn > 0 else 0

            # Need frames beyond sink + local window before retrieval is meaningful.
            min_frames_for_retrieval = sink_size + local_attn + 1
            if current_block_start < min_frames_for_retrieval:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    current_block_idx = current_block_start // self.num_frame_per_block
                    print(f"[SeqTrain-Memory] Block {current_block_idx} (frame {current_block_start}): "
                          f"Skip retrieval (need frame >= {min_frames_for_retrieval}, "
                          f"sink={sink_size}, local_attn={local_attn})")
                return None, [], []

            exclude_frame_indices = set()
            for frame_idx in range(sink_size):
                exclude_frame_indices.add(frame_idx)

            if local_attn > 0:
                window_start = max(0, current_block_start - local_attn)
                for frame_idx in range(window_start, current_block_start):
                    exclude_frame_indices.add(frame_idx)

            # Also exclude current block start itself
            exclude_frame_indices.add(current_block_start)
            exclude_frame_indices = list(exclude_frame_indices)
                
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                excluded_block_indices = sorted({
                    idx // self.num_frame_per_block for idx in exclude_frame_indices
                })
                preview = excluded_block_indices[:20]
                suffix = " ..." if len(excluded_block_indices) > 20 else ""
                print(
                    f"[SeqTrain-Memory][Debug] Block {current_block_start // self.num_frame_per_block} "
                    f"(frame {current_block_start}): exclusion policy -> "
                    f"sink_size={sink_size}, local_attn={local_attn}, "
                    f"excluded_blocks={preview}{suffix} "
                    f"(total={len(excluded_block_indices)})"
                )
            
            # Retrieve using visual-only similarity (PE-Core, no text features)
            # Multi-query: average similarity across all local-window block features
            target_device = self.generator.model.patch_embedding.weight.device
            result = self._training_memory_bank.get_retrieval_kv_cache(
                query_feature=query,
                target_device=target_device,
                target_dtype=torch.bfloat16,
                exclude_frame_indices=exclude_frame_indices,
                return_info=True,
                return_score_breakdown=True,
                query_features=multi_queries,
            )
            
            retrieval_kv_cache, frame_indices, similarities, visual_scores, _ = result
            
            if not dist.is_initialized() or dist.get_rank() == 0:
                current_block_idx = current_block_start // self.num_frame_per_block
                n_queries = len(multi_queries) if multi_queries is not None else 1
                if retrieval_kv_cache is not None:
                    retrieved_block_indices = [fid // self.num_frame_per_block for fid in frame_indices]
                    visual_str = ", ".join([f"{s:.3f}" for s in visual_scores])
                    print(f"[SeqTrain-Memory] Block {current_block_idx} (frame {current_block_start}): "
                          f"Retrieved blocks: {retrieved_block_indices}, "
                          f"Visual Scores (avg of {n_queries} queries): [{visual_str}]")
                else:
                    bank_size = len(self._training_memory_bank) if self._training_memory_bank else 0
                    print(f"[SeqTrain-Memory] Block {current_block_idx} (frame {current_block_start}): "
                          f"No retrievable block found (bank size: {bank_size}, queries: {n_queries})")
            
            return retrieval_kv_cache, frame_indices, similarities
        
        except Exception as e:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Callback] _retrieve_block_from_memory failed: {e}")
            return None, [], []
    
    def get_similar_blocks_from_segment(
        self,
        query_feature: torch.Tensor,
        segment_frame_range: Tuple[int, int],
        top_k: int = 3
    ) -> List[Tuple[int, float]]:
        """
        Get similar blocks from a specific segment in training memory bank.
        
        Each entry in the memory bank represents a block (num_frame_per_block latent 
        frames encoded together with text prompt). The frame_idx stored is the 
        block_start_frame.
        
        Args:
            query_feature: Query block feature tensor
            segment_frame_range: (start_frame, end_frame) of the target segment
            top_k: Number of similar blocks to retrieve
            
        Returns:
            List of (block_start_frame, similarity) tuples
        """
        if self._training_memory_bank is None or len(self._training_memory_bank) == 0:
            return []
        
        try:
            import torch.nn.functional as F
            
            # Normalize query
            query_cpu = F.normalize(query_feature.detach().cpu().float(), dim=-1)
            if query_cpu.dim() > 1:
                query_cpu = query_cpu.squeeze(0)
            
            # Find block entries in the target segment
            results = []
            start_frame, end_frame = segment_frame_range
            
            for entry in self._training_memory_bank.entries:
                if start_frame <= entry.frame_idx < end_frame:
                    # Compute similarity
                    entry_feature = F.normalize(entry.feature.float(), dim=-1)
                    sim = torch.dot(query_cpu, entry_feature).item()
                    results.append((entry.frame_idx, sim))
            
            # Sort by similarity and return top-k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        
        except Exception as e:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Callback] get_similar_blocks_from_segment failed: {e}")
            return []
