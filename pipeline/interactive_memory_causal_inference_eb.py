"""
Interactive Memory-augmented Causal Inference Pipeline for LongLive with PE-Core Vision Encoder

This pipeline combines:
1. Interactive prompt switching for multi-segment generation
2. Memory bank mechanism with PE-Core VisionTransformer for visual features
3. **Block-level retrieval** (matching training pipeline):
   - Each block = num_frame_per_block latent frames = 12 real frames
   - Visual feature: 12 real frames encoded by PE-Core VisionTransformer
   - Visual-only cosine similarity for retrieval (no text encoder)
4. Stack-like dedup: if new block similarity > threshold, skip it (keep older, higher-quality blocks)

Key Feature - KV Recache on Prompt Switch:
When switching prompts, memory bank entries' KV caches are recached under
the new prompt, and the sink/local sliding-window cache is refreshed too.
The visual features from PE-Core remain unchanged.

SPDX-License-Identifier: Apache-2.0
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn.functional as F

from pipeline.memory_causal_inference_eb import MemoryCausalInferencePipelineEB
from utils.memory_bank import MemoryBank, extract_kv_cache_for_frame
from utils.memory_bank_eb import PECoreMemoryEncoder
from utils.memory import gpu, get_cuda_free_memory_gb
from utils.debug_option import DEBUG


class InteractiveMemoryCausalInferencePipelineEB(MemoryCausalInferencePipelineEB):
    """
    Interactive Causal Inference Pipeline with Memory Bank using PE-Core Vision Encoder.
    
    This extends MemoryCausalInferencePipelineEB with:
    1. Support for multiple prompts with switching at specified frame indices
    2. On prompt switch: memory bank KV caches and sink/local sliding-window
       frames are recached with the new prompt
    3. Block-level retrieval (matching training pipeline)
    4. PE-Core visual features remain unchanged during prompt switch
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
        Initialize the Interactive Memory-augmented Causal Inference Pipeline with PE-Core.
        
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
        super().__init__(args, device, generator, text_encoder, vae, memory_encoder_config)
        self.global_sink = getattr(args, "global_sink", False)
    
    def _extract_kv_cache_by_offset(
        self,
        kv_cache: List[Dict[str, torch.Tensor]],
        frames_from_end: int,
        num_frames: int = 1
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Extract KV cache entries using offset from the current end.
        
        This method correctly handles sliding window mode by using the
        local_end_index to calculate positions.
        
        Args:
            kv_cache: Full KV cache from all transformer blocks
            frames_from_end: How many frames back from the current end (0 = last frame)
            num_frames: Number of frames to extract
            
        Returns:
            Extracted KV cache for the specified frame(s)
        """
        # Get local_end_index from the first block
        local_end_index = 0
        if len(kv_cache) > 0:
            local_end_idx = kv_cache[0].get('local_end_index', None)
            if local_end_idx is not None:
                if hasattr(local_end_idx, 'item'):
                    local_end_index = local_end_idx.item()
                else:
                    local_end_index = int(local_end_idx)
        
        # Calculate the range to extract
        # frames_from_end=0 means the last frame, frames_from_end=1 means second to last, etc.
        end_idx = local_end_index - frames_from_end * self.frame_seq_length
        start_idx = end_idx - num_frames * self.frame_seq_length
        
        extracted_cache = []
        for block_cache in kv_cache:
            block_extracted = {}
            for key, value in block_cache.items():
                if key in ['k', 'v']:
                    # Get actual valid range
                    cache_size = value.shape[1]
                    actual_start = max(0, min(start_idx, cache_size))
                    actual_end = max(0, min(end_idx, cache_size))
                    
                    # Extract the relevant portion
                    if actual_end > actual_start:
                        block_extracted[key] = value[:, actual_start:actual_end].detach().cpu().clone()
                    else:
                        # If no valid range, create empty tensor
                        block_extracted[key] = value[:, :0].detach().cpu().clone()
                else:
                    block_extracted[key] = block_cache[key]
            extracted_cache.append(block_extracted)
        
        return extracted_cache
    
    def _recache_memory_bank_for_new_prompt(
        self,
        output: torch.Tensor,
        current_start_frame: int,
        new_conditional_dict: Dict[str, torch.Tensor]
    ):
        """
        Handle KV recache on prompt switch when memory bank is active.
        
        Memory bank entries' KV caches are recached under the new prompt,
        then the main generation KV cache (sink + local window) is refreshed
        with the same prompt.
        
        Args:
            output: All generated output latents so far, shape [B, T, C, H, W]
            current_start_frame: Current frame index (exclusive - frames 0 to current_start_frame-1 exist)
            new_conditional_dict: New text embeddings for the new prompt
        """
        if current_start_frame == 0:
            return
        
        memory_bank = self._get_memory_bank()
        num_entries = len(memory_bank) if memory_bank is not None else 0
        
        effective_local_attn = self._memory_adjusted_local_attn_size(num_entries)

        recached_entries, total_entries = super()._recache_memory_bank_for_new_prompt(
            new_conditional_dict,
            dummy_latent=output[:, :current_start_frame],
        )
        aligned_steps = getattr(self, "_last_memory_recache_aligned_steps", total_entries)

        print(f"[InteractiveMemoryEB] Prompt switch: memory bank entries "
              f"recached {recached_entries}/{total_entries}. "
              f"Aligned recache forwards={aligned_steps}. "
              f"Also recaching sink + local_attn_size sliding-window frames "
              f"(effective local_attn_size={effective_local_attn}).")
        
        # Recache sink + local_attn_size frames with the new prompt
        self._recache_after_switch(
            output,
            current_start_frame,
            new_conditional_dict,
            local_attn_size_override=effective_local_attn,
        )
    
    def inference_interactive_with_memory(
        self,
        noise: torch.Tensor,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
        use_memory_retrieval: bool = True
    ) -> torch.Tensor:
        """
        Perform interactive inference with memory-augmented KV cache retrieval.
        
        This method generates video frame-by-frame while:
        1. Supporting multiple prompts with switching at specified frame indices
        2. Decoding each generated frame to pixel space for feature extraction
        3. Storing features and KV cache in the memory bank
        4. Retrieving similar historical KV caches for future generation
        5. Recaching memory bank KV caches when switching prompts
        
        Args:
            noise: Input noise tensor, shape [B, num_output_frames, C, H, W]
            text_prompts_list: List of prompt lists for each segment
            switch_frame_indices: Frame indices at which to switch prompts
            return_latents: Whether to return latents
            low_memory: Whether to use low memory mode
            use_memory_retrieval: Whether to use memory retrieval
            
        Returns:
            Generated video tensor, shape [B, T, C, H, W] in [0, 1] range
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        if use_memory_retrieval and batch_size > 1:
            print("[InteractiveMemoryEB] Warning: multi-sample memory retrieval uses sample 0 only.")
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block
        
        # Encode all prompts
        print(f"[InteractiveMemoryEB] Encoding {len(text_prompts_list)} prompt segments")
        cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]
        
        if use_memory_retrieval:
            print(f"[InteractiveMemoryEB] Visual-only retrieval (PE-Core, no text encoder)")
        
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
            kv_policy = f"local, size={local_attn_cfg}, max_memory_adjusted={max_local_attn_cfg}"
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        
        print(f"[InteractiveMemoryEB] KV cache size: {kv_cache_size} (policy: {kv_policy})")
        
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
        
        # Segment tracking
        segment_idx = 0
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )
        
        # Recent block features for multi-query retrieval (matching training pipeline).
        from collections import deque
        local_query_blocks = max(1, self.local_attn_size // self.num_frame_per_block)
        recent_block_features: deque = deque(maxlen=local_query_blocks)
        prev_block_feature = None
        
        # Track current segment prompt for logging
        current_segment_prompt = text_prompts_list[0][0] if isinstance(text_prompts_list[0], list) else text_prompts_list[0]
        if use_memory_retrieval:
            print(f"[InteractiveMemoryEB] Block-level retrieval (block size: {self.num_frame_per_block} latent frames)")
        
        # Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        
        if DEBUG:
            print(f"[InteractiveMemoryEB] all_num_frames: {all_num_frames}")
            print(f"[InteractiveMemoryEB] switch_frame_indices: {switch_frame_indices}")
        
        for block_idx, current_num_frames in enumerate(all_num_frames):
            # Check if we need to switch prompts
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                segment_idx += 1
                print(f"[InteractiveMemoryEB] Switching to segment {segment_idx} at frame {current_start_frame}")
                print(f"[InteractiveMemoryEB] New prompt: {text_prompts_list[segment_idx][0][:100]}...")
                
                # Update current prompt for logging
                current_segment_prompt = text_prompts_list[segment_idx][0] if isinstance(text_prompts_list[segment_idx], list) else text_prompts_list[segment_idx]
                
                # Recache memory bank with new prompt
                if use_memory_retrieval:
                    self._recache_memory_bank_for_new_prompt(
                        output, current_start_frame, cond_list[segment_idx]
                    )
                else:
                    # Just do the standard recache without memory bank
                    self._recache_after_switch(output, current_start_frame, cond_list[segment_idx])
                
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )
            
            cond_in_use = cond_list[segment_idx]
            
            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames
            ]

            if use_memory_retrieval:
                effective_local_attn_size = self._apply_memory_adjusted_attention_window()
            
            # Reset retrieval KV cache for each block
            retrieval_kv_cache = None
            retrieved_frame_ids = []
            retrieved_similarities = []
            retrieved_visual_scores = []
            
            # Retrieve KV cache using multi-query (all blocks in local window)
            if use_memory_retrieval and len(recent_block_features) > 0:
                memory_bank = self._get_memory_bank()
                if memory_bank is not None and len(memory_bank) > 0:
                    multi_q = list(recent_block_features)
                    retrieval_kv_cache, retrieved_frame_ids, retrieved_similarities, retrieved_visual_scores, _ = self._retrieve_kv_cache(
                        query_feature=multi_q[-1],
                        current_frame_idx=current_start_frame,
                        sample_idx=0,
                        return_info=True,
                        return_score_breakdown=True,
                        query_features=multi_q if len(multi_q) > 1 else None,
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
                        conditional_dict=cond_in_use,
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
                        conditional_dict=cond_in_use,
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
                conditional_dict=cond_in_use,
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
                    visual_str = ", ".join([f"{s:.3f}" for s in retrieved_visual_scores])
                    retrieved_block_indices = [fid // self.num_frame_per_block for fid in retrieved_frame_ids]
                    print(f"[InteractiveMemoryEB] Block {block_idx} (frame {current_start_frame}): "
                          f"Retrieved block KV cache (visual-only). "
                          f"Retrieved blocks: {retrieved_block_indices}, "
                          f"Visual: [{visual_str}]")
                elif block_idx > 0:
                    memory_bank = self._get_memory_bank()
                    bank_size = len(memory_bank) if memory_bank is not None else 0
                    print(f"[InteractiveMemoryEB] Block {block_idx}: No retrievable block found "
                          f"(memory bank size: {bank_size})")
                
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
                memory_bank = self._get_memory_bank()
                max_sim = memory_bank.check_similarity(block_feature)
                
                if len(memory_bank) > 0 and max_sim > self.memory_dedup_threshold:
                    # New block is too similar — skip (keep old, higher-quality entries)
                    print(f"[InteractiveMemoryEB] Block {block_idx}: "
                          f"Skipped (combined_sim={max_sim:.4f} > threshold={self.memory_dedup_threshold}). "
                          f"Bank size: {len(memory_bank)}")
                else:
                    # Store block feature + block KV cache
                    self._store_block_in_memory(
                        block_start_frame=current_start_frame,
                        num_frames=current_num_frames,
                        feature=block_feature,
                        kv_cache=self.kv_cache1,
                        sample_idx=0,
                        latent=denoised_pred,
                    )
                    
                    # Log memory bank status
                    if block_idx % 10 == 0:
                        memory_bank = self._get_memory_bank()
                        bank_size = len(memory_bank) if memory_bank is not None else 0
                        print(f"[InteractiveMemoryEB] Block {block_idx}: Memory bank has {bank_size} block entries")
                
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
                if isinstance(self.memory_bank, list):
                    for bank in self.memory_bank:
                        bank.clear()
                else:
                    self.memory_bank.clear()
        
        if return_latents:
            return video, output.to(noise.device)
        else:
            return video
    
    def _recache_after_switch(
        self,
        output,
        current_start_frame,
        new_conditional_dict,
        local_attn_size_override: Optional[int] = None,
    ):
        """
        Standard recache method (without memory bank) for compatibility.
        This is used when memory retrieval is disabled.
        """
        if not self.global_sink:
            # Reset KV cache
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
        
        # Reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        
        if current_start_frame == 0:
            return
        
        recache_local_attn_size = (
            self.local_attn_size
            if local_attn_size_override is None
            else local_attn_size_override
        )
        self._log_temp_attention_window(
            enabled=local_attn_size_override is not None,
            effective_local_attn=recache_local_attn_size,
            memory_entries=None,
        )
        num_recache_frames = (
            current_start_frame if recache_local_attn_size == -1
            else min(recache_local_attn_size, current_start_frame)
        )
        recache_start_frame = current_start_frame - num_recache_frames
        
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        
        batch_size = frames_to_recache.shape[0]
        device = frames_to_recache.device
        
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=recache_local_attn_size
        )
        
        context_timestep = torch.ones(
            [batch_size, num_recache_frames],
            device=device,
            dtype=torch.int64
        ) * self.args.context_noise
        
        self.generator.model.block_mask = block_mask
        
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                sink_recache_after_switch=not self.global_sink,
                local_attn_size=recache_local_attn_size,
            )
        
        # Reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
    
    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        """
        Main inference method - delegates to interactive memory inference with PE-Core.
        
        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W)
            text_prompts_list: List of prompt lists for each segment
            switch_frame_indices: Frame indices at which to switch prompts
            return_latents: Whether to return latents
            low_memory: Enable low-memory mode
            
        Returns:
            Generated video tensor
        """
        use_memory = getattr(self.args, "use_memory_retrieval", True)
        
        return self.inference_interactive_with_memory(
            noise=noise,
            text_prompts_list=text_prompts_list,
            switch_frame_indices=switch_frame_indices,
            return_latents=return_latents,
            low_memory=low_memory,
            use_memory_retrieval=use_memory
        )
