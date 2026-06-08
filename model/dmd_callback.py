# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0

"""
DMD Callback Model

This module extends DMDSwitch to support three-segment prompt training with
a callback mechanism for improved temporal consistency.

Key Features:
1. Three-segment prompt support: prompt1 -> prompt2 -> prompt3 (callback)
2. Two switch points for transitioning between prompts
3. Callback mechanism where prompt3 references/relates to prompt1
4. Memory-aware training support (optional)

The callback design helps the model learn to maintain consistency when
returning to similar scenes or topics after a different segment.
"""

import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import time

from model.base import SelfForcingModel
from utils.memory import log_gpu_memory
import torch.distributed as dist
from model.dmd_switch import DMDSwitch
from pipeline.streaming_callback_training import StreamingCallbackTrainingPipeline
from einops import rearrange

from utils.debug_option import DEBUG, LOG_GPU_MEMORY


class DMDCallback(DMDSwitch):
    """DMD variant that supports three-segment prompt switching with callback.
    
    This extends DMDSwitch to handle:
    - Three prompts instead of two
    - Two switch points (first switch + callback switch)
    - Optional memory-aware training for better temporal consistency
    """

    def _initialize_inference_pipeline(self):
        """Initialize the callback training pipeline."""
        # Get memory-aware training settings
        use_memory_in_training = getattr(self.args, "use_memory_in_training", False)
        memory_top_k = getattr(self.args, "memory_top_k", 3)
        
        # Memory encoder configuration (PE-Core Vision Encoder, visual-only)
        memory_encoder_config = {
            "pe_config": getattr(self.args, "memory_encoder_pe_config", "PE-Core-B16-224"),
            "checkpoint_path": getattr(self.args, "memory_encoder_pe_checkpoint", None),
        }
        
        # Memory deduplication threshold
        memory_dedup_threshold = getattr(self.args, "memory_dedup_threshold", 0.95)
        
        # Get VAE for memory encoding (decode latents to pixels)
        vae = getattr(self, 'vae', None)
        
        self.inference_pipeline = StreamingCallbackTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            context_noise=self.args.context_noise,
            local_attn_size=getattr(self.args, "model_kwargs", {}).get("local_attn_size", -1),
            sink_size=getattr(self.args, "model_kwargs", {}).get("sink_size", 0),
            slice_last_frames=getattr(self.args, "slice_last_frames", 21),
            global_sink=getattr(self.args, "global_sink", False),
            # Callback-specific settings with memory encoder
            use_memory_in_training=use_memory_in_training,
            memory_top_k=memory_top_k,
            memory_dedup_threshold=memory_dedup_threshold,
            memory_encoder_config=memory_encoder_config,
            vae=vae,
        )
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[DMDCallback] Initialized StreamingCallbackTrainingPipeline")
            print(f"[DMDCallback] use_memory_in_training={use_memory_in_training}, "
                  f"memory_top_k={memory_top_k}")
            if use_memory_in_training:
                print(f"[DMDCallback] Memory encoder: PE-Core VisionTransformer (visual-only, block-level encoding)")
                print(f"[DMDCallback] Memory dedup threshold: {memory_dedup_threshold}")
                print(f"[DMDCallback] PE config: {memory_encoder_config['pe_config']}")
                print(f"[DMDCallback] PE checkpoint: {memory_encoder_config['checkpoint_path']}")

    def _get_switch_frame_index(self, max_length: Optional[int] = None):
        """
        Get the switch frame index for the first switch point.
        This method is needed by _get_callback_frame_indices.
        """
        import random
        
        switch_mode = getattr(self.args, "switch_mode", "fixed")
        
        if switch_mode == "random":
            block = self.args.num_frame_per_block
            min_idx = self.args.min_switch_frame_index
            max_idx = self.args.max_switch_frame_index
            if min_idx == max_idx:
                switch_idx = min_idx
            else:
                choices = list(range(min_idx, max_idx, block))
                if max_length is not None:
                    choices = [choice for choice in choices if choice < max_length]
                
                if len(choices) == 0:
                    if max_length is not None:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")
                    else:
                        switch_idx = block
                else:
                    if dist.get_rank() == 0:
                        switch_idx = random.choice(choices)
                    else:
                        switch_idx = 0  # placeholder; will be overwritten by broadcast
                switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
                if dist.is_initialized():
                    dist.broadcast(switch_idx_tensor, src=0)
                switch_idx = switch_idx_tensor.item()
        elif switch_mode == "fixed":
            switch_idx = getattr(self.args, "fixed_switch_index", 21)
            if max_length is not None:
                assert max_length > switch_idx, f"max_length {max_length} is not greater than switch_idx {switch_idx}"
        elif switch_mode == "random_choice":
            switch_choices = getattr(self.args, "switch_choices", [])
            if len(switch_choices) == 0:
                raise ValueError("switch_choices is empty")
            else:
                if max_length is not None:
                    switch_choices = [choice for choice in switch_choices if choice < max_length]
                    if len(switch_choices) == 0:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")
                
                if dist.get_rank() == 0:
                    switch_idx = random.choice(switch_choices)
                else:
                    switch_idx = 0
            switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
            if dist.is_initialized():
                dist.broadcast(switch_idx_tensor, src=0)
            switch_idx = switch_idx_tensor.item()
        else:
            raise ValueError(f"Invalid switch_mode: {switch_mode}")
        
        return switch_idx

    def _get_callback_frame_indices(self, max_length: Optional[int] = None):
        """
        Get both switch indices for three-segment training.
        
        Returns:
            Tuple[int, int]: (first_switch_index, callback_switch_index)
        """
        import random
        
        # Get first switch index using parent's method
        first_switch_idx = self._get_switch_frame_index(max_length)
        
        # Get callback switch index
        callback_mode = getattr(self.args, "callback_switch_mode", "random_choice")
        min_callback_gap = getattr(self.args, "min_callback_gap", 36)
        
        # Calculate minimum callback frame (must be after first switch + gap)
        min_callback_frame = first_switch_idx + min_callback_gap
        
        if callback_mode == "random_choice":
            callback_choices = getattr(self.args, "callback_switch_choices", [])
            if len(callback_choices) == 0:
                # Default to some reasonable values
                callback_choices = [93, 111, 129, 147, 165, 183, 201]
            
            # Filter choices based on constraints
            valid_choices = [c for c in callback_choices if c >= min_callback_frame]
            if max_length is not None:
                valid_choices = [c for c in valid_choices if c < max_length]
            
            if len(valid_choices) == 0:
                # Fallback: use the minimum valid callback frame
                callback_switch_idx = min_callback_frame
                if max_length is not None and callback_switch_idx >= max_length:
                    raise ValueError(
                        f"No valid callback switch choices available. "
                        f"min_callback_frame={min_callback_frame}, max_length={max_length}"
                    )
            else:
                if dist.get_rank() == 0:
                    callback_switch_idx = random.choice(valid_choices)
                else:
                    callback_switch_idx = 0
                
            callback_idx_tensor = torch.tensor(callback_switch_idx, device=self.device)
            if dist.is_initialized():
                dist.broadcast(callback_idx_tensor, src=0)
            callback_switch_idx = callback_idx_tensor.item()
            
        elif callback_mode == "fixed":
            callback_switch_idx = getattr(self.args, "fixed_callback_index", 111)
            if callback_switch_idx < min_callback_frame:
                callback_switch_idx = min_callback_frame
            if max_length is not None:
                assert max_length > callback_switch_idx, \
                    f"max_length {max_length} must be > callback_switch_idx {callback_switch_idx}"
        else:
            raise ValueError(f"Invalid callback_switch_mode: {callback_mode}")
        
        return first_switch_idx, callback_switch_idx

    def generator_loss_with_callback(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        switch_conditional_dict: dict,
        callback_conditional_dict: dict,
        clean_latent: torch.Tensor = None,
        initial_latent: torch.Tensor = None,
        switch_frame_index: int = None,
        callback_frame_index: int = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos with three-segment callback and compute the DMD loss.
        
        Args:
            image_or_video_shape: Shape of the video to generate
            conditional_dict: First segment's conditional info (prompt 1)
            unconditional_dict: Negative prompt conditional info
            switch_conditional_dict: Second segment's conditional info (prompt 2)
            callback_conditional_dict: Callback segment's conditional info (prompt 3)
            clean_latent: Optional clean latents
            initial_latent: Optional initial latent for i2v
            switch_frame_index: First switch frame index
            callback_frame_index: Callback switch frame index
            
        Returns:
            Tuple[torch.Tensor, dict]: (loss, log_dict)
        """
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Callback Generator loss: Before generator unroll", 
                          device=self.device, rank=dist.get_rank())
        
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        _t_gen_start = time.time()
        
        if DEBUG and dist.get_rank() == 0:
            print(f"[DMDCallback] generator_rollout with callback")
            print(f"[DMDCallback] switch_frame_index={switch_frame_index}, "
                  f"callback_frame_index={callback_frame_index}")
        
        # Run generator with callback
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = \
            self._run_generator_with_callback(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                switch_conditional_dict=switch_conditional_dict,
                callback_conditional_dict=callback_conditional_dict,
                initial_latent=initial_latent,
                slice_last_frames=slice_last_frames,
                switch_frame_index=switch_frame_index,
                callback_frame_index=callback_frame_index,
            )
        
        if dist.get_rank() == 0 and DEBUG:
            print(f"[DMDCallback] pred_image: {pred_image.shape}")
            if gradient_mask is not None:
                print(f"[DMDCallback] gradient_mask: {gradient_mask[0, :, 0, 0, 0]}")
        
        gen_time = time.time() - _t_gen_start
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Callback Generator loss: After generator unroll", 
                          device=self.device, rank=dist.get_rank())
        
        # For callback training, we compute loss on the callback segment
        # This helps the model learn temporal consistency when returning to similar content
        _t_loss_start = time.time()
        
        # Use callback conditional dict for loss computation (focus on callback segment)
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=callback_conditional_dict,  # Use callback prompt for loss
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Callback Generator loss: After compute_distribution_matching_loss", 
                          device=self.device, rank=dist.get_rank())
        
        loss_time = time.time() - _t_loss_start
        
        dmd_log_dict.update({
            "gen_time": gen_time,
            "loss_time": loss_time,
            "switch_frame_index": switch_frame_index,
            "callback_frame_index": callback_frame_index,
        })
        
        return dmd_loss, dmd_log_dict

    def critic_loss_with_callback(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        switch_conditional_dict: dict,
        callback_conditional_dict: dict,
        clean_latent: torch.Tensor = None,
        initial_latent: torch.Tensor = None,
        switch_frame_index: int = None,
        callback_frame_index: int = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos with three-segment callback and train the critic.
        
        Similar to generator_loss_with_callback but for critic training.
        """
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Callback Critic loss: Before generator unroll", 
                          device=self.device, rank=dist.get_rank())
        
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        _t_gen_start = time.time()
        
        with torch.no_grad():
            if DEBUG and dist.get_rank() == 0:
                print(f"[DMDCallback] critic_rollout with callback")
            
            generated_image, _, denoised_timestep_from, denoised_timestep_to = \
                self._run_generator_with_callback(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    switch_conditional_dict=switch_conditional_dict,
                    callback_conditional_dict=callback_conditional_dict,
                    initial_latent=initial_latent,
                    slice_last_frames=slice_last_frames,
                    switch_frame_index=switch_frame_index,
                    callback_frame_index=callback_frame_index,
                )
        
        gen_time = time.time() - _t_gen_start
        batch_size, num_frame = generated_image.shape[:2]
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Callback Critic loss: After generator unroll", 
                          device=self.device, rank=dist.get_rank())
        
        _t_loss_start = time.time()
        
        # Compute the fake prediction using callback conditional dict
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            batch_size,
            num_frame,
            self.num_frame_per_block,
            uniform_timestep=True
        )
        
        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000
        
        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)
        
        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        
        # Use callback conditional dict for critic
        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=callback_conditional_dict,
            timestep=critic_timestep
        )
        
        # Compute the denoising loss
        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        
        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )
        
        loss_time = time.time() - _t_loss_start
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Callback Critic loss: After denoising loss", 
                          device=self.device, rank=dist.get_rank())
        
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach(),
            "gen_time": gen_time,
            "loss_time": loss_time,
            "switch_frame_index": switch_frame_index,
            "callback_frame_index": callback_frame_index,
        }
        
        return denoising_loss, critic_log_dict

    def _run_generator_with_callback(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        switch_conditional_dict: dict,
        callback_conditional_dict: dict,
        initial_latent: torch.Tensor = None,
        slice_last_frames: int = 21,
        switch_frame_index: int = None,
        callback_frame_index: int = None,
        prompt_strings: dict = None,  # {0: prompt1, 1: prompt2, 2: prompt3}
    ):
        """
        Run the generator with three-segment callback support.
        
        This extends _run_generator to handle callback mechanism.
        
        Args:
            prompt_strings: Optional dict mapping segment index to prompt string
                           for memory-aware training with joint frame+prompt encoding
        """
        batch_size, num_frame = image_or_video_shape[:2]
        
        # Initialize pipeline if needed
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()
        
        # Set prompts for memory-aware training (joint frame+prompt encoding)
        if prompt_strings is not None and hasattr(self.inference_pipeline, 'set_current_prompts'):
            self.inference_pipeline.set_current_prompts(prompt_strings)
            if DEBUG and dist.get_rank() == 0:
                print(f"[DMDCallback] Set prompt strings for memory-aware training")
        
        # Initialize KV cache
        self.inference_pipeline._initialize_kv_cache(
            batch_size=batch_size,
            dtype=torch.bfloat16,
            device=self.device
        )
        self.inference_pipeline._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=torch.bfloat16,
            device=self.device
        )
        
        # Sample noise
        noise = torch.randn(
            image_or_video_shape,
            device=self.device,
            dtype=torch.bfloat16
        )
        
        # Generate with callback pipeline
        output, denoised_timestep_from, denoised_timestep_to = \
            self.inference_pipeline.generate_chunk_with_cache(
                noise=noise,
                conditional_dict=conditional_dict,
                current_start_frame=0,
                requires_grad=True,
                switch_frame_index=switch_frame_index,
                switch_conditional_dict=switch_conditional_dict,
                callback_frame_index=callback_frame_index,
                callback_conditional_dict=callback_conditional_dict,
            )
        
        # Create gradient mask for callback segment
        if callback_frame_index is not None:
            gradient_mask = torch.zeros_like(output, dtype=torch.bool)
            # Only compute gradients for callback segment (after callback_frame_index)
            gradient_mask[:, callback_frame_index:] = True
            # Also optionally include the transition region
            if slice_last_frames > 0:
                start_idx = max(0, callback_frame_index - slice_last_frames)
                gradient_mask[:, start_idx:callback_frame_index] = True
        else:
            gradient_mask = None
        
        # Clear KV cache
        self.inference_pipeline.clear_kv_cache()
        
        return output, gradient_mask, denoised_timestep_from, denoised_timestep_to
