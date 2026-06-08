# LongLive Interactive Inference with Memory KV Cache (PE-Core Vision Encoder Version)
#
# This script implements interactive inference with memory bank mechanism.
# Uses PE-Core VisionTransformer for visual-only feature extraction.
#
# It supports:
# 1. Multiple prompt segments with switching at specified frame indices
# 2. Memory bank for KV cache storage and retrieval based on visual similarity
# 3. KV recache when switching prompts - recalculates KV cache for historical
#    frames with new prompt and updates memory bank
#
# Key features:
# - Visual-only block-level encoding (no text encoder)
# - PE-Core VisionTransformer for efficient visual feature extraction
# - Cosine similarity for KV cache retrieval
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import List

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from torchvision import transforms  # noqa: F401
from einops import rearrange

from utils.misc import set_seed
from utils.distributed import barrier
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

from pipeline.interactive_memory_causal_inference_eb import (
    InteractiveMemoryCausalInferencePipelineEB,
)
from utils.dataset import MultiTextDataset


def main():
    # ----------------------------- Argument parsing -----------------------------
    parser = argparse.ArgumentParser("Interactive inference with memory (PE-Core Vision Encoder)")
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    # ----------------------------- Distributed setup -----------------------------
    if "LOCAL_RANK" in os.environ:
        os.environ["NCCL_CROSS_NIC"] = "1"
        os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
        os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")
        
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", str(local_rank)))
        
        # Set device first
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        # Initialize process group with backend and timeout
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.constants.default_pg_timeout
            )
        
        set_seed(config.seed + local_rank)
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda")
        set_seed(config.seed)
        print(f"Single GPU mode on device {device}")

    print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
    low_memory = get_cuda_free_memory_gb(device) < 40
    torch.set_grad_enabled(False)

    # Memory encoder configuration (PE-Core Vision Encoder, visual-only)
    memory_encoder_config = {
        "pe_config": getattr(config, "memory_encoder_pe_config", "PE-Core-B16-224"),
        "checkpoint_path": getattr(config, "memory_encoder_pe_checkpoint", None),
    }

    # Initialize pipeline with memory support (PE-Core version)
    pipeline = InteractiveMemoryCausalInferencePipelineEB(
        config,
        device=device,
        memory_encoder_config=memory_encoder_config
    )

    # Load generator checkpoint
    if config.generator_ckpt:
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]

        if config.use_ema:
            def _clean_key(name: str) -> str:
                return name.replace("_fsdp_wrapped_module.", "")

            cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
            missing, unexpected = pipeline.generator.load_state_dict(
                cleaned_state_dict, strict=False
            )
            if local_rank == 0:
                if missing:
                    print(f"[Warning] {len(missing)} parameters missing: {missing[:8]} ...")
                if unexpected:
                    print(f"[Warning] {len(unexpected)} unexpected params: {unexpected[:8]} ...")
        else:
            pipeline.generator.load_state_dict(raw_gen_state_dict)

    # --------------------------- LoRA support (optional, multi-stage merge) ---------------------------
    from utils.lora_utils import configure_lora_for_model
    import peft

    def _get_lora_ckpt_paths_with_configs(cfg) -> list[tuple[str, dict]]:
        """Get LoRA checkpoint paths with their corresponding adapter configs.
        
        Returns:
            List of (path, adapter_config) tuples. Each stage can have its own
            adapter config (e.g., different alpha values).
        """
        results: list[tuple[str, dict]] = []
        default_adapter = getattr(cfg, "adapter", None)
        
        # Check for stage-specific paths and configs
        stage2_path = getattr(cfg, "lora_ckpt_stage2", None)
        stage3_path = getattr(cfg, "lora_ckpt_stage3", None)
        
        # Stage-specific adapter configs (fallback to default adapter)
        adapter_stage2 = getattr(cfg, "adapter_stage2", None) or default_adapter
        adapter_stage3 = getattr(cfg, "adapter_stage3", None) or default_adapter
        
        if stage2_path:
            results.append((stage2_path, adapter_stage2))
        if stage3_path:
            results.append((stage3_path, adapter_stage3))

        # Backward compatibility: lora_ckpt_list
        if not results:
            raw_list = getattr(cfg, "lora_ckpt_list", None)
            if raw_list:
                if isinstance(raw_list, str):
                    paths = [p.strip() for p in raw_list.split(",") if p.strip()]
                elif isinstance(raw_list, (list, tuple)):
                    paths = [str(p) for p in raw_list if str(p).strip()]
                else:
                    paths = []
                for p in paths:
                    results.append((p, default_adapter))
        
        # Backward compatibility: single LoRA path
        if not results:
            single = getattr(cfg, "lora_ckpt", None)
            if single:
                results.append((single, default_adapter))

        return [(p, c) for p, c in results if p]

    pipeline.is_lora_enabled = False
    lora_items = _get_lora_ckpt_paths_with_configs(config)
    if lora_items:
        if not getattr(config, "adapter", None) or configure_lora_for_model is None:
            if local_rank == 0:
                print("[Warning] LoRA checkpoints provided but adapter config is missing.")
        else:
            if local_rank == 0:
                print(f"Default LoRA config: {config.adapter}")
                print(f"Applying and merging {len(lora_items)} LoRA checkpoint(s)...")

            for idx, (lora_ckpt_path, stage_adapter_config) in enumerate(lora_items, start=1):
                if local_rank == 0:
                    print(f"[LoRA] Stage {idx}/{len(lora_items)}: {lora_ckpt_path}")
                    if stage_adapter_config != config.adapter:
                        print(f"       Using stage-specific config: rank={stage_adapter_config.get('rank')}, alpha={stage_adapter_config.get('alpha')}")

                pipeline.generator.model = configure_lora_for_model(
                    pipeline.generator.model,
                    model_name="generator",
                    lora_config=stage_adapter_config,
                    is_main_process=(local_rank == 0),
                )

                lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
                if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                    peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
                else:
                    peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)

                if hasattr(pipeline.generator.model, "merge_and_unload"):
                    pipeline.generator.model = pipeline.generator.model.merge_and_unload()
                else:
                    pipeline.generator.model = peft.PeftModel.merge_and_unload(pipeline.generator.model)

            pipeline.is_lora_enabled = True

    # Move pipeline to appropriate dtype and device
    print("dtype", pipeline.generator.model.dtype)
    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    # ----------------------------- Build dataset -----------------------------
    # Parse switch_frame_indices
    if isinstance(config.switch_frame_indices, int):
        switch_frame_indices: List[int] = [int(config.switch_frame_indices)]
    else:
        switch_frame_indices: List[int] = [
            int(x) for x in str(config.switch_frame_indices).split(",") if str(x).strip()
        ]

    # Create dataset
    dataset = MultiTextDataset(config.data_path)

    # Validate number of segments & switch_frame_indices length
    num_segments = len(dataset[0]["prompts_list"])
    assert len(switch_frame_indices) == num_segments - 1, (
        "The number of switch_frame_indices should be the number of prompt segments minus 1"
    )

    print("Number of segments:", num_segments)
    print("Switch frame indices:", switch_frame_indices)

    num_prompts_total = len(dataset)
    print(f"Number of prompt lines: {num_prompts_total}")

    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

    # Create output directory
    if local_rank == 0:
        os.makedirs(config.output_folder, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    # Memory retrieval configuration logging
    use_memory = getattr(config, "use_memory_retrieval", True)
    memory_top_k = getattr(config, "memory_top_k", 3)

    if local_rank == 0:
        print("=" * 70)
        print("LongLive Interactive Inference with Memory KV Cache (PE-Core Vision Encoder)")
        print("=" * 70)
        print(f"  Memory Retrieval: {'Enabled' if use_memory else 'Disabled'}")
        if use_memory:
            print(f"  Top-K Retrieval: {memory_top_k}")
            print(f"  Retrieval Mode: Visual-only (PE-Core)")
            print(f"  Memory Encoder: {memory_encoder_config['pe_config']}")
            print(f"  Checkpoint: {memory_encoder_config['checkpoint_path']}")
        print(f"  Prompt Segments: {num_segments}")
        print(f"  Switch Frame Indices: {switch_frame_indices}")
        print("=" * 70)

    # ----------------------------- Inference loop -----------------------------
    for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
        idx = batch_data["idx"].item()
        prompts_list: List[str] = batch_data["prompts_list"]  # type: ignore

        sampled_noise = torch.randn(
            [
                config.num_samples,
                config.num_output_frames,
                16,
                60,
                104,
            ],
            device=device,
            dtype=torch.bfloat16,
        )

        video = pipeline.inference(
            noise=sampled_noise,
            text_prompts_list=prompts_list,
            switch_frame_indices=switch_frame_indices,
            return_latents=False,
            low_memory=low_memory,
        )

        current_video = rearrange(video, "b t c h w -> b t h w c").cpu() * 255.0

        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        # Determine model type for filename
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora_interactive_memory_eb"
        elif getattr(config, 'use_ema', False):
            model_type = "ema_interactive_memory_eb"
        else:
            model_type = "interactive_memory_eb"

        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                output_path = os.path.join(config.output_folder, f"rank{rank}-{idx}-{seed_idx}_{model_type}.mp4")
            else:
                # Use the first prompt segment as the filename prefix to avoid overly long names
                short_name = prompts_list[0][0][:100].replace("/", "_")
                output_path = os.path.join(config.output_folder, f"rank{rank}-{short_name}-{seed_idx}_{model_type}.mp4")
            
            write_video(output_path, current_video[seed_idx].to(torch.uint8), fps=16)
            if local_rank == 0:
                print(f"Saved video to {output_path}")

        # Clear VAE cache
        pipeline.vae.model.clear_cache()

        if config.inference_iter != -1 and i >= config.inference_iter:
            break

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
