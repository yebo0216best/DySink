# LongLive with Memory KV Cache - Inference Script (PE-Core Vision Encoder Version)
# 
# This script implements a training-free memory bank mechanism for LongLive
# that retrieves historical KV caches based on visual feature similarity.
# Uses PE-Core VisionTransformer for visual-only feature extraction.
#
# Key features:
# - Visual-only block-level encoding (no text encoder)
# - PE-Core VisionTransformer for efficient visual feature extraction
# - Cosine similarity for KV cache retrieval
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline.memory_causal_inference_eb import MemoryCausalInferencePipelineEB
from utils.dataset import TextDataset
from utils.misc import set_seed
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, log_gpu_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    # Initialize distributed inference
    if "LOCAL_RANK" in os.environ:
        os.environ["NCCL_CROSS_NIC"] = "1"
        os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
        os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", str(local_rank)))

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.constants.default_pg_timeout,
            )
        set_seed(config.seed + local_rank)
        config.distributed = True
        if rank == 0:
            print(f"[Rank {rank}] Initialized distributed processing on device {device}")
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda")
        set_seed(config.seed)
        config.distributed = False
        print(f"Single GPU mode on device {device}")

    print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
    low_memory = get_cuda_free_memory_gb(device) < 40
    low_memory = True

    torch.set_grad_enabled(False)

    # Memory encoder configuration (PE-Core Vision Encoder, visual-only)
    memory_encoder_config = {
        "pe_config": getattr(config, "memory_encoder_pe_config", "PE-Core-B16-224"),
        "checkpoint_path": getattr(config, "memory_encoder_pe_checkpoint", None),
    }

    # Initialize pipeline with memory support (PE-Core version)
    pipeline = MemoryCausalInferencePipelineEB(
        config, 
        device=device,
        memory_encoder_config=memory_encoder_config
    )

    # Load generator checkpoint
    if config.generator_ckpt:
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        if "generator" in state_dict or "generator_ema" in state_dict:
            raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
        elif "model" in state_dict:
            raw_gen_state_dict = state_dict["model"]
        else:
            raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
        
        if config.use_ema:
            def _clean_key(name: str) -> str:
                name = name.replace("_fsdp_wrapped_module.", "")
                return name

            cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
            missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
            if local_rank == 0:
                if len(missing) > 0:
                    print(f"[Warning] {len(missing)} parameters are missing: {missing[:8]} ...")
                if len(unexpected) > 0:
                    print(f"[Warning] {len(unexpected)} unexpected parameters: {unexpected[:8]} ...")
        else:
            pipeline.generator.load_state_dict(raw_gen_state_dict)

    # LoRA support (optional, supports multi-stage merge)
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

                # Apply LoRA modules, load weights, then merge into base
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

                # Merge LoRA into the base model and unload adapters
                if hasattr(pipeline.generator.model, "merge_and_unload"):
                    pipeline.generator.model = pipeline.generator.model.merge_and_unload()
                else:
                    pipeline.generator.model = peft.PeftModel.merge_and_unload(pipeline.generator.model)

            pipeline.is_lora_enabled = True

    # Move pipeline to appropriate dtype and device
    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    # Load dataset
    extended_prompt_path = config.data_path
    dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=extended_prompt_path)
    num_prompts = len(dataset)
    print(f"Number of prompts: {num_prompts}")

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
        print("=" * 60)
        print("LongLive with Memory KV Cache (PE-Core Vision Encoder)")
        print("=" * 60)
        print(f"  Memory Retrieval: {'Enabled' if use_memory else 'Disabled'}")
        if use_memory:
            print(f"  Top-K Retrieval: {memory_top_k}")
            print(f"  Retrieval Mode: Visual-only (PE-Core)")
            print(f"  Memory Encoder: {memory_encoder_config['pe_config']}")
            print(f"  Checkpoint: {memory_encoder_config['checkpoint_path']}")
        print("=" * 60)

    # Main inference loop
    for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
        idx = batch_data['idx'].item()

        if isinstance(batch_data, dict):
            batch = batch_data
        elif isinstance(batch_data, list):
            batch = batch_data[0]

        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * config.num_samples
        else:
            prompts = [prompt] * config.num_samples
        sample_batch_size = int(getattr(config, "sample_batch_size", config.num_samples))
        if sample_batch_size <= 0:
            sample_batch_size = 1

        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        # Save the video
        if idx < num_prompts:
            if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
                model_type = "lora_memory_eb"
            elif getattr(config, 'use_ema', False):
                model_type = "ema_memory_eb"
            else:
                model_type = "memory_eb"

            for start in range(0, config.num_samples, sample_batch_size):
                end = min(start + sample_batch_size, config.num_samples)
                batch_prompts = prompts[start:end]

                sampled_noise = torch.randn(
                    [end - start, config.num_output_frames, 16, 60, 104],
                    device=device, dtype=torch.bfloat16
                )

                print("sampled_noise.device", sampled_noise.device)
                print("prompts", batch_prompts)

                # Generate video with memory-augmented inference (PE-Core version)
                video, latents = pipeline.inference(
                    noise=sampled_noise,
                    text_prompts=batch_prompts,
                    return_latents=True,
                    low_memory=low_memory,
                    profile=False,
                )

                current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
                video = 255.0 * current_video

                # Clear VAE cache
                pipeline.vae.model.clear_cache()

                for local_idx in range(end - start):
                    seed_idx = start + local_idx
                    if config.save_with_index:
                        output_path = os.path.join(
                            config.output_folder,
                            f'rank{rank}-{idx}-{seed_idx}_{model_type}.mp4'
                        )
                    else:
                        output_path = os.path.join(
                            config.output_folder,
                            f'rank{rank}-{prompt[:100]}-{seed_idx}.mp4'
                        )
                    write_video(output_path, video[local_idx], fps=16)
                    if local_rank == 0:
                        print(f"Saved video to {output_path}")

                del video, latents, sampled_noise, current_video
                torch.cuda.empty_cache()

        if config.inference_iter != -1 and i >= config.inference_iter:
            break

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
