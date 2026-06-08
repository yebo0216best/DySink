#!/usr/bin/env python3
"""
Merge LoRA weights into base model using PEFT's native merge functionality.

This is the recommended way to merge LoRA - it uses PEFT's merge_and_unload()
which handles all the weight merging automatically.

Usage:
    python scripts/merge_lora_peft.py \
        --base_ckpt /path/to/base_model.pt \
        --lora_ckpt /path/to/lora.pt \
        --output_ckpt /path/to/merged_base.pt \
        --model_name Wan2.1-T2V-1.3B

Workflow for Stage 3 training:
    1. Train Stage 2: base.pt + fresh LoRA → stage2_lora.pt
    2. Merge: python scripts/merge_lora_peft.py ... → merged_stage2.pt
    3. Train Stage 3: merged_stage2.pt + fresh LoRA → stage3_lora.pt
    
    Now Stage 2 abilities are frozen in merged_stage2.pt,
    Stage 3 learns new abilities without forgetting Stage 2.
"""

import argparse
import torch
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_lora_config():
    """Get LoRA config matching the training configuration."""
    import peft
    
    target_modules = [
        "to_q", "to_k", "to_v", "to_out.0",
        "ff.0", "ff.2",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "head",
        "img_emb.proj.0", "img_emb.proj.2",
        "modulation",
        "patch_embedding",
    ]
    
    return peft.LoraConfig(
        r=256,
        lora_alpha=256,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA using PEFT's native functionality")
    parser.add_argument("--base_ckpt", type=str, required=True,
                        help="Path to base model checkpoint")
    parser.add_argument("--lora_ckpt", type=str, required=True,
                        help="Path to LoRA checkpoint")
    parser.add_argument("--output_ckpt", type=str, required=True,
                        help="Path to save merged checkpoint")
    parser.add_argument("--model_name", type=str, default="Wan2.1-T2V-1.3B",
                        help="Model name for loading architecture")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for merging (cpu recommended to save memory)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Merge LoRA to Base Model (PEFT Method)")
    print("=" * 60)
    print(f"Base checkpoint: {args.base_ckpt}")
    print(f"LoRA checkpoint: {args.lora_ckpt}")
    print(f"Output checkpoint: {args.output_ckpt}")
    print(f"Model: {args.model_name}")
    print()
    
    # Import required modules
    import peft
    from model.wan_wrapper import get_model
    from omegaconf import OmegaConf
    
    # Create minimal config for model loading
    config = OmegaConf.create({
        "generator_ckpt": args.base_ckpt,
        "fake_name": args.model_name,
        "real_name": args.model_name,
    })
    
    print("Loading base model...")
    # Load only generator (we only need to merge generator LoRA)
    model = get_model(config, mode="generator_only")
    generator = model.generator
    
    # Load base weights
    base_ckpt = torch.load(args.base_ckpt, map_location=args.device)
    if "model" in base_ckpt:
        generator.model.load_state_dict(base_ckpt["model"], strict=False)
    else:
        generator.model.load_state_dict(base_ckpt, strict=False)
    print(f"  Base model loaded")
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    lora_config = get_lora_config()
    generator.model = peft.get_peft_model(generator.model, lora_config)
    print(f"  LoRA applied to model")
    
    # Load LoRA weights
    print("Loading LoRA weights...")
    lora_ckpt = torch.load(args.lora_ckpt, map_location=args.device)
    
    if "generator_lora" in lora_ckpt:
        peft.set_peft_model_state_dict(generator.model, lora_ckpt["generator_lora"])
        print(f"  Loaded {len(lora_ckpt['generator_lora'])} LoRA parameters")
    else:
        raise ValueError(f"No 'generator_lora' key in checkpoint. Keys: {list(lora_ckpt.keys())}")
    
    if "step" in lora_ckpt:
        print(f"  LoRA from training step: {lora_ckpt['step']}")
    
    # Merge LoRA into base model
    print("\nMerging LoRA weights into base model...")
    generator.model = generator.model.merge_and_unload()
    print("  Merge complete!")
    
    # Get merged state dict
    merged_state_dict = generator.model.state_dict()
    print(f"  Merged model has {len(merged_state_dict)} parameters")
    
    # Prepare output checkpoint
    output_ckpt = {
        "model": merged_state_dict,
        "_merged_info": {
            "base_ckpt": args.base_ckpt,
            "lora_ckpt": args.lora_ckpt,
            "merged_from_step": lora_ckpt.get("step", "unknown"),
            "merge_method": "peft_merge_and_unload",
        }
    }
    
    # Save
    print(f"\nSaving merged checkpoint to {args.output_ckpt}...")
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save(output_ckpt, args.output_ckpt)
    
    file_size = os.path.getsize(args.output_ckpt) / (1024 ** 3)
    print(f"  Saved ({file_size:.2f} GB)")
    
    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)
    print(f"\nTo use for Stage 3 training:")
    print(f"  generator_ckpt: {args.output_ckpt}")
    print(f"  # Do NOT specify lora_ckpt - fresh LoRA will be initialized")
    print(f"\nOr use the prepared config:")
    print(f"  configs/longlive_train_long_callback_fresh_lora.yaml")


if __name__ == "__main__":
    main()
