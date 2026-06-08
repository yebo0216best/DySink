#!/usr/bin/env python3
"""
Merge LoRA weights into base model to create a new merged checkpoint.

This script merges Stage N LoRA weights into the base model, creating a new
base checkpoint that can be used for Stage N+1 training with a fresh LoRA.

Usage:
    python scripts/merge_lora_to_base.py \
        --base_ckpt /path/to/base_model.pt \
        --lora_ckpt /path/to/lora.pt \
        --output_ckpt /path/to/merged_base.pt \
        --lora_rank 256 \
        --lora_alpha 256

Why use this?
    - Prevents catastrophic forgetting when training multiple stages
    - Stage 2 abilities are "baked into" the merged base, won't be overwritten
    - Stage 3 trains a fresh LoRA on top of the merged base
    
Workflow:
    Stage 1: base.pt → train → stage1_lora.pt
    Merge:   base.pt + stage1_lora.pt → merged_stage1.pt
    
    Stage 2: merged_stage1.pt → train → stage2_lora.pt  
    Merge:   merged_stage1.pt + stage2_lora.pt → merged_stage2.pt
    
    Stage 3: merged_stage2.pt → train with fresh LoRA → stage3_lora.pt
"""

import argparse
import torch
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_lora_target_modules():
    """Get the target modules for LoRA (same as training config)."""
    return [
        "to_q", "to_k", "to_v", "to_out.0",
        "ff.0", "ff.2",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "head",
        "img_emb.proj.0", "img_emb.proj.2",
        "modulation",
        "patch_embedding",
    ]


def extract_base_key_from_lora_key(lora_key: str) -> str:
    """
    Extract the base model key from a LoRA key.
    
    PEFT LoRA keys have formats like:
    - base_model.model.xxx.lora_A.default.weight
    - base_model.model.xxx.lora_A.weight
    
    We need to extract 'xxx.weight' (the original base model key).
    """
    # Remove lora_A/lora_B and related parts
    key = lora_key
    
    # Remove PEFT prefixes
    key = key.replace("base_model.model.", "")
    key = key.replace("base_model.", "")
    
    # Remove lora_A/lora_B parts
    # Format: xxx.lora_A.default.weight or xxx.lora_A.weight
    if ".lora_A.default.weight" in key:
        key = key.replace(".lora_A.default.weight", ".weight")
    elif ".lora_A.weight" in key:
        key = key.replace(".lora_A.weight", ".weight")
    elif ".lora_B.default.weight" in key:
        key = key.replace(".lora_B.default.weight", ".weight")
    elif ".lora_B.weight" in key:
        key = key.replace(".lora_B.weight", ".weight")
    
    return key


def merge_lora_weights(base_state_dict: dict, lora_state_dict: dict, 
                       lora_alpha: float, lora_rank: int) -> dict:
    """
    Merge LoRA weights into base model weights.
    
    LoRA formula: W' = W + (alpha/rank) * B @ A
    Where:
        - W is the original weight
        - A is lora_A (low-rank down projection)  
        - B is lora_B (low-rank up projection)
        - alpha/rank is the scaling factor
    
    Args:
        base_state_dict: Base model state dict
        lora_state_dict: LoRA checkpoint (contains 'generator_lora' key)
        lora_alpha: LoRA alpha value
        lora_rank: LoRA rank value
        
    Returns:
        Merged state dict
    """
    scaling = lora_alpha / lora_rank
    
    # Get LoRA weights
    if "generator_lora" in lora_state_dict:
        lora_weights = lora_state_dict["generator_lora"]
    else:
        lora_weights = lora_state_dict
    
    # Create a copy of base state dict
    merged_state_dict = {k: v.clone() for k, v in base_state_dict.items()}
    
    # Build a mapping from base keys to their variations
    # This helps match LoRA keys to base keys more reliably
    base_key_mapping = {}
    for key in base_state_dict.keys():
        # Store original key
        base_key_mapping[key] = key
        # Also store without 'model.' prefix if present
        if key.startswith("model."):
            base_key_mapping[key[6:]] = key
    
    # Find all LoRA A/B pairs
    lora_a_keys = sorted([k for k in lora_weights.keys() if "lora_A" in k])
    
    merged_count = 0
    skipped_count = 0
    skipped_keys = []
    
    for lora_a_key in lora_a_keys:
        # Get corresponding lora_B key
        lora_b_key = lora_a_key.replace("lora_A", "lora_B")
        
        if lora_b_key not in lora_weights:
            print(f"  Warning: No matching lora_B for {lora_a_key}")
            continue
        
        # Extract the base model key
        base_key = extract_base_key_from_lora_key(lora_a_key)
        
        # Try to find the matching base weight
        found_base_key = None
        
        # Direct match
        if base_key in base_key_mapping:
            found_base_key = base_key_mapping[base_key]
        # Try with 'model.' prefix
        elif "model." + base_key in base_key_mapping:
            found_base_key = base_key_mapping["model." + base_key]
        # Try without 'model.' prefix
        elif base_key.startswith("model.") and base_key[6:] in base_key_mapping:
            found_base_key = base_key_mapping[base_key[6:]]
        
        if found_base_key is None:
            skipped_count += 1
            skipped_keys.append(base_key)
            continue
        
        # Get weights
        lora_a = lora_weights[lora_a_key]
        lora_b = lora_weights[lora_b_key]
        base_weight = merged_state_dict[found_base_key]
        
        # Compute merged weight: W' = W + scaling * B @ A
        # lora_a: [rank, in_features]
        # lora_b: [out_features, rank]
        # delta: [out_features, in_features]
        try:
            delta = scaling * (lora_b.float() @ lora_a.float())
            
            # Handle shape mismatches (e.g., conv layers might need reshaping)
            if delta.shape != base_weight.shape:
                if delta.numel() == base_weight.numel():
                    delta = delta.view(base_weight.shape)
                else:
                    print(f"  Warning: Shape mismatch for {found_base_key}: "
                          f"delta {delta.shape} vs base {base_weight.shape}")
                    skipped_count += 1
                    continue
            
            merged_state_dict[found_base_key] = base_weight.float() + delta
            merged_state_dict[found_base_key] = merged_state_dict[found_base_key].to(base_weight.dtype)
            merged_count += 1
            
        except Exception as e:
            print(f"  Error merging {lora_a_key}: {e}")
            skipped_count += 1
            continue
    
    print(f"  Successfully merged {merged_count} LoRA weight pairs")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} pairs (no matching base weight)")
        if skipped_count <= 10:
            for key in skipped_keys:
                print(f"    - {key}")
        else:
            for key in skipped_keys[:5]:
                print(f"    - {key}")
            print(f"    ... and {skipped_count - 5} more")
    
    return merged_state_dict


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base_ckpt", type=str, required=True,
                        help="Path to base model checkpoint")
    parser.add_argument("--lora_ckpt", type=str, required=True,
                        help="Path to LoRA checkpoint")
    parser.add_argument("--output_ckpt", type=str, required=True,
                        help="Path to save merged checkpoint")
    parser.add_argument("--lora_rank", type=int, default=256,
                        help="LoRA rank (default: 256)")
    parser.add_argument("--lora_alpha", type=int, default=256,
                        help="LoRA alpha (default: 256)")
    parser.add_argument("--merge_critic", action="store_true",
                        help="Also merge critic LoRA weights")
    
    args = parser.parse_args()
    
    print(f"=" * 60)
    print("Merge LoRA to Base Model")
    print(f"=" * 60)
    print(f"Base checkpoint: {args.base_ckpt}")
    print(f"LoRA checkpoint: {args.lora_ckpt}")
    print(f"Output checkpoint: {args.output_ckpt}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"Scaling factor: {args.lora_alpha / args.lora_rank}")
    print()
    
    # Load checkpoints
    print("Loading base checkpoint...")
    base_ckpt = torch.load(args.base_ckpt, map_location="cpu")
    
    # Handle different checkpoint formats
    if "model" in base_ckpt:
        base_state_dict = base_ckpt["model"]
        ckpt_format = "dict"
    elif "state_dict" in base_ckpt:
        base_state_dict = base_ckpt["state_dict"]
        ckpt_format = "dict"
    else:
        base_state_dict = base_ckpt
        ckpt_format = "raw"
    
    print(f"  Base model has {len(base_state_dict)} parameters")
    
    print("Loading LoRA checkpoint...")
    lora_ckpt = torch.load(args.lora_ckpt, map_location="cpu")
    
    if "generator_lora" in lora_ckpt:
        print(f"  Generator LoRA has {len(lora_ckpt['generator_lora'])} parameters")
    if "critic_lora" in lora_ckpt:
        print(f"  Critic LoRA has {len(lora_ckpt['critic_lora'])} parameters")
    if "step" in lora_ckpt:
        print(f"  LoRA checkpoint step: {lora_ckpt['step']}")
    
    # Merge generator LoRA
    print("\nMerging generator LoRA weights...")
    merged_state_dict = merge_lora_weights(
        base_state_dict, 
        lora_ckpt,
        lora_alpha=args.lora_alpha,
        lora_rank=args.lora_rank
    )
    
    # Prepare output checkpoint
    if ckpt_format == "dict":
        output_ckpt = base_ckpt.copy()
        if "model" in output_ckpt:
            output_ckpt["model"] = merged_state_dict
        else:
            output_ckpt["state_dict"] = merged_state_dict
        # Remove step info (this is a new base, not a training checkpoint)
        if "step" in output_ckpt:
            del output_ckpt["step"]
    else:
        output_ckpt = merged_state_dict
    
    # Add metadata
    output_ckpt["_merged_info"] = {
        "base_ckpt": args.base_ckpt,
        "lora_ckpt": args.lora_ckpt,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "merged_from_step": lora_ckpt.get("step", "unknown"),
    }
    
    # Save
    print(f"\nSaving merged checkpoint to {args.output_ckpt}...")
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save(output_ckpt, args.output_ckpt)
    
    # Print file size
    file_size = os.path.getsize(args.output_ckpt) / (1024 ** 3)
    print(f"  Saved ({file_size:.2f} GB)")
    
    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Update your Stage 3 config to use the merged base:")
    print(f"   generator_ckpt: {args.output_ckpt}")
    print(f"   # Remove or comment out lora_ckpt")
    print(f"   # lora_ckpt: ...")
    print(f"2. The Stage 3 training will start with a fresh LoRA")
    print(f"3. Stage 2 abilities are now baked into the base model")


if __name__ == "__main__":
    main()
