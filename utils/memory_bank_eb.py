"""
Memory Bank for LongLive with Memory KV Cache - PE-Core Vision Encoder Version

This module implements a training-free memory bank mechanism that stores
KV cache entries with their corresponding visual features for retrieval.
Uses PE-Core vision encoder for visual feature extraction (no text encoder).

Key Features:
- Stores KV cache and visual features on CPU to minimize GPU memory usage
- Uses PE-Core VisionTransformer for efficient visual feature extraction
- Uses cosine similarity for visual-only feature retrieval
- Supports top-k retrieval of similar frames
- Lightweight: no text encoder, no temp file I/O
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import MemoryEntry and MemoryBank from the base module to reuse
from utils.memory_bank import MemoryEntry, MemoryBank, extract_kv_cache_for_frame, merge_retrieval_kv_cache


class PECoreMemoryEncoder(nn.Module):
    """
    Memory Encoder using PE-Core VisionTransformer (visual-only, no text encoder).
    
    Uses the vision encoder from lcm_pe to extract per-frame visual features,
    then mean-pools across frames to get a single block-level feature.
    
    Compared to multimodal memory encoders:
    - Visual-only: no text encoder, no multimodal encoding
    - No temp file I/O: processes tensors directly
    - Lightweight and fast
    - Output dim depends on PE-Core model variant
    
    Available PE-Core models:
        PE-Core-B16-224: output_dim=1024, image_size=224
        PE-Core-S16-384: output_dim=512,  image_size=384
        PE-Core-L14-336: output_dim=1024, image_size=336
        PE-Core-T16-384: output_dim=512,  image_size=384
    
    Usage:
        encoder = PECoreMemoryEncoder(
            pe_config="PE-Core-B16-224",
            checkpoint_path="/path/to/PE-Core-B16-224/PE-Core-B16-224.pt"
        )
        
        # Encode video frames (visual-only)
        features = encoder.encode_frames(frames)  # [B, output_dim]
    """
    
    def __init__(
        self,
        pe_config: str = "PE-Core-B16-224",
        checkpoint_path: Optional[str] = None,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the PE-Core Memory Encoder.
        
        Args:
            pe_config: PE-Core model config name (e.g. "PE-Core-B16-224")
            checkpoint_path: Path to the PE-Core checkpoint file (.pt).
                            If None, loads from HuggingFace via fetch_pe_checkpoint.
            device: Device to run the encoder on
            dtype: Data type for computation
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.pe_config_name = pe_config
        
        # Add lcm_pe to sys.path so we can import from core.*
        lcm_pe_dir = str(Path(__file__).parent.parent / "lcm_pe")
        if lcm_pe_dir not in sys.path:
            sys.path.insert(0, lcm_pe_dir)
        
        from core.vision_encoder.pe import VisionTransformer
        from core.vision_encoder.config import PE_VISION_CONFIG
        
        if pe_config not in PE_VISION_CONFIG:
            raise ValueError(
                f"Unknown PE-Core config '{pe_config}'. "
                f"Available: {list(PE_VISION_CONFIG.keys())}"
            )
        
        print(f"[PECoreMemoryEncoder] Loading {pe_config} model")
        if checkpoint_path:
            print(f"[PECoreMemoryEncoder] Checkpoint: {checkpoint_path}")
        
        self.model = VisionTransformer.from_config(
            pe_config,
            pretrained=True,
            checkpoint_path=checkpoint_path,
        )
        self.model.eval()
        self.model.to(device=device, dtype=dtype)
        
        # Cache model properties
        self.image_size = self.model.image_size
        self._feature_dim = self.model.output_dim
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"[PECoreMemoryEncoder] Loaded successfully: "
              f"image_size={self.image_size}, output_dim={self._feature_dim}")
    
    @property
    def feature_dim(self) -> int:
        """Get the feature dimension of the model."""
        return self._feature_dim
    
    @torch.no_grad()
    def encode_frames(
        self,
        frames: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode frames and return the mean feature.
        
        Args:
            frames: Input frames, shape [num_frames, C, H, W] or [B, num_frames, C, H, W]
                   Values should be in range [0, 1]
            normalize: Whether to L2 normalize the output features
            
        Returns:
            Mean feature across all frames, shape [feature_dim] or [B, feature_dim]
        """
        if frames.dim() == 4:
            # [num_frames, C, H, W] -> [1, num_frames, C, H, W]
            frames = frames.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        batch_size, num_frames, c, h, w = frames.shape
        
        # Resize if needed
        if h != self.image_size or w != self.image_size:
            frames_resized = F.interpolate(
                frames.reshape(-1, c, h, w),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).reshape(batch_size, num_frames, c, self.image_size, self.image_size)
        else:
            frames_resized = frames
        
        # Convert from [0, 1] to [-1, 1] (PE uses Normalize([0.5]*3, [0.5]*3))
        frames_normalized = frames_resized * 2.0 - 1.0
        
        # Move to device
        frames_normalized = frames_normalized.to(device=self.device, dtype=self.dtype)
        
        # Flatten batch and frame dims: [B * N, C, H, W]
        flat_frames = frames_normalized.reshape(-1, c, self.image_size, self.image_size)
        
        # Forward through VisionTransformer -> [B * N, output_dim]
        frame_features = self.model(flat_frames)
        
        # Reshape back: [B, N, output_dim]
        frame_features = frame_features.reshape(batch_size, num_frames, -1)
        
        # Mean-pool across frames: [B, output_dim]
        features = frame_features.mean(dim=1)
        
        if normalize:
            features = F.normalize(features, dim=-1)
        
        if squeeze_batch:
            features = features.squeeze(0)  # [feature_dim]
        
        return features
    
    @torch.no_grad()
    def encode_frames_with_prompt(
        self,
        frames: torch.Tensor,
        text_prompt: Optional[str] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode frames to a visual feature (text_prompt is ignored).
        
        This method accepts text_prompt for API compatibility with callers that
        pass both frames and prompts.
        Since PE-Core is visual-only, the text_prompt parameter is accepted but ignored.
        
        Args:
            frames: Input frames, shape [num_frames, C, H, W] or [B, num_frames, C, H, W]
                   Values should be in range [0, 1]
            text_prompt: Ignored (kept for API compatibility)
            normalize: Whether to L2 normalize the output features
            
        Returns:
            Visual feature vector, shape [feature_dim] or [B, feature_dim]
        """
        return self.encode_frames(frames, normalize=normalize)
