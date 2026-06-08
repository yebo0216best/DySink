"""
Memory Bank for LongLive with Memory KV Cache

This module implements a training-free memory bank mechanism that stores
KV cache entries with their corresponding visual features for retrieval.

Key Features:
- Stores KV cache and visual features on CPU to minimize GPU memory usage
- Uses cosine similarity for feature retrieval
- Supports top-k retrieval of similar frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class MemoryEntry:
    """A single memory entry containing KV cache and its corresponding feature."""
    frame_idx: int  # The frame index this entry corresponds to
    feature: torch.Tensor  # Shape: [feature_dim], stored on CPU
    text_feature: Optional[torch.Tensor]  # Shape: [feature_dim], stored on CPU if available
    kv_cache: List[Dict[str, torch.Tensor]]  # KV cache for all transformer blocks, stored on CPU
    latent: Optional[torch.Tensor] = None  # Block latent frames for KV recache, stored on CPU


class MemoryBank:
    """
    Memory Bank for storing and retrieving KV cache based on visual features.
    
    This implements a training-free retrieval mechanism where:
    1. Each generated frame's latent is decoded to 4 real images (due to 4x temporal compression)
    2. These 4 images are encoded using PEWithAdapter (memory encoder)
    3. The mean of the 4 features is stored along with the corresponding KV cache
    4. When generating a new frame, we retrieve similar frames using the previous frame's feature
    """
    
    def __init__(
        self,
        feature_dim: int = 1024,
        top_k: int = 3,
        max_entries: Optional[int] = None,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize the Memory Bank.
        
        Args:
            feature_dim: Dimension of the visual features
            top_k: Number of top similar entries to retrieve
            max_entries: Maximum number of entries to store (None for unlimited)
            device: Device to store features and KV cache (should be CPU for memory efficiency)
        """
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.max_entries = max_entries
        self.device = device
        
        self.entries: List[MemoryEntry] = []
        self.feature_matrix: Optional[torch.Tensor] = None  # [num_entries, feature_dim]

    @staticmethod
    def _to_1d_feature(feature: torch.Tensor) -> torch.Tensor:
        """
        Normalize feature to a 1D vector.
        If input is batched (e.g., [B, D]), reduce across batch with mean.
        """
        feature_cpu = feature.detach().cpu().float()
        if feature_cpu.dim() == 1:
            return F.normalize(feature_cpu, dim=-1)
        if feature_cpu.dim() == 2:
            if feature_cpu.size(0) == 1:
                feature_cpu = feature_cpu.squeeze(0)
            else:
                feature_cpu = feature_cpu.mean(dim=0)
            return F.normalize(feature_cpu, dim=-1)
        # Fallback: flatten to [N, D] then mean over N
        feature_cpu = feature_cpu.view(-1, feature_cpu.shape[-1]).mean(dim=0)
        return F.normalize(feature_cpu, dim=-1)
        
    def clear(self):
        """Clear all entries from the memory bank."""
        self.entries = []
        self.feature_matrix = None
        
    def add_entry(
        self,
        frame_idx: int,
        feature: torch.Tensor,
        kv_cache: List[Dict[str, torch.Tensor]],
        text_feature: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
    ):
        """
        Add a new entry to the memory bank.
        
        Args:
            frame_idx: The frame index
            feature: The visual feature tensor, shape [feature_dim]
            kv_cache: The KV cache for all transformer blocks
            text_feature: Optional text feature associated with this entry
            latent: Optional block latent frames for KV recache on prompt switch
        """
        # Ensure feature is on CPU and normalized to 1D
        # Input may be [feature_dim] or [1, feature_dim] or [B, feature_dim]
        feature_cpu = self._to_1d_feature(feature)
        text_feature_cpu = self._to_1d_feature(text_feature) if text_feature is not None else None
        latent_cpu = latent.detach().cpu().clone() if latent is not None else None
        
        # Deep copy KV cache to CPU
        kv_cache_cpu = []
        for block_cache in kv_cache:
            block_cache_cpu = {}
            for key, value in block_cache.items():
                if isinstance(value, torch.Tensor):
                    block_cache_cpu[key] = value.detach().cpu().clone()
                else:
                    block_cache_cpu[key] = value
            kv_cache_cpu.append(block_cache_cpu)
        
        # Create entry
        entry = MemoryEntry(
            frame_idx=frame_idx,
            feature=feature_cpu,
            text_feature=text_feature_cpu,
            kv_cache=kv_cache_cpu,
            latent=latent_cpu,
        )
        
        self.entries.append(entry)
        
        # Update feature matrix for efficient batch similarity computation
        if self.feature_matrix is None:
            self.feature_matrix = feature_cpu.unsqueeze(0)
        else:
            self.feature_matrix = torch.cat([
                self.feature_matrix,
                feature_cpu.unsqueeze(0)
            ], dim=0)
        
        # Optionally remove oldest entries if exceeding max_entries
        if self.max_entries is not None and len(self.entries) > self.max_entries:
            self.entries.pop(0)
            self.feature_matrix = self.feature_matrix[1:]
    
    def _build_exclusion_mask(
        self, exclude_frame_indices: Optional[List[int]]
    ) -> Optional[torch.Tensor]:
        """Build a boolean mask that is False for entries to exclude."""
        if exclude_frame_indices is None:
            return None
        mask = torch.ones(len(self.entries), dtype=torch.bool)
        for idx in exclude_frame_indices:
            for i, entry in enumerate(self.entries):
                if entry.frame_idx == idx:
                    mask[i] = False
        return mask

    def _compute_similarities(
        self,
        query_features: List[torch.Tensor],
        exclude_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mean cosine similarity from multiple query features.

        Args:
            query_features: List of query tensors, each [feature_dim].
            exclude_mask: Boolean mask (True = keep). Excluded entries get -inf.

        Returns:
            Averaged similarity scores [num_entries].
        """
        all_sims = []
        for qf in query_features:
            query_cpu = self._to_1d_feature(qf)
            sims = torch.matmul(self.feature_matrix, query_cpu)  # [num_entries]
            all_sims.append(sims)
        similarities = torch.stack(all_sims, dim=0).mean(dim=0)  # [num_entries]

        if exclude_mask is not None:
            similarities = similarities.masked_fill(~exclude_mask, float('-inf'))
        return similarities

    def retrieve(
        self,
        query_feature: torch.Tensor,
        exclude_frame_indices: Optional[List[int]] = None,
        query_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[MemoryEntry], torch.Tensor]:
        """
        Retrieve the most similar entries based on query feature(s).

        When *query_features* (a list) is provided, similarity is computed for
        each query independently and then **averaged** across queries before
        ranking.  This implements the multi-query retrieval strategy where all
        blocks in the local attention window contribute to the retrieval score.

        Args:
            query_feature: Single query feature tensor, shape [feature_dim].
                           Used when *query_features* is None.
            exclude_frame_indices: Frame indices to exclude from retrieval.
            query_features: Optional list of query feature tensors for
                            multi-query retrieval.  Takes precedence over
                            *query_feature* when provided.

        Returns:
            Tuple of (list of retrieved MemoryEntry objects, similarity scores)
        """
        if len(self.entries) == 0:
            return [], torch.tensor([])

        queries = query_features if query_features is not None else [query_feature]
        exclude_mask = self._build_exclusion_mask(exclude_frame_indices)
        similarities = self._compute_similarities(queries, exclude_mask)

        # Get top-k indices
        valid_mask = torch.isfinite(similarities)
        k = min(self.top_k, valid_mask.sum().item())
        if k == 0:
            return [], torch.tensor([])

        top_k_values, top_k_indices = torch.topk(similarities, k)

        # Filter out excluded entries represented by -inf scores.
        valid_results = torch.isfinite(top_k_values)
        top_k_values = top_k_values[valid_results]
        top_k_indices = top_k_indices[valid_results]

        # Retrieve entries
        retrieved_entries = [self.entries[idx.item()] for idx in top_k_indices]

        return retrieved_entries, top_k_values
    
    def retrieve_with_text(
        self,
        query_feature: torch.Tensor,
        text_feature: Optional[torch.Tensor] = None,
        text_weight: float = 0.5,
        exclude_frame_indices: Optional[List[int]] = None,
        text_text_weight: Optional[float] = None,
        return_score_breakdown: bool = False,
        query_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Any, ...]:
        """
        Retrieve the most similar entries based on visual + text-text similarity.

        Supports **multi-query** retrieval: when *query_features* (list) is
        provided, the visual similarity is computed for each query and averaged
        before combining with the text-text term.

        When text_text_weight is positive, the final similarity score is:
            combined = (visual + text_text_weight * tt) / (1 + text_text_weight)

        Args:
            query_feature: Single query visual feature tensor, shape [feature_dim].
                           Used when *query_features* is None.
            text_feature: Text prompt feature tensor, shape [feature_dim].
            text_weight: Backward-compatible fallback weight.
            exclude_frame_indices: Frame indices to exclude from retrieval.
            text_text_weight: Weight for text-text term (tt).
            return_score_breakdown: If True, also return component scores.
            query_features: Optional list of query visual features for
                            multi-query retrieval.

        Returns:
            If return_score_breakdown=False:
                (retrieved_entries, combined_scores)
            If return_score_breakdown=True:
                (retrieved_entries, combined_scores, visual_scores, tt_scores)
        """
        if len(self.entries) == 0:
            if return_score_breakdown:
                empty = torch.tensor([])
                return [], empty, empty, empty, empty
            return [], torch.tensor([])

        if text_text_weight is None:
            text_text_weight = text_weight

        queries = query_features if query_features is not None else [query_feature]
        exclude_mask = self._build_exclusion_mask(exclude_frame_indices)

        # Visual: mean similarity across all queries
        visual_similarities = self._compute_similarities(queries, exclude_mask)

        # Compute text-text term if query text feature is provided
        if text_feature is not None:
            text_cpu = self._to_1d_feature(text_feature)

            tt_vals = []
            for entry in self.entries:
                if entry.text_feature is None:
                    tt_vals.append(0.0)
                else:
                    tt_vals.append(torch.dot(entry.text_feature.float(), text_cpu).item())
            tt_similarities = torch.tensor(tt_vals, dtype=visual_similarities.dtype)
            if text_text_weight > 0 and exclude_mask is not None:
                tt_similarities = tt_similarities.masked_fill(~exclude_mask, float('-inf'))

            if text_text_weight > 0:
                combined_similarities = (
                    visual_similarities
                    + text_text_weight * tt_similarities
                ) / (1.0 + text_text_weight)
            else:
                combined_similarities = visual_similarities
        else:
            tt_similarities = torch.zeros_like(visual_similarities)
            combined_similarities = visual_similarities

        # Get top-k indices
        valid_mask = torch.isfinite(combined_similarities)
        k = min(self.top_k, valid_mask.sum().item())
        if k == 0:
            if return_score_breakdown:
                empty = torch.tensor([])
                return [], empty, empty, empty
            return [], torch.tensor([])

        top_k_values, top_k_indices = torch.topk(combined_similarities, k)

        # Filter out excluded entries represented by -inf scores.
        valid_results = torch.isfinite(top_k_values)
        top_k_values = top_k_values[valid_results]
        top_k_indices = top_k_indices[valid_results]

        # Retrieve entries
        retrieved_entries = [self.entries[idx.item()] for idx in top_k_indices]

        if not return_score_breakdown:
            return retrieved_entries, top_k_values

        visual_scores = visual_similarities[top_k_indices]
        tt_scores = tt_similarities[top_k_indices]
        return retrieved_entries, top_k_values, visual_scores, tt_scores
    
    def get_retrieval_kv_cache(
        self,
        query_feature: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
        exclude_frame_indices: Optional[List[int]] = None,
        return_info: bool = False,
        text_feature: Optional[torch.Tensor] = None,
        text_weight: float = 0.5,
        return_score_breakdown: bool = False,
        text_text_weight: Optional[float] = None,
        query_features: Optional[List[torch.Tensor]] = None,
    ) -> Optional[List[Dict[str, torch.Tensor]]]:
        """
        Retrieve similar entries and return their combined KV cache.

        Supports **multi-query** retrieval via *query_features*.  When a list of
        query features is supplied, similarity is averaged across queries before
        ranking – see :meth:`retrieve` and :meth:`retrieve_with_text`.

        Args:
            query_feature: Single query visual feature tensor.
            target_device: Device to move the KV cache to.
            target_dtype: Data type for the KV cache.
            exclude_frame_indices: Frame indices to exclude.
            return_info: If True, also return retrieved frame indices and similarities.
            text_feature: Optional text prompt feature for combined retrieval.
            text_weight: Backward-compatible fallback weight.
            text_text_weight: Weight for text-text similarity.
            query_features: Optional list of query visual features (multi-query).

        Returns:
            Combined KV cache from retrieved entries, or None if no entries retrieved.
            If return_info=True, returns (kv_cache, frame_indices, similarities).
            If return_info=True and return_score_breakdown=True, returns:
            (kv_cache, frame_indices, combined_scores, visual_scores, tt_scores).
        """
        # Use text-aware retrieval if text_feature is provided
        if text_feature is not None:
            retrieval_result = self.retrieve_with_text(
                query_feature,
                text_feature=text_feature,
                text_weight=text_weight,
                exclude_frame_indices=exclude_frame_indices,
                text_text_weight=text_text_weight,
                return_score_breakdown=return_score_breakdown,
                query_features=query_features,
            )
            if return_score_breakdown:
                retrieved_entries, similarities, visual_scores, tt_scores = retrieval_result
            else:
                retrieved_entries, similarities = retrieval_result
        else:
            retrieved_entries, similarities = self.retrieve(
                query_feature, exclude_frame_indices,
                query_features=query_features,
            )
            if return_score_breakdown:
                visual_scores = similarities
                tt_scores = torch.zeros_like(similarities)
        
        if len(retrieved_entries) == 0:
            if return_info:
                if return_score_breakdown:
                    return None, [], [], [], []
                return None, [], []
            return None
        
        # Concatenate KV caches from ALL top-k retrieved entries along the
        # sequence dimension (sorted by frame_idx ascending so that earlier
        # blocks appear first in the attention context, matching causal order).
        entries_sorted = sorted(retrieved_entries, key=lambda e: e.frame_idx)
        entry_frame_indices = [e.frame_idx for e in entries_sorted]

        num_blocks = len(entries_sorted[0].kv_cache)
        retrieval_kv_cache = []

        for block_idx in range(num_blocks):
            k_parts, v_parts = [], []
            entry_seq_lens = []
            meta = {}
            for entry in entries_sorted:
                block_cache = entry.kv_cache[block_idx]
                for key, value in block_cache.items():
                    if key == 'k':
                        k_parts.append(value.to(device=target_device, dtype=target_dtype))
                        entry_seq_lens.append(value.shape[1])
                    elif key == 'v':
                        v_parts.append(value.to(device=target_device, dtype=target_dtype))
                    elif key not in meta:
                        meta[key] = value if not isinstance(value, torch.Tensor) \
                            else value.to(device=target_device, dtype=target_dtype)

            merged_block: Dict[str, Any] = dict(meta)
            if k_parts:
                merged_block['k'] = torch.cat(k_parts, dim=1)
            if v_parts:
                merged_block['v'] = torch.cat(v_parts, dim=1)
            if len(entry_seq_lens) > 1:
                merged_block['entry_seq_lens'] = entry_seq_lens
                merged_block['entry_frame_indices'] = entry_frame_indices
            retrieval_kv_cache.append(merged_block)
        
        if return_info:
            # Return frame indices and similarities for all retrieved entries
            retrieved_frame_indices = [entry.frame_idx for entry in retrieved_entries]
            retrieved_similarities = similarities.tolist() if hasattr(similarities, 'tolist') else list(similarities)
            if not return_score_breakdown:
                return retrieval_kv_cache, retrieved_frame_indices, retrieved_similarities

            return (
                retrieval_kv_cache,
                retrieved_frame_indices,
                retrieved_similarities,
                visual_scores.tolist() if hasattr(visual_scores, "tolist") else list(visual_scores),
                tt_scores.tolist() if hasattr(tt_scores, "tolist") else list(tt_scores),
            )
        
        return retrieval_kv_cache
    
    def __len__(self) -> int:
        return len(self.entries)
    
    @property
    def num_entries(self) -> int:
        return len(self.entries)
    
    def remove_entry_by_idx(self, entry_idx: int):
        """
        Remove an entry at a specific index.
        
        Args:
            entry_idx: Index of the entry to remove (0-based)
        """
        if entry_idx < 0 or entry_idx >= len(self.entries):
            return
        
        self.entries.pop(entry_idx)
        
        # Update feature matrix
        if self.feature_matrix is not None and len(self.entries) > 0:
            self.feature_matrix = torch.cat([
                self.feature_matrix[:entry_idx],
                self.feature_matrix[entry_idx + 1:]
            ], dim=0)
        elif len(self.entries) == 0:
            self.feature_matrix = None
    
    def remove_similar_entries(
        self,
        query_feature: torch.Tensor,
        dedup_threshold: float = 0.95
    ) -> int:
        """
        Remove entries that are too similar to the query feature.
        
        This is used for memory deduplication: before adding a new frame,
        remove old frames that are very similar to avoid redundancy.
        
        Args:
            query_feature: Query feature tensor, shape [feature_dim]
            dedup_threshold: Similarity threshold above which entries are removed
            
        Returns:
            Number of entries removed
        """
        if len(self.entries) == 0 or self.feature_matrix is None:
            return 0
        
        # Normalize query feature to 1D
        query_cpu = self._to_1d_feature(query_feature)
        
        # Compute similarity with all entries
        similarities = torch.matmul(self.feature_matrix, query_cpu)
        
        # Find entries to remove (similarity > threshold)
        to_remove = (similarities > dedup_threshold).nonzero(as_tuple=True)[0].tolist()
        
        # Remove in reverse order to maintain correct indices
        removed_count = 0
        for idx in sorted(to_remove, reverse=True):
            self.remove_entry_by_idx(idx)
            removed_count += 1
        
        return removed_count
    
    def check_similarity(
        self,
        query_feature: torch.Tensor,
        additional_features: Optional[List[torch.Tensor]] = None,
        text_feature: Optional[torch.Tensor] = None,
        text_weight: float = 0.0,
    ) -> float:
        """
        Check the maximum similarity between query and all entries (plus additional features).
        
        This is used for deduplication: check if a new frame is too similar to
        existing frames before adding it.
        
        When ``text_feature`` is provided, uses a normalised combined score:
            combined = (visual_sim + text_sim * text_weight) / (1 + text_weight)
        where ``text_sim`` is the cosine similarity between the query text feature
        and each memory entry's stored text feature.  This ensures text information
        is considered during deduplication.
        
        Args:
            query_feature: Query visual feature tensor, shape [feature_dim]
            additional_features: Optional list of additional features to compare against
                                (e.g., features from other frames in the same block)
            text_feature: Optional text prompt feature for combined scoring.
                         If None, only visual similarity is used.
            text_weight: Weight for the text similarity term (only used when
                        text_feature is provided).  Default 0.0 (visual-only).
            
        Returns:
            Maximum (combined) similarity score (0.0 if no entries to compare)
        """
        # Normalize query feature to 1D
        query_cpu = self._to_1d_feature(query_feature)
        
        max_sim = 0.0
        
        # Check against existing entries
        if len(self.entries) > 0 and self.feature_matrix is not None:
            # Visual similarity
            visual_sims = torch.matmul(self.feature_matrix, query_cpu)  # [num_entries]
            
            if text_feature is not None and text_weight > 0:
                # Compute text-text similarity for each entry
                text_cpu = self._to_1d_feature(text_feature)
                tt_vals = []
                for entry in self.entries:
                    if entry.text_feature is None:
                        tt_vals.append(0.0)
                    else:
                        tt_vals.append(torch.dot(entry.text_feature.float(), text_cpu).item())
                tt_sims = torch.tensor(tt_vals, dtype=visual_sims.dtype)
                
                # Normalised combined score
                combined_sims = (visual_sims + tt_sims * text_weight) / (1.0 + text_weight)
                max_sim = max(max_sim, combined_sims.max().item())
            else:
                max_sim = max(max_sim, visual_sims.max().item())
        
        # Check against additional features (visual-only, no text info available)
        if additional_features:
            for feat in additional_features:
                feat_cpu = self._to_1d_feature(feat)
                sim = torch.dot(query_cpu, feat_cpu).item()
                max_sim = max(max_sim, sim)
        
        return max_sim


class MemoryEncoder(nn.Module):
    """
    Wrapper for PEWithAdapter that provides a clean interface for encoding frames.
    
    This handles:
    1. Loading the pretrained PE model with adapter (supports loading fine-tuned adapter weights)
    2. Preprocessing frames for encoding
    3. Computing mean features across multiple frames
    
    Usage:
        # Option 1: Load with fine-tuned adapter checkpoint (recommended after warmup training)
        encoder = MemoryEncoder(
            adapter_checkpoint_path="checkpoints/pe_warmup/best_checkpoint.pt"
        )
        
        # Option 2: Load original PE model without adapter fine-tuning
        encoder = MemoryEncoder(
            model_config="PE-Core-L14-336",
            backbone_checkpoint_path="/path/to/PE-Core-L14-336.pt"
        )
    """
    
    def __init__(
        self,
        model_config: str = "PE-Core-L14-336",
        backbone_checkpoint_path: Optional[str] = None,
        adapter_checkpoint_path: Optional[str] = None,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize the Memory Encoder.
        
        The loading logic follows wramup_memory_encoder.py:
        - If adapter_checkpoint_path is provided, it loads the fine-tuned adapter weights
          along with all necessary configurations (model_config, backbone_checkpoint_path,
          adapter_hidden_dim, adapter_dropout, adapter_scale) from the checkpoint.
        - If only backbone_checkpoint_path is provided, it creates a new PEWithAdapter
          with default adapter settings.
        
        Args:
            model_config: PE model configuration name (e.g., "PE-Core-L14-336")
                         Can be overridden by checkpoint if adapter_checkpoint_path is provided
            backbone_checkpoint_path: Path to PE backbone model checkpoint
                                     Can be overridden by checkpoint if adapter_checkpoint_path is provided
            adapter_checkpoint_path: Path to trained adapter checkpoint from wramup_memory_encoder.py
                                    If provided, this takes precedence and loads all configs from checkpoint
            device: Device to run the encoder on
            dtype: Data type for computation
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # Import PEWithAdapter and related functions from wramup_memory_encoder
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from wramup_memory_encoder import PEWithAdapter, load_adapter_checkpoint
        
        # Initialize or load the model
        if adapter_checkpoint_path is not None and os.path.exists(adapter_checkpoint_path):
            # Load fine-tuned adapter checkpoint (preferred method after warmup training)
            # This automatically loads:
            # - model_config from checkpoint (or uses provided model_config as fallback)
            # - backbone_checkpoint_path from checkpoint
            # - adapter_hidden_dim, adapter_dropout, adapter_scale from checkpoint
            # - adapter weights (adapter_state_dict)
            print(f"[MemoryEncoder] Loading fine-tuned adapter from: {adapter_checkpoint_path}")
            self.model, checkpoint_info = load_adapter_checkpoint(
                checkpoint_path=adapter_checkpoint_path,
                model_config=model_config,  # Used as fallback if not in checkpoint
                backbone_checkpoint_path=backbone_checkpoint_path,  # Used as fallback if not in checkpoint
                device=device,
                freeze_backbone=True
            )
            
            # Log loaded configuration
            loaded_config = checkpoint_info.get('model_config', model_config)
            loaded_backbone = checkpoint_info.get('backbone_checkpoint_path', backbone_checkpoint_path)
            adapter_hidden_dim = checkpoint_info.get('adapter_hidden_dim', None)
            adapter_scale = checkpoint_info.get('adapter_scale', 0.1)
            print(f"[MemoryEncoder] Loaded config: model={loaded_config}, "
                  f"adapter_hidden_dim={adapter_hidden_dim}, adapter_scale={adapter_scale}")
        else:
            # Create new PEWithAdapter without fine-tuned adapter weights
            print(f"[MemoryEncoder] Creating new PEWithAdapter with model_config={model_config}")
            if adapter_checkpoint_path is not None:
                print(f"[MemoryEncoder] Warning: adapter_checkpoint_path '{adapter_checkpoint_path}' not found, "
                      f"using default adapter initialization")
            
            self.model = PEWithAdapter(
                model_config=model_config,
                checkpoint_path=backbone_checkpoint_path,
                adapter_hidden_dim=None,  # Default: feature_dim // 4
                adapter_dropout=0.1,
                adapter_scale=0.1
            )
        
        self.model = self.model.to(device=device)
        self.model.eval()
        
        # Freeze all parameters for inference
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get image size and feature dim
        self.image_size = self.model.image_size
        self.feature_dim = self.model.feature_dim
        
        print(f"[MemoryEncoder] Initialized with image_size={self.image_size}, feature_dim={self.feature_dim}")
        
        # Initialize preprocessing transform
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lcm_pe'))
        import core.vision_encoder.transforms as pe_transforms
        self.preprocess = pe_transforms.get_image_transform(self.image_size)
        
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
            # Add batch dimension: [num_frames, C, H, W] -> [1, num_frames, C, H, W]
            frames = frames.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        batch_size, num_frames, c, h, w = frames.shape
        
        # Resize if needed
        if h != self.image_size or w != self.image_size:
            frames_resized = F.interpolate(
                frames.view(-1, c, h, w),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).view(batch_size, num_frames, c, self.image_size, self.image_size)
        else:
            frames_resized = frames
        
        # Convert from [0, 1] to [-1, 1] (same normalization as PE model expects)
        # PE model uses Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) which converts [0,1] to [-1,1]
        frames_normalized = frames_resized * 2.0 - 1.0
        
        # Ensure correct device and dtype
        frames_normalized = frames_normalized.to(device=self.device, dtype=self.dtype)
        
        # Encode using PE model
        # PE expects [B, N, C, H, W] and returns [B, feature_dim] (video-level)
        # or [B, N, feature_dim] if return_frame_features=True
        features = self.model.encode_video(
            frames_normalized,
            normalize=False,
            return_frame_features=True
        )  # [B, N, feature_dim]
        
        # Compute mean across frames
        mean_features = features.mean(dim=1)  # [B, feature_dim]
        
        if normalize:
            mean_features = F.normalize(mean_features, dim=-1)
        
        if squeeze_batch:
            mean_features = mean_features.squeeze(0)  # [feature_dim]
        
        return mean_features
    
    @torch.no_grad()
    def encode_decoded_frames(
        self,
        decoded_video: torch.Tensor,
        latent_frame_idx: int = 0,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode decoded frames from VAE output for a specific latent frame.
        
        Due to VAE's 4x temporal compression, one latent frame corresponds to 4 real frames.
        This function extracts the 4 frames for a given latent frame index and computes
        their mean feature.
        
        Args:
            decoded_video: Decoded video tensor, shape [B, T, C, H, W] where T = 4 * num_latent_frames
                          Values should be in range [0, 1]
            latent_frame_idx: Index of the latent frame (0-indexed)
            normalize: Whether to L2 normalize the output features
            
        Returns:
            Mean feature for the 4 frames corresponding to the latent frame, shape [B, feature_dim]
        """
        # Extract 4 frames corresponding to this latent frame
        # Frame indices: latent_frame_idx * 4 to latent_frame_idx * 4 + 4
        start_idx = latent_frame_idx * 4
        end_idx = start_idx + 4
        
        # Handle edge case where we don't have 4 frames
        total_frames = decoded_video.shape[1]
        end_idx = min(end_idx, total_frames)
        
        frames = decoded_video[:, start_idx:end_idx]  # [B, 4, C, H, W] (or fewer)
        
        return self.encode_frames(frames, normalize=normalize)
    

def extract_kv_cache_for_frame(
    full_kv_cache: List[Dict[str, torch.Tensor]],
    frame_idx: int,
    frame_seq_length: int = 1560,
    num_frames_to_extract: int = 1,
    local_end_index: Optional[int] = None
) -> List[Dict[str, torch.Tensor]]:
    """
    Extract KV cache entries for a specific frame from the full KV cache.
    
    In sliding window mode, the KV cache is a rolling buffer. We need to use
    local_end_index to determine the actual position in the cache.
    
    Args:
        full_kv_cache: Full KV cache from all transformer blocks
        frame_idx: The absolute frame index to extract
        frame_seq_length: Sequence length per frame (default 1560 for Wan model)
        num_frames_to_extract: Number of frames to extract
        local_end_index: The current end index in the local KV cache buffer.
                        If None, assumes global indexing (frame_idx * frame_seq_length)
        
    Returns:
        Extracted KV cache for the specified frame(s)
    """
    # Get the first block to determine the actual end index
    if local_end_index is None and len(full_kv_cache) > 0:
        local_end_index = full_kv_cache[0].get('local_end_index', None)
        if local_end_index is not None and hasattr(local_end_index, 'item'):
            local_end_index = local_end_index.item()
    
    # Calculate the position in the cache
    # In sliding window mode, the most recent frame ends at local_end_index
    # So if we want the last frame that was just added, we extract from
    # local_end_index - frame_seq_length to local_end_index
    if local_end_index is not None:
        # Use the most recent frame's KV cache (the one just generated)
        # This is more reliable in sliding window mode
        end_idx = local_end_index
        start_idx = max(0, end_idx - num_frames_to_extract * frame_seq_length)
    else:
        # Fallback to absolute indexing
        start_idx = frame_idx * frame_seq_length
        end_idx = start_idx + num_frames_to_extract * frame_seq_length
    
    extracted_cache = []
    for block_cache in full_kv_cache:
        block_extracted = {}
        for key, value in block_cache.items():
            if key in ['k', 'v']:
                # Get actual valid range
                actual_end = min(end_idx, value.shape[1])
                actual_start = min(start_idx, actual_end)
                # Extract the relevant portion
                if actual_end > actual_start:
                    block_extracted[key] = value[:, actual_start:actual_end].clone()
                else:
                    # If no valid range, create empty tensor with correct shape
                    block_extracted[key] = value[:, :0].clone()
            else:
                block_extracted[key] = value
        extracted_cache.append(block_extracted)
    
    return extracted_cache


def merge_retrieval_kv_cache(
    base_kv_cache: List[Dict[str, torch.Tensor]],
    retrieval_kv_cache: List[Dict[str, torch.Tensor]],
    merge_mode: str = "prepend"
) -> List[Dict[str, torch.Tensor]]:
    """
    Merge retrieval KV cache with the base KV cache.
    
    Args:
        base_kv_cache: The base KV cache (current sliding window)
        retrieval_kv_cache: The retrieved KV cache from memory bank
        merge_mode: How to merge - 'prepend' adds before base, 'append' adds after
        
    Returns:
        Merged KV cache
    """
    if retrieval_kv_cache is None:
        return base_kv_cache
    
    merged_cache = []
    for base_block, retrieval_block in zip(base_kv_cache, retrieval_kv_cache):
        merged_block = {}
        for key in ['k', 'v']:
            if merge_mode == "prepend":
                merged_block[key] = torch.cat([
                    retrieval_block[key],
                    base_block[key]
                ], dim=1)
            else:  # append
                merged_block[key] = torch.cat([
                    base_block[key],
                    retrieval_block[key]
                ], dim=1)
        
        # Copy other metadata
        for key in base_block:
            if key not in ['k', 'v']:
                merged_block[key] = base_block[key]
        
        merged_cache.append(merged_block)
    
    return merged_cache
