"""
visionbox/video_analyzer/key_selector.py
Selects key clips based on V-JEPA embedding diversity and magnitude.
"""

import torch
import logging

logger = logging.getLogger(__name__)

def select_key_clips(embeddings: torch.Tensor, k: int = 3) -> list[int]:
    """
    Selects k key clips from the embeddings tensor.
    We pick based on highest L2 norm (magnitude ~= visual saliency/motion)
    while leaning toward temporal spacing.
    
    Args:
        embeddings: torch.Tensor of shape [N, D]
        k: number of clips to select
        
    Returns:
        List of chronological integer indices of the selected clips.
    """
    N = embeddings.shape[0]
    if k >= N:
        logger.info("Requested %d clips but only %d available. Returning all.", k, N)
        return list(range(N))
        
    # Compute L2 norms (magnitudes)
    # Intuitively, dynamic or complex scenes have larger/different embeddings than static ones
    magnitudes = torch.norm(embeddings, p=2, dim=1)  # [N]
    
    # Sort indices by magnitude descending
    sorted_idx = torch.argsort(magnitudes, descending=True).tolist()
    
    selected = []
    # Greedy selection: pick high-magnitude clips, but try to avoid
    # picking clips that are immediately adjacent temporally (too similar)
    for idx in sorted_idx:
        # Check if adjacent to currently selected
        is_adjacent = any(abs(idx - s) <= 1 for s in selected)
        
        # We can afford to skip if we have enough remaining candidates
        remaining_candidates = len([x for x in sorted_idx if x not in selected])
        
        if not is_adjacent or (len(selected) + remaining_candidates) <= k:
            selected.append(idx)
            
        if len(selected) == k:
            break
            
    # Fallback to fill up to k if we were too strict
    if len(selected) < k:
        for idx in sorted_idx:
            if idx not in selected:
                selected.append(idx)
            if len(selected) == k:
                break
                
    # Return chronologically sorted indices for logical storytelling
    final_indices = sorted(selected)
    logger.info("Selected top %d key clips out of %d. Indices: %s", k, N, final_indices)
    
    return final_indices
