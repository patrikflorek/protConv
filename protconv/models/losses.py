"""
Loss functions for protein CA backbone prediction.
- Structure (Kabsch RMSD) loss
- CA–CA distance loss (consecutive)
- All-pairs CA–CA distance loss
"""
import torch
from protconv.utils.geometry import ca_trace_reconstruction_torch, kabsch_torch

def struct_loss_kabsch(pred_trace, target_coords):
    aligned_pred = kabsch_torch(pred_trace, target_coords)
    return ((aligned_pred - target_coords) ** 2).sum(-1).mean()

def ca_ca_consecutive_loss(pred_trace, target_mean=3.8, target_std=0.2):
    if pred_trace.shape[0] > 1:
        ca_diffs = pred_trace[1:] - pred_trace[:-1]
        ca_dists = torch.norm(ca_diffs, dim=1)
        return ((ca_dists - target_mean) ** 2 / (target_std ** 2)).mean()
    return torch.tensor(0.0, device=pred_trace.device)

def ca_ca_allpairs_loss(pred_trace, target_mean=3.8, target_std=1.5):
    N = pred_trace.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=pred_trace.device)
    diff = pred_trace.unsqueeze(1) - pred_trace.unsqueeze(0)  # (N, N, 3)
    dists = torch.norm(diff, dim=-1)  # (N, N)
    mask = torch.ones_like(dists, dtype=torch.bool)
    idx = torch.arange(N)
    mask[idx, idx] = 0  # remove self-distances
    mask[idx[:-1], idx[1:]] = 0  # remove consecutive
    mask[idx[1:], idx[:-1]] = 0
    selected = dists[mask]
    return ((selected - target_mean) ** 2 / (target_std ** 2)).mean() if selected.numel() > 0 else torch.tensor(0.0, device=pred_trace.device)


def compute_fragment_losses(
    pred_vectors, target_coords, mask,
    lambda_struct=1.0, lambda_caca=1.0, lambda_allpairs=1.0,
    allpairs_mean=3.8, allpairs_std=1.5
):
    """
    Compute total and sub-losses for a predicted fragment.
    Returns a dict with total_loss, struct_loss, ca_ca_loss, allpairs_loss (all tensors).
    """
    pred_trace = ca_trace_reconstruction_torch(pred_vectors, mask)
    struct_loss = struct_loss_kabsch(pred_trace, target_coords)
    ca_ca_loss = ca_ca_consecutive_loss(pred_trace)
    allpairs_loss = ca_ca_allpairs_loss(pred_trace, target_mean=allpairs_mean, target_std=allpairs_std)
    total_loss = (
        lambda_struct * struct_loss +
        lambda_caca * ca_ca_loss +
        lambda_allpairs * allpairs_loss
    )
    return {
        'total_loss': total_loss,
        'struct_loss': struct_loss,
        'ca_ca_loss': ca_ca_loss,
        'allpairs_loss': allpairs_loss,
    }
