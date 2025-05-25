"""
geometry.py
-----------
Differentiable geometry utilities for protein backbone analysis.
Includes:
- kabsch_torch: differentiable Kabsch alignment
- ca_trace_reconstruction_torch: differentiable CA trace reconstruction from predicted vectors
"""
import torch


def kabsch_torch(P, Q):
    """
    Differentiable Kabsch alignment in PyTorch.
    Args:
        P, Q: (N, 3) tensors (pred, target)
    Returns:
        P_aligned: (N, 3) tensor, P aligned to Q
    """
    P_mean = P.mean(dim=0, keepdim=True)
    Q_mean = Q.mean(dim=0, keepdim=True)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean
    C = torch.matmul(P_centered.t(), Q_centered)
    V, S, W = torch.svd(C)
    d = torch.det(torch.matmul(V, W.t()))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=P.device, dtype=P.dtype))
    R = torch.matmul(V, torch.matmul(D, W.t()))
    P_rot = torch.matmul(P_centered, R)
    P_aligned = P_rot + Q_mean
    return P_aligned


def ca_trace_reconstruction_torch(pred_vectors, mask):
    """
    Reconstruct CA trace from predicted half-bond vectors (PyTorch, differentiable).
    Args:
        pred_vectors: (seq_len, 6) tensor
        mask: (seq_len,) bool tensor
    Returns:
        ca_trace: (seq_len, 3) tensor
    """
    pred_vectors = pred_vectors[mask]
    seq_len = pred_vectors.shape[0]
    to_prev_ca = pred_vectors[:, :3]
    to_next_ca = pred_vectors[:, 3:]
    ca_trace = [pred_vectors.new_zeros(3)]  # Start at origin
    for i in range(1, seq_len):
        bond = to_next_ca[i-1] + to_prev_ca[i]  # sum of two half-bonds = full bond
        ca_trace.append(ca_trace[-1] + bond)
    ca_trace = torch.stack(ca_trace, dim=0)
    return ca_trace


def fragment_to_ca_trace_vectors(ca_coords):
    """
    Given CA atom coordinates (N,3), return (N,6) tensor of [to_prev_ca, to_next_ca] per residue as half-bond vectors.
    to_prev_ca[i] = 0.5 * (ca[i] - ca[i-1]) for i>0, 0 for i=0
    to_next_ca[i] = 0.5 * (ca[i+1] - ca[i]) for i<N-1, 0 for i=N-1
    Args:
        ca_coords: (N,3) tensor
    Returns:
        ca_trace_vectors: (N,6) tensor
    """
    N = ca_coords.shape[0]
    to_prev = torch.zeros_like(ca_coords)
    to_next = torch.zeros_like(ca_coords)
    if N > 1:
        to_prev[1:] = 0.5 * (ca_coords[1:] - ca_coords[:-1])
        to_next[:-1] = 0.5 * (ca_coords[1:] - ca_coords[:-1])
    ca_trace_vectors = torch.cat([to_prev, to_next], dim=1)
    return ca_trace_vectors
