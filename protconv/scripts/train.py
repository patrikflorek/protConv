"""
train.py
--------
Train a 1D CNN model to predict protein backbone coordinates from amino acid fragments.

Usage:
    python -m protconv train

Loads fragment datasets from data/ProteinNet/casp7/fragments using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# --- Model import ---
from protconv.models.simple_cnn import Simple1DCNN
# --- Data loader import ---
from protconv.data.loader import get_train_val_loaders
# --- Geometry utilities ---
from protconv.utils.geometry import kabsch_torch, ca_trace_reconstruction_torch

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Training ---
def train():
    # Load datasets
    train_loader, val_loader = get_train_val_loaders(batch_size=BATCH_SIZE)

    model = Simple1DCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss(reduction="none")

    history = {
        "train_loss": [],
        "val_loss": [],
        "struct_loss": [],
        "norm_loss": [],
        "parallel_loss": [],
        "val_struct_loss": [],
        "val_norm_loss": [],
        "val_parallel_loss": [],
    }
    best_val = float("inf")
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        struct_loss_sum = 0.0
        norm_loss_sum = 0.0
        parallel_loss_sum = 0.0
        for seqs, coords, mask in train_bar:
            seqs, coords, mask = seqs.to(DEVICE), coords.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = 0.0
            lambda_norm = 1.0  # Weight for half-bond norm penalty
            lambda_parallel = 1.0  # Weight for parallelism penalty
            batch_struct = 0.0
            batch_norm = 0.0
            batch_parallel = 0.0
            for i in range(seqs.size(0)):
                seq_len = int(mask[i].sum())
                if seq_len < 3:
                    continue  # skip too-short fragments
                pred_vectors = outputs[i]  # (seq_len, 6)
                pred_trace = ca_trace_reconstruction_torch(
                    pred_vectors, mask[i]
                )  # (seq_len, 3)
                target_trace = coords[i, mask[i], :]  # (seq_len, 3)
                aligned_pred = kabsch_torch(pred_trace, target_trace)
                # Main structure loss
                struct_loss = criterion(aligned_pred, target_trace).mean()
                # Vector norm penalty (half-bond vectors)
                into_ca = pred_vectors[:, :3]
                out_from = pred_vectors[:, 3:]
                into_norm = torch.norm(into_ca, dim=1)
                out_from_norm = torch.norm(out_from, dim=1)
                norm_penalty = (((into_norm - 1.9) ** 2) / (0.2**2)).mean() + (
                    ((out_from_norm - 1.9) ** 2) / (0.2**2)
                ).mean()
                norm_penalty = 0.5 * norm_penalty  # average the two
                # Parallelism penalty
                if pred_vectors.shape[0] > 1:
                    out_from_n = out_from[:-1] / (
                        out_from_norm[:-1].unsqueeze(1) + 1e-8
                    )
                    into_ca_n = into_ca[1:] / (into_norm[1:].unsqueeze(1) + 1e-8)
                    dot_products = (out_from_n * into_ca_n).sum(dim=1)
                    parallel_penalty = ((dot_products - 1) ** 2).mean()
                else:
                    parallel_penalty = 0.0
                batch_struct += struct_loss.item()
                batch_norm += (lambda_norm * norm_penalty).item()
                batch_parallel += (lambda_parallel * parallel_penalty).item()
                loss += (
                    struct_loss
                    + lambda_norm * norm_penalty
                    + lambda_parallel * parallel_penalty
                )
            num_frags = seqs.size(0)
            loss /= num_frags
            batch_struct /= num_frags
            batch_norm /= num_frags
            batch_parallel /= num_frags
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * num_frags
            struct_loss_sum += batch_struct * num_frags
            norm_loss_sum += batch_norm * num_frags
            parallel_loss_sum += batch_parallel * num_frags
            train_batches += num_frags
            train_bar.set_postfix(batch_loss=loss.item())
        N = len(train_loader.dataset)
        train_loss /= N
        history["train_loss"].append(train_loss)
        history["struct_loss"].append(struct_loss_sum / N)
        history["norm_loss"].append(norm_loss_sum / N)
        history["parallel_loss"].append(parallel_loss_sum / N)

        # Validation
        model.eval()
        val_loss = 0.0
        val_struct_sum = 0.0
        val_norm_sum = 0.0
        val_parallel_sum = 0.0
        val_batches = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [val]", leave=False)
        with torch.no_grad():
            for seqs, coords, mask in val_bar:
                seqs, coords, mask = seqs.to(DEVICE), coords.to(DEVICE), mask.to(DEVICE)
                outputs = model(seqs)
                loss = 0.0
                lambda_norm = 1.0
                lambda_parallel = 1.0
                batch_struct = 0.0
                batch_norm = 0.0
                batch_parallel = 0.0
                for i in range(seqs.size(0)):
                    seq_len = int(mask[i].sum())
                    if seq_len < 3:
                        continue  # skip too-short fragments
                    pred_vectors = outputs[i]
                    pred_trace = ca_trace_reconstruction_torch(pred_vectors, mask[i])
                    target_trace = coords[i, mask[i], :]
                    aligned_pred = kabsch_torch(pred_trace, target_trace)
                    # Main structure loss
                    struct_loss = criterion(aligned_pred, target_trace).mean()
                    # CA-CA distance penalty
                    ca_distances = torch.norm(
                        aligned_pred[1:] - aligned_pred[:-1], dim=1
                    )
                    dist_penalty = ((ca_distances - 1.8) ** 2 / (0.2**2)).mean()
                    loss += struct_loss + lambda_dist * dist_penalty
                loss /= seqs.size(0)
                val_loss += loss.item() * seqs.size(0)
                val_batches += seqs.size(0)
                num_frags = seqs.size(0)
                loss /= num_frags
                batch_struct /= num_frags
                batch_norm /= num_frags
                batch_parallel /= num_frags
                val_loss += loss.item() * num_frags
                val_struct_sum += batch_struct * num_frags
                val_norm_sum += batch_norm * num_frags
                val_parallel_sum += batch_parallel * num_frags
                val_batches += num_frags
                val_bar.set_postfix(batch_loss=loss.item())
            N_val = len(val_loader.dataset)
            val_loss /= N_val
            history["val_loss"].append(val_loss)
            history["val_struct_loss"].append(val_struct_sum / N_val)
            history["val_norm_loss"].append(val_norm_sum / N_val)
            history["val_parallel_loss"].append(val_parallel_sum / N_val)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
        print(f"Epoch {epoch}: Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")

    print(f"Best epoch: {best_epoch} (Val loss: {best_val:.4f})")
    return history


if __name__ == "__main__":
    history = train()
