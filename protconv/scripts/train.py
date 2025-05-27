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
VOCAB_SIZE = 21
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Training ---
def compute_fragment_losses(
    pred_vectors, target_coords, mask, criterion, lambda_norm=1.0, lambda_parallel=1.0
):
    # pred_vectors: (seq_len, 6), target_coords: (seq_len, 3), mask: (seq_len,)
    pred_trace = ca_trace_reconstruction_torch(pred_vectors, mask)
    aligned_pred = kabsch_torch(pred_trace, target_coords)
    struct_loss = criterion(aligned_pred, target_coords).mean()
    # Vector norm penalty
    into_ca = pred_vectors[:, :3]
    out_from = pred_vectors[:, 3:]
    into_norm = torch.norm(into_ca, dim=1)
    out_from_norm = torch.norm(out_from, dim=1)
    norm_penalty = 0.5 * (
        (((into_norm - 1.9) ** 2) / (0.2**2)).mean()
        + (((out_from_norm - 1.9) ** 2) / (0.2**2)).mean()
    )
    # Parallelism penalty
    if pred_vectors.shape[0] > 1:
        out_from_n = out_from[:-1] / (out_from_norm[:-1].unsqueeze(1) + 1e-8)
        into_ca_n = into_ca[1:] / (into_norm[1:].unsqueeze(1) + 1e-8)
        dot_products = (out_from_n * into_ca_n).sum(dim=1)
        parallel_penalty = ((dot_products - 1) ** 2).mean()
    else:
        parallel_penalty = 0.0
    total_loss = (
        struct_loss + lambda_norm * norm_penalty + lambda_parallel * parallel_penalty
    )
    return (
        total_loss,
        struct_loss.item(),
        (lambda_norm * norm_penalty).item(),
        (lambda_parallel * parallel_penalty).item(),
    )


def train(
    model_class=Simple1DCNN,
    batch_size=BATCH_SIZE,
    vocab_size=VOCAB_SIZE,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_EPOCHS,
    lambda_norm=1.0,
    lambda_parallel=1.0,
):
    # Load datasets
    train_loader, val_loader = get_train_val_loaders(batch_size=batch_size)
    model = model_class(vocab_size=vocab_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
    lambda_norm = 1.0
    lambda_parallel = 1.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        struct_loss_sum = 0.0
        norm_loss_sum = 0.0
        parallel_loss_sum = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for seqs, coords, mask in train_bar:
            seqs, coords, mask = seqs.to(DEVICE), coords.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(seqs)
            batch_loss = 0.0
            batch_struct = 0.0
            batch_norm = 0.0
            batch_parallel = 0.0
            valid_count = 0
            for i in range(seqs.size(0)):
                seq_len = int(mask[i].sum())
                if seq_len < 3:
                    continue
                pred_vectors = outputs[i]
                target_trace = coords[i, mask[i], :]
                total, struct, norm, parallel = compute_fragment_losses(
                    pred_vectors,
                    target_trace,
                    mask[i],
                    criterion,
                    lambda_norm,
                    lambda_parallel,
                )
                batch_loss += total
                batch_struct += struct
                batch_norm += norm
                batch_parallel += parallel
                valid_count += 1
            if valid_count > 0:
                batch_loss /= valid_count
                batch_struct /= valid_count
                batch_norm /= valid_count
                batch_parallel /= valid_count
                batch_loss.backward()
                optimizer.step()
                train_loss += batch_loss.item() * valid_count
                struct_loss_sum += batch_struct * valid_count
                norm_loss_sum += batch_norm * valid_count
                parallel_loss_sum += batch_parallel * valid_count
            train_bar.set_postfix(
                batch_loss=batch_loss.item() if valid_count > 0 else 0
            )
        N = len(train_loader.dataset)
        history["train_loss"].append(train_loss / N)
        history["struct_loss"].append(struct_loss_sum / N)
        history["norm_loss"].append(norm_loss_sum / N)
        history["parallel_loss"].append(parallel_loss_sum / N)
        # Validation
        model.eval()
        val_loss = 0.0
        val_struct_sum = 0.0
        val_norm_sum = 0.0
        val_parallel_sum = 0.0
        ca_distances_all = []  # For CA-CA distance statistics
        to_prev_norms_all = []  # For to_prev_ca vector norms
        to_next_norms_all = []  # For to_next_ca vector norms
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [val]", leave=False)
        with torch.no_grad():
            for seqs, coords, mask in val_bar:
                seqs, coords, mask = seqs.to(DEVICE), coords.to(DEVICE), mask.to(DEVICE)
                outputs = model(seqs)
                batch_loss = 0.0
                batch_struct = 0.0
                batch_norm = 0.0
                batch_parallel = 0.0
                valid_count = 0
                for i in range(seqs.size(0)):
                    seq_len = int(mask[i].sum())
                    if seq_len < 3:
                        continue
                    pred_vectors = outputs[i]
                    target_trace = coords[i, mask[i], :]
                    total, struct, norm, parallel = compute_fragment_losses(
                        pred_vectors,
                        target_trace,
                        mask[i],
                        criterion,
                        lambda_norm,
                        lambda_parallel,
                    )
                    batch_loss += total
                    batch_struct += struct
                    batch_norm += norm
                    batch_parallel += parallel
                    valid_count += 1
                    # --- CA-CA distance and vector norms logging ---
                    # Reconstruct CA trace from predicted vectors using mask
                    pred_trace = ca_trace_reconstruction_torch(pred_vectors, mask[i])  # (seq_len, 3)
                    # Compute pairwise distances between consecutive CAs
                    if pred_trace.shape[0] > 1:
                        diffs = pred_trace[1:] - pred_trace[:-1]
                        dists = torch.norm(diffs, dim=1)
                        ca_distances_all.append(dists.cpu())
                    # Compute vector norms for to_prev_ca and to_next_ca
                    to_prev_ca = pred_vectors[mask[i], :3]  # (seq_len, 3)
                    to_next_ca = pred_vectors[mask[i], 3:]  # (seq_len, 3)
                    if to_prev_ca.shape[0] > 0:
                        to_prev_norms_all.append(torch.norm(to_prev_ca, dim=1).cpu())
                    if to_next_ca.shape[0] > 0:
                        to_next_norms_all.append(torch.norm(to_next_ca, dim=1).cpu())
                if valid_count > 0:
                    batch_loss /= valid_count
                    batch_struct /= valid_count
                    batch_norm /= valid_count
                    batch_parallel /= valid_count
                    val_loss += batch_loss.item() * valid_count
                    val_struct_sum += batch_struct * valid_count
                    val_norm_sum += batch_norm * valid_count
                    val_parallel_sum += batch_parallel * valid_count
                val_bar.set_postfix(
                    batch_loss=batch_loss.item() if valid_count > 0 else 0
                )
        N_val = len(val_loader.dataset)
        history["val_loss"].append(val_loss / N_val)
        history["val_struct_loss"].append(val_struct_sum / N_val)
        history["val_norm_loss"].append(val_norm_sum / N_val)
        history["val_parallel_loss"].append(val_parallel_sum / N_val)
        # --- CA-CA distance and vector norm statistics logging ---
        if ca_distances_all:
            all_dists = torch.cat(ca_distances_all)
            mean_dist = all_dists.mean().item()
            std_dist = all_dists.std().item()
            print(f"[Validation CA-CA] mean: {mean_dist:.3f} Å, std: {std_dist:.3f} Å")
        if to_prev_norms_all:
            all_prev_norms = torch.cat(to_prev_norms_all)
            mean_prev = all_prev_norms.mean().item()
            std_prev = all_prev_norms.std().item()
            print(f"[Validation to_prev_ca norm] mean: {mean_prev:.3f} Å, std: {std_prev:.3f} Å")
        if to_next_norms_all:
            all_next_norms = torch.cat(to_next_norms_all)
            mean_next = all_next_norms.mean().item()
            std_next = all_next_norms.std().item()
            print(f"[Validation to_next_ca norm] mean: {mean_next:.3f} Å, std: {std_next:.3f} Å")
        if (val_loss / N_val) < best_val:
            best_val = val_loss / N_val
            best_epoch = epoch
        print(
            f"Epoch {epoch}: Train loss {train_loss / N:.4f}, Val loss {val_loss / N_val:.4f}"
        )
    print(f"Best epoch: {best_epoch} (Val loss: {best_val:.4f})")
    return model, history


def main():
    model, history = train()

    model_output_path = os.path.join(os.getcwd(), "models/latest.pt")
    torch.save(model.state_dict(), model_output_path)


if __name__ == "__main__":
    raise NotImplementedError("Call script with `python -m protconv train`")
