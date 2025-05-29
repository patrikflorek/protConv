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
LAMBDA_STRUCT = 1.0
LAMBDA_CACA = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Training ---
from protconv.models.losses import compute_fragment_losses


def train(
    model_class=Simple1DCNN,
    batch_size=BATCH_SIZE,
    vocab_size=VOCAB_SIZE,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_EPOCHS,
    lambda_struct=1.0,
    lambda_caca=1.0,
    lambda_allpairs=1.0,
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
        "caca_loss": [],
        "val_struct_loss": [],
        "val_caca_loss": [],
    }
    best_val = float("inf")
    best_epoch = 0
    lambda_caca = lambda_caca
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        struct_loss_sum = 0.0
        caca_loss_sum = 0.0
        allpairs_loss_sum = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for seqs, coords, mask in train_bar:
            seqs, coords, mask = seqs.to(DEVICE), coords.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(seqs)
            batch_loss = 0.0
            batch_struct = 0.0
            batch_caca = 0.0
            batch_allpairs = 0.0
            valid_count = 0
            for i in range(seqs.size(0)):
                seq_len = int(mask[i].sum())
                if seq_len < 3:
                    continue
                pred_vectors = outputs[i]
                target_trace = coords[i, mask[i], :]
                losses = compute_fragment_losses(
                    pred_vectors,
                    target_trace,
                    mask[i],
                    lambda_struct=lambda_struct,
                    lambda_caca=lambda_caca,
                    lambda_allpairs=lambda_allpairs,
                )
                total = losses["total_loss"]
                struct = losses["struct_loss"]
                caca = losses["ca_ca_loss"]
                allpairs = losses["allpairs_loss"]
                batch_loss += total
                batch_struct += struct
                batch_caca += caca
                batch_allpairs += allpairs
                valid_count += 1
            if valid_count > 0:
                batch_loss /= valid_count
                batch_struct /= valid_count
                batch_caca /= valid_count
                batch_allpairs /= valid_count
                batch_loss.backward()
                optimizer.step()
                train_loss += batch_loss.item() * valid_count
                struct_loss_sum += batch_struct * valid_count
                caca_loss_sum += batch_caca * valid_count
                allpairs_loss_sum += batch_allpairs * valid_count
            train_bar.set_postfix(
                batch_loss=batch_loss.item() if valid_count > 0 else 0
            )
        N = len(train_loader.dataset)
        history["train_loss"].append(train_loss / N)
        history["struct_loss"].append(struct_loss_sum / N)
        history["caca_loss"].append(caca_loss_sum / N)
        # Validation
        model.eval()
        val_loss = 0.0
        val_struct_sum = 0.0
        val_caca_sum = 0.0
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
                batch_caca = 0.0
                valid_count = 0
                for i in range(seqs.size(0)):
                    seq_len = int(mask[i].sum())
                    if seq_len < 3:
                        continue
                    pred_vectors = outputs[i]
                    target_trace = coords[i, mask[i], :]
                    total, struct, caca = compute_fragment_losses(
                        pred_vectors,
                        target_trace,
                        mask[i],
                        lambda_struct,
                        lambda_caca,
                    )
                    batch_loss += total
                    batch_struct += struct
                    batch_caca += caca
                    valid_count += 1
                    # --- CA-CA distance and vector norms logging ---
                    # Reconstruct CA trace from predicted vectors using mask
                    pred_trace = ca_trace_reconstruction_torch(
                        pred_vectors, mask[i]
                    )  # (seq_len, 3)
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
                    batch_caca /= valid_count
                    val_loss += batch_loss.item() * valid_count
                    val_struct_sum += batch_struct * valid_count
                    val_caca_sum += batch_caca * valid_count
                val_bar.set_postfix(
                    batch_loss=batch_loss.item() if valid_count > 0 else 0
                )
        N_val = len(val_loader.dataset)
        history["val_loss"].append(val_loss / N_val)
        history["val_struct_loss"].append(val_struct_sum / N_val)
        history["val_caca_loss"].append(val_caca_sum / N_val)
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
            print(
                f"[Validation to_prev_ca norm] mean: {mean_prev:.3f} Å, std: {std_prev:.3f} Å"
            )
        if to_next_norms_all:
            all_next_norms = torch.cat(to_next_norms_all)
            mean_next = all_next_norms.mean().item()
            std_next = all_next_norms.std().item()
            print(
                f"[Validation to_next_ca norm] mean: {mean_next:.3f} Å, std: {std_next:.3f} Å"
            )
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
