"""
Sanity check script for protein CA backbone prediction models.
- Loads a trained model and validation set
- Selects random validation examples
- Predicts CA traces, compares to targets, computes RMSD
- Saves both predicted and target traces as PDB files
"""
import os
import random
import torch
from protconv.utils.geometry import ca_trace_reconstruction_torch, kabsch_torch
from protconv.utils.pdb import save_ca_trace_as_pdb  # Assumed utility
from protconv.scripts.train import get_train_val_loaders
from protconv.models.latent_cnn import Latent1DCNN  # Change as needed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sanity_check_model(
    model_path,
    num_examples=3,
    output_dir="sanity_check_outputs",
    model_class=Latent1DCNN,
    batch_size=32,
    vocab_size=21,
):
    os.makedirs(output_dir, exist_ok=True)
    _, val_loader = get_train_val_loaders(batch_size=batch_size)
    model = model_class(vocab_size=vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    # Gather all val examples
    val_examples = []
    for seqs, coords, mask in val_loader:
        for i in range(seqs.size(0)):
            seq_len = int(mask[i].sum())
            if seq_len < 3:
                continue
            val_examples.append((seqs[i], coords[i], mask[i]))
    if len(val_examples) == 0:
        print("No valid validation examples found.")
        return
    chosen = random.sample(val_examples, min(num_examples, len(val_examples)))
    for idx, (seq, coord, mask) in enumerate(chosen):
        seq = seq.unsqueeze(0).to(DEVICE)  # (1, L)
        mask = mask.bool()
        with torch.no_grad():
            pred_vectors = model(seq)[0]  # (L, 6)
        pred_trace = ca_trace_reconstruction_torch(pred_vectors, mask)
        target_trace = coord[mask, :]
        # Align and compute RMSD
        aligned_pred = kabsch_torch(pred_trace, target_trace)
        rmsd = ((aligned_pred - target_trace) ** 2).sum(-1).sqrt().mean().item()
        print(f"Example {idx+1}: Kabsch RMSD = {rmsd:.3f} Å")
        # Save PDBs
        pred_pdb_path = os.path.join(output_dir, f"pred_{idx+1}.pdb")
        target_pdb_path = os.path.join(output_dir, f"target_{idx+1}.pdb")
        save_ca_trace_as_pdb(pred_trace.cpu().numpy(), pred_pdb_path)
        save_ca_trace_as_pdb(target_trace.cpu().numpy(), target_pdb_path)
        print(f"Saved: {pred_pdb_path}, {target_pdb_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sanity check for CA backbone prediction model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of validation examples to check")
    parser.add_argument("--output_dir", type=str, default="sanity_check_outputs", help="Where to save PDBs")
    args = parser.parse_args()
    sanity_check_model(args.model_path, args.num_examples, args.output_dir)
