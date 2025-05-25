import os
import sys
import json
import torch
import numpy as np
from protconv.utils.geometry import (
    fragment_to_ca_trace_vectors,
    ca_trace_reconstruction_torch,
    kabsch_torch,
)

# Example rotation (about z-axis by 45 degrees) and translation
ROTATION_MATRIX = torch.tensor(
    [
        [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
        [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
        [0, 0, 1],
    ],
    dtype=torch.float32,
)
TRANSLATION = torch.tensor([10.0, -5.0, 3.0], dtype=torch.float32)

DATA_DIR = os.path.join(os.getcwd(), "data/ProteinNet/casp7/fragments/")


def rmsd(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


def load_fragment(dataset_name, fragment_id):
    frag_path = os.path.join(DATA_DIR, f"{dataset_name}.json")
    with open(frag_path, "r") as f:
        fragments = json.load(f)
    frag = next((fr for fr in fragments if fr["id"] == fragment_id), None)
    if frag is None:
        raise ValueError(f"Fragment {fragment_id} not found in {dataset_name}")
    ca_coords = torch.tensor(frag["tertiary"], dtype=torch.float32)
    return ca_coords


def main(dataset_name, fragment_id):
    ca_coords = load_fragment(dataset_name, fragment_id)
    # Apply rotation and translation
    ca_transformed = torch.matmul(ca_coords, ROTATION_MATRIX.T) + TRANSLATION
    # Convert to ca_trace vectors
    ca_trace_vectors = fragment_to_ca_trace_vectors(ca_transformed)

    # Vector statistics
    into_ca_vectors = ca_trace_vectors[:, :3]
    out_from_vectors = ca_trace_vectors[:, 3:]
    into_norms = torch.norm(into_ca_vectors, dim=1)
    out_from_norms = torch.norm(out_from_vectors, dim=1)
    print(
        f"into_ca_vectors: min={into_norms.min():.2f}, max={into_norms.max():.2f}, mean={into_norms.mean():.2f}, std={into_norms.std():.2f}"
    )
    print(
        f"out_from_vectors: min={out_from_norms.min():.2f}, max={out_from_norms.max():.2f}, mean={out_from_norms.mean():.2f}, std={out_from_norms.std():.2f}"
    )

    # Save ca_trace as JSON
    frag_path = os.path.join(DATA_DIR, f"{dataset_name}.json")
    with open(frag_path, "r") as f:
        fragments = json.load(f)
    frag = next((fr for fr in fragments if fr["id"] == fragment_id), None)
    if frag is not None and "primary" in frag:
        primary = frag["primary"]
    else:
        primary = ["A"] * ca_trace_vectors.shape[0]
    ca_trace_json = {
        "id": fragment_id,
        "primary": primary,
        "ca_trace": [
            [into_ca_vectors[i].cpu().tolist(), out_from_vectors[i].cpu().tolist()]
            for i in range(ca_trace_vectors.shape[0])
        ],
    }
    out_json = os.path.join(DATA_DIR, f"{fragment_id}_{dataset_name}_ca_trace.json")
    with open(out_json, "w") as f:
        json.dump(ca_trace_json, f, indent=2)
    print(f"CA trace vectors saved as: {out_json}")

    # Reconstruct coordinates
    mask = torch.ones(ca_trace_vectors.shape[0], dtype=torch.bool)
    ca_reconstructed = ca_trace_reconstruction_torch(ca_trace_vectors, mask)
    # Align reconstructed to original (untransformed)
    ca_aligned = kabsch_torch(ca_reconstructed, ca_coords)
    # RMSD
    print(f"RMSD after round-trip and alignment: {rmsd(ca_aligned, ca_coords):.4f}")

    # Save aligned CA trace as PDB
    from protconv.utils import pdb as pdb_utils

    # Try to load the real sequence if possible
    frag_path = os.path.join(DATA_DIR, f"{dataset_name}.json")
    with open(frag_path, "r") as f:
        fragments = json.load(f)
    frag = next((fr for fr in fragments if fr["id"] == fragment_id), None)
    if frag is not None and "primary" in frag:
        primary = frag["primary"]
    else:
        primary = ["A"] * ca_aligned.shape[0]
    record = {
        "primary": primary,
        "tertiary": ca_aligned.cpu().numpy().tolist(),
    }
    out_pdb = os.path.join(DATA_DIR, f"{fragment_id}_{dataset_name}_reconstructed.pdb")
    pdb_utils.fragment_to_pdb(record, out_pdb)
    print(f"Aligned CA trace saved as: {out_pdb}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python -m protconv.scripts.analysis.analyze_geometry <dataset_name> <fragment_id>"
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
