"""
analyze_fragment_example.py
--------------------------
Analyze a single fragment example: export as PDB and print statistical reports.

Reports include:
- Fragment length (number of residues)
- Atom coordinate statistics (min, max, mean, std)

Usage:
    python -m protconv analysis.analyze_fragment_example <dataset_name> <example_id>
"""

import os
import json
import math
import protconv.utils.pdb as pdb_utils

DATA_DIR = os.path.join(os.getcwd(), "data/ProteinNet/casp7/fragments/")


def main(dataset_name, example_id):
    """
    Analyze a fragment record: export as PDB and print statistics.

    Args:
        dataset_name (str): Name of the fragment dataset split (e.g., 'validation').
        example_id (str): Fragment record ID to analyze.
    """
    dataset_path = os.path.join(DATA_DIR, dataset_name + ".json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    record = next(r for r in dataset if r["id"] == example_id)
    pdb_path = os.path.join(DATA_DIR, record["id"] + "_" + dataset_name + ".pdb")

    # Export to PDB
    pdb_utils.fragment_to_pdb(record, pdb_path)
    print(f"Example {example_id} written to {pdb_path}")

    # --- Statistical reports ---
    primary = record.get("primary", "")
    tertiary = record.get("tertiary", [])

    n_residues = len(primary)
    print(f"Fragment length (number of residues): {n_residues}")

    # Atom coordinate statistics (flattened list)
    if tertiary:
        # Flatten nested list of coordinates
        coords_flat = [c for atom in tertiary for c in atom]
        print(
            f"Atom coordinates: min={min(coords_flat):.2f}, max={max(coords_flat):.2f}, mean={sum(coords_flat)/len(coords_flat):.2f}, std={(sum((x - sum(coords_flat)/len(coords_flat))**2 for x in coords_flat)/len(coords_flat))**0.5:.2f}"
        )

        # CA-CA distances
        ca_distances = [
            math.sqrt(
                (tertiary[i][0] - tertiary[i - 1][0]) ** 2
                + (tertiary[i][1] - tertiary[i - 1][1]) ** 2
                + (tertiary[i][2] - tertiary[i - 1][2]) ** 2
            )
            for i in range(1, len(tertiary))
        ]
        if ca_distances:
            mean_dist = sum(ca_distances) / len(ca_distances)
            std_dist = (
                sum((x - mean_dist) ** 2 for x in ca_distances) / len(ca_distances)
            ) ** 0.5
            print(
                f"CA-CA distances: min={min(ca_distances):.2f}, max={max(ca_distances):.2f}, mean={mean_dist:.2f}, std={std_dist:.2f}"
            )
        else:
            print("CA-CA distances: not enough residues to compute distances.")
    else:
        print("No atom coordinate data available.")
