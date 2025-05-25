"""
analyze_proteinnet_example.py
-----------------------------
Analyze a single ProteinNet example: export as PDB and print statistical reports.

Reports include:
- Number of residues
- Number of residues with defined structure (mask)
- Atom coordinate statistics (min, max, mean, std)

Usage:
    python -m protconv analysis.analyze_proteinnet_example <dataset_name> <example_id>
"""

import os
import json
import statistics
import protconv.utils.pdb as pdb_utils

DATA_DIR = os.path.join(os.getcwd(), "data/ProteinNet/casp7")


def main(dataset_name, example_id):
    """
    Analyze a ProteinNet record: export as PDB and print statistics.

    Args:
        dataset_name (str): Name of the dataset split (e.g., 'validation').
        example_id (str): Record ID to analyze.
    """
    dataset_path = os.path.join(DATA_DIR, dataset_name + ".json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    record = next(r for r in dataset if r["id"] == example_id)
    pdb_path = os.path.join(DATA_DIR, record["id"] + "_" + dataset_name + ".pdb")

    # Export to PDB
    pdb_utils.proteinnet_to_pdb(record, pdb_path)
    print(f"Example {example_id} written to {pdb_path}")

    # --- Statistical reports ---
    primary = record.get("primary", "")
    mask = record.get("mask", [])
    tertiary = record.get("tertiary", [])

    n_residues = len(primary)
    n_defined = sum(mask) if mask else 0
    print(f"Number of residues: {n_residues}")
    print(f"Residues with defined structure: {n_defined}")

    # Atom coordinate statistics (flattened list)
    if tertiary:
        coords = tertiary
        min_coord = min(coords)
        max_coord = max(coords)
        mean_coord = statistics.mean(coords)
        stdev_coord = statistics.stdev(coords) if len(coords) > 1 else 0.0
        print(
            f"Atom coordinates: min={min_coord:.2f}, max={max_coord:.2f}, "
            f"mean={mean_coord:.2f}, std={stdev_coord:.2f}"
        )
    else:
        print("No atom coordinate data available.")
