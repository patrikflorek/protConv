"""
extract_fragments.py
-------------------
Extracts contiguous fragments with defined C-alpha atoms from ProteinNet records.

For each record, fragments of minimum length (default 3) are extracted where the mask indicates structure is defined. Outputs all fragments as a JSON list.

Reports:
- Number of fragments
- Fragment length statistics (min, max, mean)

Usage:
    python -m protconv extract_fragments <dataset_name>
"""

import os
import json
import math

DATA_DIR = os.path.join(os.getcwd(), "data/ProteinNet/casp7")
OUTPUT_DIR = os.path.join(DATA_DIR, "fragments")


def fragments_from_record(record, min_len=3):
    """
    Extract contiguous fragments with defined structure from a ProteinNet record.

    Args:
        record (dict): ProteinNet record with 'primary', 'mask', and 'tertiary'.
        min_len (int): Minimum fragment length (default 3).
    Returns:
        list[dict]: List of fragment records with 'id', 'primary', and 'tertiary'.
    """
    fragment_indices = []
    start = None
    for i, m in enumerate(record["mask"]):
        if m == 1:
            if start is None:
                start = i
        else:
            if start is not None and (i - start) >= min_len:
                fragment_indices.append((start, i - 1))
            start = None
    # Handle fragment ending at last residue
    if start is not None and (len(record["mask"]) - start) >= min_len:
        fragment_indices.append((start, len(record["mask"]) - 1))
    fragment_records = []
    for fragment_start, fragment_end in fragment_indices:
        fragment_id = f"{record['id']}_{fragment_start:04d}-{fragment_end:04d}"
        fragment_primary = record["primary"][fragment_start : fragment_end + 1]
        fragment_tertiary = [
            [
                record["tertiary"][i * 3 + 1] / 100,
                record["tertiary"][i * 3 + len(record["primary"]) * 3 + 1] / 100,
                record["tertiary"][i * 3 + len(record["primary"]) * 6 + 1] / 100,
            ]
            for i in range(fragment_start, fragment_end + 1)
        ]

        fragment_records.append(
            {
                "id": fragment_id,
                "primary": fragment_primary,
                "tertiary": fragment_tertiary,
            }
        )

    return fragment_records


def main(dataset_name):
    """
    Extract fragments from all records in a dataset, write to JSON, and print statistics.

    Args:
        dataset_name (str): Name of the dataset split (e.g., 'validation').
    Side Effects:
        Writes extracted fragments to a JSON file and prints statistics.
    """
    dataset_path = os.path.join(DATA_DIR, dataset_name + ".json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    all_fragments = []
    fragment_lengths = []
    all_ca_distances = []
    for record in dataset:
        fragments = fragments_from_record(record)
        all_fragments.extend(fragments)
        fragment_lengths.extend(len(f["primary"]) for f in fragments)
        # Aggregate CA-CA distances for each fragment
        for frag in fragments:
            tertiary = frag["tertiary"]
            ca_distances = [
                math.sqrt(
                    (tertiary[i][0] - tertiary[i - 1][0]) ** 2
                    + (tertiary[i][1] - tertiary[i - 1][1]) ** 2
                    + (tertiary[i][2] - tertiary[i - 1][2]) ** 2
                )
                for i in range(1, len(tertiary))
            ]
            all_ca_distances.extend(ca_distances)

    output_path = os.path.join(OUTPUT_DIR, dataset_name + ".json")
    with open(output_path, "w") as f:
        json.dump(all_fragments, f)

    print(f"Success: Output written to {output_path}")
    print(f"Number of fragments: {len(all_fragments)}")
    if fragment_lengths:
        print(
            f"Fragment length: min={min(fragment_lengths)}, max={max(fragment_lengths)}, mean={sum(fragment_lengths)/len(fragment_lengths):.2f}"
        )
    if all_ca_distances:
        mean_dist = sum(all_ca_distances) / len(all_ca_distances)
        std_dist = (
            sum((x - mean_dist) ** 2 for x in all_ca_distances) / len(all_ca_distances)
        ) ** 0.5
        print(
            f"CA-CA distances (all fragments): min={min(all_ca_distances):.2f}, max={max(all_ca_distances):.2f}, mean={mean_dist:.2f}, std={std_dist:.2f}"
        )
    else:
        print(
            "CA-CA distances (all fragments): not enough residues to compute distances."
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m protconv extract_fragments <dataset_name>")
        sys.exit(1)
    main(sys.argv[1])
