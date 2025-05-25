"""
proteinnet_to_json.py
--------------------
Converts CASP7 ProteinNet-formatted text files to JSON format for downstream processing.

Usage:
    python -m protconv proteinnet_to_json <dataset_name>

Input files are expected in the data/ProteinNet/casp7/ directory.
Output JSON is written to the same directory with a .json extension.
"""

import os
import json

DATA_DIR = os.path.join(os.getcwd(), "data/ProteinNet/casp7")


def get_id(buffer):
    """
    Concatenate lines representing the record ID.

    Args:
        buffer (list[str]): Lines containing the ID.
    Returns:
        str: Concatenated ID string.
    """
    return "".join([line for line in buffer if line])


def get_primary(buffer):
    """
    Concatenate lines representing the primary amino acid sequence.

    Args:
        buffer (list[str]): Lines containing the sequence.
    Returns:
        str: Amino acid sequence.
    """
    return "".join([line for line in buffer if line])


def get_tertiary(buffer):
    """
    Parse and flatten lines of tertiary structure coordinates into a float list.

    Args:
        buffer (list[str]): Lines with whitespace-separated floats.
    Returns:
        list[float]: Flattened list of coordinates.
    """
    tertiary_floats = []
    for line in buffer:
        tertiary_floats.extend([float(x) for x in line.split()])
    return tertiary_floats


def get_mask(buffer):
    """
    Parse mask lines into a list of booleans.

    Args:
        buffer (list[str]): Lines with '+' or '-' characters.
    Returns:
        list[bool]: True for '+', False for '-'.
    """
    mask_bools = []
    for line in buffer:
        mask_bools.extend([x == "+" for x in line if x in "+-"])
    return mask_bools


def parse_record(lines):
    """
    Parse a block of lines representing a single ProteinNet record into a dictionary.

    Args:
        lines (list[str]): Lines for one record, including section headers (e.g., [ID], [PRIMARY]).
    Returns:
        dict: Parsed record with keys 'id', 'primary', 'tertiary', and 'mask' (if present).
    """
    record = {}
    key = None
    buffer = []
    for line in lines:
        # Section header indicates a new field
        if line.startswith("[") and line.endswith("]"):
            if key and buffer:
                record[key] = buffer
                buffer = []
            key = line[1:-1]
        else:
            buffer.append(line)
    if key and buffer:
        record[key] = buffer

    clean_record = {}
    for key, buffer in record.items():
        if key == "ID":
            clean_record["id"] = get_id(buffer)
        elif key == "PRIMARY":
            clean_record["primary"] = get_primary(buffer)
        elif key == "TERTIARY":
            clean_record["tertiary"] = get_tertiary(buffer)
        elif key == "MASK":
            clean_record["mask"] = get_mask(buffer)

    return clean_record


def parse_dataset(input_path):
    """
    Parse a ProteinNet dataset file into a list of structured records.

    Args:
        input_path (str): Path to the ProteinNet-formatted text file.
    Returns:
        list[dict]: List of parsed records, each as a dictionary.
    """
    records = []
    with open(input_path, "r") as f:
        lines = []
        for line in f:
            sline = line.strip()
            # Each new record starts with [ID]
            if sline.startswith("[ID]"):
                if lines:
                    record = parse_record(lines)
                    records.append(record)
                    lines = []
            lines.append(sline)
        # Don't forget the last record
        if lines:
            record = parse_record(lines)
            records.append(record)

    return records


def main(dataset_name):
    """
    Main entry point: parses a ProteinNet dataset and writes it to JSON.

    Args:
        dataset_name (str): Name of the dataset file (without .json extension).
    Raises:
        FileNotFoundError: If the input file does not exist.
    Side Effects:
        Writes parsed records to a JSON file and prints summary statistics.
    """
    input_path = os.path.join(DATA_DIR, dataset_name)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Dataset file {input_path} does not exist")

    output_path = os.path.join(DATA_DIR, dataset_name + ".json")

    try:
        records = parse_dataset(input_path)
        with open(output_path, "w") as f:
            json.dump(records, f)
        print(f"Success: Output written to {output_path}")

        # Print dataset statistics
        num_records = len(records)
        seq_lengths = [len(r["primary"]) for r in records if "primary" in r]
        if seq_lengths:
            min_len = min(seq_lengths)
            max_len = max(seq_lengths)
            avg_len = sum(seq_lengths) / len(seq_lengths)
        else:
            min_len = max_len = avg_len = 0
        print(f"Records: {num_records}")
        print(f"Sequence length: min={min_len}, max={max_len}, avg={avg_len:.2f}")

    except Exception as e:
        print(f"Failed to parse and save dataset {input_path}: {e}")
        return


if __name__ == "__main__":
    raise NotImplementedError(
        "Call script with `python -m protconv proteinnet_to_json <dataset_name>`"
    )
