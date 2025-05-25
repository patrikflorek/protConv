"""
pdb.py
------
Protein structure utility functions for exporting ProteinNet records and fragments to PDB format.

Functions:
- proteinnet_to_pdb(record, output_path): Export full ProteinNet record (N, CA, C atoms per residue).
- fragment_to_pdb(record, output_path): Export fragment (CA atom per residue).

Coordinates are converted from picometers to angstroms (divide by 100).
"""

AA_TO_TRIPLET = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


def proteinnet_to_pdb(record, output_path):
    """
    Write a ProteinNet record to a PDB file, including N, CA, C atoms for each residue.

    Args:
        record (dict): ProteinNet record with 'primary' (sequence) and 'tertiary' (flat coords).
        output_path (str): Output PDB file path.
    """
    primary = record["primary"]
    tertiary = record["tertiary"]
    n_residues = len(primary)
    chain_identifier = "A"

    with open(output_path, "w") as f:
        for i, aa in enumerate(primary):
            residue_name = AA_TO_TRIPLET[aa]
            residue_seq = i + 1
            # Indices for N, CA, C atoms in flat tertiary array
            base = i * 3
            offsets = [0, 1, 2]
            atoms = [
                (
                    "N",
                    (
                        tertiary[base + o],
                        tertiary[base + n_residues * 3 + o],
                        tertiary[base + n_residues * 6 + o],
                    ),
                )
                for o in offsets
            ]
            atom_names = ["N", "CA", "C"]
            for j, (atom_name, coords) in enumerate(zip(atom_names, atoms)):
                serial = i * 3 + j + 1
                x, y, z = (c / 100 for c in coords[1])
                f.write(
                    f"ATOM  {serial:>5}  {atom_name:<4}{residue_name:<3} {chain_identifier:<1}{residue_seq:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00 0.00           {atom_name[0]}\n"
                )
        f.write("END\n")


def fragment_to_pdb(record, output_path):
    """
    Write a fragment record to a PDB file, including only CA atoms for each residue.

    Args:
        record (dict): Fragment record with 'primary' (sequence) and 'tertiary' (Nx3 coords).
        output_path (str): Output PDB file path.
    """
    primary = record["primary"]
    tertiary = record["tertiary"]
    chain_identifier = "A"

    with open(output_path, "w") as f:
        for i, aa in enumerate(primary):
            residue_name = AA_TO_TRIPLET[aa]
            residue_seq = i + 1
            serial = i + 1
            atom_name = "CA"
            x, y, z = tertiary[i]
            f.write(
                f"ATOM  {serial:>5}  {atom_name:<4}{residue_name:<3} {chain_identifier:<1}{residue_seq:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00 0.00           {atom_name[0]}\n"
            )
        f.write("END\n")
