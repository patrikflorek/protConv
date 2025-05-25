# protConv: 1D CNN for Protein Tertiary Structure Prediction

`protConv` provides a workflow for predicting the C-alpha backbone of polypeptides from their primary amino acid sequence using a 1D Convolutional Neural Network (CNN). The project leverages curated data from the CASP7 ProteinNet dataset.

## Overview

- **Objective:** Predict the C-alpha backbone coordinates of polypeptides from their primary sequence.
- **Method:** Train and apply a 1D CNN model using processed ProteinNet data.

## Project Structure

- **Scripts:** Located in `protconv/scripts/`
- **Raw Data:** CASP7 ProteinNet files (`testing`, `training_30`, `validation`) in `data/ProteinNet/casp7/`
- **Extracted Fragments:** `data/ProteinNet/casp7/fragments/`

## Data Preparation & Workflow

### 1. Convert ProteinNet to JSON

Converts ProteinNet files to JSON for each dataset split.

```bash
python -m protconv proteinnet_to_json <dataset_name>
# Example:
python -m protconv proteinnet_to_json validation
```
- Output: `data/ProteinNet/casp7/<dataset_name>.json`
- Prints summary statistics (number of records, sequence lengths)

**Example Output Record:**
```json
{
  "id": "10#1R8K_1_A_0002-0328",
  "primary": "EVPLPQL...",
  "tertiary": [float, float, float, ...],
  "mask": [bool, bool, bool, ...]
}
```

- `primary`: Amino acid sequence (20-character alphabet)
- `tertiary`: 3D backbone coordinates (N, C_alpha, C'), as real-valued 3x3 matrices per residue, in picometers (×100 PDB units)
- `mask`: Boolean mask indicating presence of atomic coordinates

See `pdb.proteinnet_to_pdb` and `pdb.fragment_to_pdb` for implementation details.

### 1a. Analyze ProteinNet Example

Analyze a ProteinNet record, export as PDB, and print statistical reports.

```bash
python -m protconv analysis.analyze_proteinnet_example <dataset_name> <example_id>
# Example:
python -m protconv analysis.analyze_proteinnet_example validation 70#2AIO_1_A
```
- Output: `data/ProteinNet/casp7/validation_70#2AIO_1_A.pdb`
- Prints statistical reports: number of residues, number with defined structure, atom coordinate statistics (min, max, mean, std)

### 2. Extract Fragments

Extract fragments (length ≥ 3) with defined C-alpha atoms.

```bash
python -m protconv extract_fragments <dataset_name>
# Example:
python -m protconv extract_fragments validation
```
- Output: `data/ProteinNet/casp7/fragments/<dataset_name>.json`
- Prints statistics on number of fragments and fragment length (min, max, mean)

#### Example Output Record (Fragment)
```json
{
  "id": "70#2AIO_1_A_0001-0266",
  "primary": "EVPLPQL...",
  "tertiary": [[[float, float, float], [float, float, float], [float, float, float]], ...],
}
```

### 2a. Analyze Fragment Example
Analyzes a Fragment example.

```bash
python -m protconv analysis.analyze_fragment_example <dataset_name> <example_id>
```
- Example: `python -m protconv analysis.analyze_fragment_example validation 70#2AIO_1_A_0001-0266`
- Output: `data/ProteinNet/casp7/fragments/validation_70#2AIO_1_A_0001-0266.pdb`

## Roadmap / Future Steps
*(The following are planned features and are intentionally retained for project continuity)*

- Model training and evaluation pipelines
- Expanded dataset support and augmentation

## Dataset Preparation
- **Source:** [CASP7 ProteinNet dataset](https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp7.tar.gz)
- **Reference:** [ProteinNet Records Documentation](https://github.com/aqlaboratory/proteinnet/blob/master/docs/proteinnet_records.md) (note: secondary structure records are not available in this release)

## Usage
Run scripts using the module syntax:

```bash
python -m _contrib <script_name>
```
Replace `<script_name>` with one of the scripts above (without `.py`).

Please refer to individual script docstrings for detailed arguments and options.
