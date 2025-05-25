"""
loader.py
---------
Utility to provide PyTorch DataLoaders for training and validation.
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.getcwd(), "data/ProteinNet/casp7/fragments/")

class FragmentDataset(Dataset):
    def __init__(self, dataset_name, max_len=700):
        path = os.path.join(DATA_DIR, f"{dataset_name}.json")
        with open(path, "r") as f:
            self.fragments = json.load(f)
        self.max_len = max_len

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx):
        frag = self.fragments[idx]
        seq = frag["primary"]
        coords = frag["tertiary"]
        seq_tensor = torch.tensor([ord(aa) for aa in seq], dtype=torch.long)
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        return seq_tensor, coords_tensor

def collate_batch(batch):
    seqs, coords = zip(*batch)
    max_len = max(s.shape[0] for s in seqs)
    seqs_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    coords_padded = torch.zeros(len(batch), max_len, 3, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (s, c) in enumerate(zip(seqs, coords)):
        l = s.shape[0]
        seqs_padded[i, :l] = s
        coords_padded[i, :l, :] = c
        mask[i, :l] = 1
    return seqs_padded, coords_padded, mask

def get_train_val_loaders(batch_size=16, max_len=700, num_workers=0):
    train_set = FragmentDataset("training_30", max_len=max_len)
    val_set = FragmentDataset("validation", max_len=max_len)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    return train_loader, val_loader
