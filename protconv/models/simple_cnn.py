import torch
import torch.nn as nn


class Simple1DCNN(nn.Module):
    """
    Simple 1D CNN for regression on protein fragment sequences.
    Input: (batch, seq_len)
    Output: (batch, seq_len, 6) - two 3D vectors per residue (to_prev_ca, to_next_ca)
    """

    def __init__(self, vocab_size=21, embed_dim=32, hidden_dim=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.out = nn.Conv1d(hidden_dim, 6, kernel_size=1)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = self.relu(self.conv1(x))  # (batch, hidden_dim, seq_len)
        x = self.relu(self.conv2(x))  # (batch, hidden_dim, seq_len)
        x = self.out(x)  # (batch, 6, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, 6)
        return x
