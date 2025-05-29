"""
Latent1DCNN: Encoder–latent–decoder 1D CNN for protein CA backbone prediction.
The latent space is exposed for future manipulation/experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Latent1DCNN(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=64, cnn_channels=128, latent_dim=32, out_dim=6, seq_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global pooling
        self.to_latent = nn.Linear(cnn_channels, latent_dim)
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, out_dim, kernel_size=3, padding=1),
        )
        self.latent = None  # Exposed for future use

    def encode(self, x):
        x = self.embed(x).transpose(1, 2)  # (B, embed_dim, L)
        h = self.encoder(x)  # (B, cnn_channels, L)
        pooled = self.pool(h).squeeze(-1)  # (B, cnn_channels)
        latent = self.to_latent(pooled)    # (B, latent_dim)
        return latent

    def decode(self, latent, seq_len):
        # latent: (B, latent_dim)
        # Repeat latent for each sequence position
        x = latent.unsqueeze(-1).repeat(1, 1, seq_len)  # (B, latent_dim, L)
        out = self.decoder(x)  # (B, out_dim, L)
        return out.transpose(1, 2)  # (B, L, out_dim)

    def forward(self, x, return_latent=False, latent_override=None):
        # x: (B, L) sequence indices
        latent = self.encode(x) if latent_override is None else latent_override
        self.latent = latent  # Expose for future use
        out = self.decode(latent, seq_len=x.shape[1])
        if return_latent:
            return out, latent
        return out
