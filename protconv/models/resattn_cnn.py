import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block: Conv1d -> BatchNorm -> ReLU -> Conv1d -> BatchNorm, with skip connection.
    """

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResAttn1DCNN(nn.Module):
    """
    1D CNN with residual blocks and self-attention for protein fragment structure regression.
    Input: (batch, seq_len) integer-encoded sequence
    Output: (batch, seq_len, 6) - two 3D vectors per residue
    """

    def __init__(
        self, vocab_size=21, embed_dim=32, hidden_dim=64, num_blocks=3, attn_heads=4
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.input_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock1D(hidden_dim) for _ in range(num_blocks)]
        )
        self.attn = nn.MultiheadAttention(hidden_dim, attn_heads, batch_first=True)
        self.out = nn.Linear(hidden_dim, 6)

    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = self.input_proj(x)  # (batch, hidden_dim, seq_len)
        for block in self.res_blocks:
            x = block(x)  # (batch, hidden_dim, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)
        # Self-attention
        attn_out, _ = self.attn(
            x, x, x, key_padding_mask=mask
        )  # attn_out: (batch, seq_len, hidden_dim)
        out = self.out(attn_out)  # (batch, seq_len, 6)
        return out


# Example usage:
# model = ResAttn1DCNN()
# x = torch.randint(0, 21, (batch_size, seq_len))
# mask = (x == padding_idx)  # Optional: for variable length masking
# output = model(x, mask=mask)
