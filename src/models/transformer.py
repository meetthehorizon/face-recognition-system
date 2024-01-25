import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, feat_dim, mlp_dim, num_heads, num_layers, dropout=0.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.feat_dim, self.mlp_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.mlp_dim, self.feat_dim),
                    nn.Dropout(self.dropout),
                )
                for _ in range(self.num_layers)
            ]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.feat_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    bias=False,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.attention_norm = nn.ModuleList(
            [nn.LayerNorm(self.feat_dim) for _ in range(self.num_layers)]
        )

        self.mlp_norm = nn.ModuleList(
            [nn.LayerNorm(self.feat_dim) for _ in range(self.num_layers)]
        )

    def forward(self, batch):
        for num, (attention, attention_norm, mlp, mlp_norm) in enumerate(
            zip(
                self.attentions,
                self.attention_norm,
                self.mlps,
                self.mlp_norm,
            )
        ):
            batch = batch + attention(batch, batch, batch)[0]
            batch = attention_norm(batch)
            batch = batch + mlp(batch)
            batch = mlp_norm(batch)

        return batch


import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, feat_dim, mlp_dim, num_heads, num_layers, dropout=0.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.feat_dim, self.mlp_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.mlp_dim, self.feat_dim),
                    nn.Dropout(self.dropout),
                )
                for _ in range(self.num_layers)
            ]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.feat_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    bias=False,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.attention_norm = nn.ModuleList(
            [nn.LayerNorm(self.feat_dim) for _ in range(self.num_layers)]
        )

        self.mlp_norm = nn.ModuleList(
            [nn.LayerNorm(self.feat_dim) for _ in range(self.num_layers)]
        )

    def forward(self, batch):
        for num, (attention, attention_norm, mlp, mlp_norm) in enumerate(
            zip(
                self.attentions,
                self.attention_norm,
                self.mlps,
                self.mlp_norm,
            )
        ):
            batch = batch + attention(batch, batch, batch)[0]
            batch = attention_norm(batch)
            batch = batch + mlp(batch)
            batch = mlp_norm(batch)

        return batch
