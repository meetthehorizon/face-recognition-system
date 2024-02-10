import torch
import torch.nn as nn


class Transformer(nn.Module):
    """Transformer

    Reference: https://arxiv.org/pdf/1706.03762.pdf"""

    def __init__(self, feat_dim, mlp_dim, num_heads, num_layers, dropout=0.0):
        """
        Parameters
        ----------
        feat_dim : int
            Dimension of hidden feature vector
        mlp_dim : int
            Dimension of MLP in transformer
        num_heads : int
            Number of heads in transformer
        num_layers : int
            Number of layers of MSA and MLP used
        dropout : float
            Dropout rate
        """
        super().__init__()

        # Multi-layer perceptron
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feat_dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, feat_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        # Multi-head self-attention
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=feat_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        # Layer normalisation
        self.attention_norm = nn.ModuleList(
            [nn.LayerNorm(feat_dim) for _ in range(num_layers)]
        )
        self.mlp_norm = nn.ModuleList(
            [nn.LayerNorm(feat_dim) for _ in range(num_layers)]
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        """Forward pass of transformer
        Parameters
        ----------
        batch : torch.Tensor
            Batch of feature vectors of shape (batch_size, seq_len, feat_dim)
        Returns
        -------
        batch : torch.Tensor
            Embeddings of shape (batch_size, seq_len, feat_dim)
        """
        for attention, attention_norm, mlp, mlp_norm in zip(
            self.attentions,
            self.attention_norm,
            self.mlps,
            self.mlp_norm,
        ):
            batch = batch + attention(batch, batch, batch)[0]
            batch = attention_norm(batch)
            batch = batch + mlp(batch)
            batch = mlp_norm(batch)

        return batch


if __name__ == "__main__":
    print("passed")
