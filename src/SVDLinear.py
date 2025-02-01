import math
import mlx.core as mx
import mlx.nn as nn

class SVDLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Create an initial weight matrix with normal distribution.
        weight = mx.random.normal(shape=(out_features, in_features)) * (1.0 / math.sqrt(in_features))
        # Compute the SVD: weight = U @ diag(S) @ Vᵀ
        U_full, S_full, V_full = mx.linalg.svd(weight,stream=mx.cpu)
        # Thin SVD: keep only the first k singular values/vectors.
        k = in_features  # assuming in_features < out_features
        self.U = U_full[:, :k]
        self.S = S_full[:k]
        self.V = V_full[:, :k]
        # Store the SVD factors as non-trainable attributes.
        # self.freeze(keys=["U", "S", "V"])

        self.mask = mx.ones_like(self.S)
        self.bias = mx.zeros((out_features,), dtype=self.U.dtype)

    def __call__(self, x):
        # Recompose the weight using the learned mask:
        # new_weight = U @ diag(S * mask) @ Vᵀ
        new_weight = self.U @ mx.diag(self.S * self.mask) @ self.V.transpose()
        # Apply a normalization factor to preserve scale.
        norm_factor = self.S.sum() / ((self.S * self.mask).sum() + 1e-8)
        reweighted = new_weight * norm_factor
        # x is assumed to have shape (..., in_features)
        return x @ reweighted.transpose() + self.bias

class TransformerLMSVD(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, dims: int, num_heads: int, checkpoint: bool):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dims)
        self.pe = nn.SinusoidalPositionalEncoding(dims)
        self.transformer = nn.TransformerEncoder(
            num_layers, dims, num_heads, norm_first=True, checkpoint=checkpoint
        )
        # Use the SVD-based linear layer as the final projection.
        self.out_proj = SVDLinear(dims, vocab_size)

    def __call__(self, x):
        L = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        x = self.embedding(x)
        x = x + self.pe(mx.arange(L))
        x = self.transformer(x, mask)
        return self.out_proj(x)
