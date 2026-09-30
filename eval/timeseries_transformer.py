"""Patch Transformer encoder for the offline 60-minute Maps-duration forecast.

Design (sized for ~6k training windows on one route; see docs/deep-learning-assessment.md):

1. **Patch tokens.** The 36-step (3 h) input window of per-step channels is cut into
   non-overlapping patches of ``patch_len`` steps (default 6 = 30 min, so 6 tokens). Each patch is
   flattened and projected to ``d_model``. Patching (Nie et al., 2023, "PatchTST") gives each token
   local shape information and keeps attention to a handful of tokens instead of 36 noisy steps.
2. **Learned positional embedding** per patch position (the window length is fixed).
3. **Pre-LayerNorm encoder blocks** (``n_layers``): LN -> multi-head self-attention -> dropout ->
   residual; LN -> feed-forward (GELU) -> dropout -> residual. Pre-LN trains stably without
   learning-rate warm-up, which matters with few epochs. Attention is not masked: every token is
   at or before the forecast origin, so there is no look-ahead inside the window.
4. **Flatten head.** Final LN, flatten all tokens, dropout, one linear output (PatchTST's head;
   it keeps "where in the window" information that average pooling would discard).
5. **Persistence-anchored target** (applied by ``deep_forecast.fit_sequence_model``): the network
   predicts the standardised change from the value at the forecast origin, so an untrained or
   heavily regularised model falls back to persistence rather than to the training mean.

The default budget (``d_model=32``, 4 heads, 2 blocks, ``ff_dim=64``) is about 19k weights, close
to LSTM(64) in ``timeseries_lstm`` (about 21k), so the comparison is about architecture rather
than size. Training (loss, optimiser, early stopping, seeds) is shared with the recurrent models
in ``deep_forecast``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from timeseries_lstm import _require_keras


@dataclass(frozen=True)
class TransformerConfig:
    patch_len: int = 6
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 64
    dropout: float = 0.1
    attention_dropout: float = 0.1
    head_dropout: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.patch_len < 1 or self.n_layers < 1:
            raise ValueError("patch_len and n_layers must be >= 1")


def n_patches(window_size: int, patch_len: int) -> int:
    if window_size % patch_len:
        raise ValueError(f"window_size {window_size} is not a multiple of patch_len {patch_len}")
    return window_size // patch_len


def _layers():
    """Custom layers, defined lazily so importing this module does not import TensorFlow."""
    keras = _require_keras()

    class PatchEmbedding(keras.layers.Layer):
        """(batch, steps, channels) -> (batch, patches, d_model)."""

        def __init__(self, patch_len: int, d_model: int, **kwargs):
            super().__init__(**kwargs)
            self.patch_len = patch_len
            self.d_model = d_model
            self.proj = keras.layers.Dense(d_model, name="patch_proj")

        def call(self, x):
            steps, channels = x.shape[1], x.shape[2]
            patches = keras.ops.reshape(x, (-1, steps // self.patch_len, self.patch_len * channels))
            return self.proj(patches)

        def get_config(self):
            return {**super().get_config(), "patch_len": self.patch_len, "d_model": self.d_model}

    class PositionalEmbedding(keras.layers.Layer):
        """Adds a learned vector per token position."""

        def build(self, input_shape):
            self.pos = self.add_weight(
                name="pos",
                shape=(int(input_shape[1]), int(input_shape[2])),
                initializer=keras.initializers.RandomNormal(stddev=0.02),
            )

        def call(self, x):
            return x + self.pos

    return keras, PatchEmbedding, PositionalEmbedding


def build_transformer_model(input_shape: Tuple[int, int], config: TransformerConfig = TransformerConfig()):
    """Uncompiled Keras model: (steps, channels) -> one standardised output."""
    keras = _require_keras(config.seed)
    keras, PatchEmbedding, PositionalEmbedding = _layers()
    L = keras.layers
    steps, _channels = input_shape
    tokens = n_patches(steps, config.patch_len)

    inputs = keras.Input(shape=input_shape, name="sequence")
    x = PatchEmbedding(config.patch_len, config.d_model, name="patch_embedding")(inputs)
    x = PositionalEmbedding(name="positional_embedding")(x)
    x = L.Dropout(config.dropout, name="embedding_dropout")(x)
    for i in range(config.n_layers):
        h = L.LayerNormalization(epsilon=1e-6, name=f"block{i}_ln_attn")(x)
        h = L.MultiHeadAttention(
            num_heads=config.n_heads,
            key_dim=config.d_model // config.n_heads,
            dropout=config.attention_dropout,
            name=f"block{i}_attention",
        )(h, h)
        x = L.Add(name=f"block{i}_res_attn")([x, L.Dropout(config.dropout)(h)])
        h = L.LayerNormalization(epsilon=1e-6, name=f"block{i}_ln_ff")(x)
        h = L.Dense(config.ff_dim, activation="gelu", name=f"block{i}_ff1")(h)
        h = L.Dropout(config.dropout)(h)
        h = L.Dense(config.d_model, name=f"block{i}_ff2")(h)
        x = L.Add(name=f"block{i}_res_ff")([x, L.Dropout(config.dropout)(h)])
    x = L.LayerNormalization(epsilon=1e-6, name="final_ln")(x)
    x = L.Reshape((tokens * config.d_model,), name="flatten_tokens")(x)
    x = L.Dropout(config.head_dropout, name="head_dropout")(x)
    outputs = L.Dense(1, name="delta")(x)
    return keras.Model(inputs, outputs, name="causeway_patch_transformer")
