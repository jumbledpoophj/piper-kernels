"""Model-neutral attention shapes and configuration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

type AttentionInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class AttentionShape:
    """The logical dimensions of an attention invocation."""

    batch_size: int
    num_query_heads: int
    query_length: int
    key_value_length: int
    head_dim: int
    num_key_value_heads: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.batch_size,
            self.num_query_heads,
            self.query_length,
            self.key_value_length,
            self.head_dim,
        )
        if any(value <= 0 for value in values):
            raise ValueError("attention dimensions must be positive")
        if self.num_key_value_heads is not None and self.num_key_value_heads <= 0:
            raise ValueError("key/value heads must be positive")
        if self.num_query_heads % self.effective_num_key_value_heads:
            raise ValueError("query heads must be divisible by key/value heads")

    @property
    def effective_num_key_value_heads(self) -> int:
        """Return the explicit or implicit number of key/value heads."""
        return self.num_key_value_heads or self.num_query_heads

    def as_dict(self) -> dict[str, int]:
        """Return stable machine-readable field names."""
        return {
            "batch_size": self.batch_size,
            "num_query_heads": self.num_query_heads,
            "num_key_value_heads": self.effective_num_key_value_heads,
            "query_length": self.query_length,
            "key_value_length": self.key_value_length,
            "head_dim": self.head_dim,
        }


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    """Common non-shape settings for attention providers."""

    dtype: str
    is_causal: bool = False
    scale: float | None = None
    qkv_layout: str = "BHSD"
    value_bias_amplitude: float = 0.0

    def as_dict(self) -> dict[str, str | bool | float | None]:
        """Return stable machine-readable field names."""
        return {
            "dtype": self.dtype,
            "is_causal": self.is_causal,
            "scale": self.scale,
            "qkv_layout": self.qkv_layout,
            "value_bias_amplitude": self.value_bias_amplitude,
        }


def make_attention_inputs(
    shape: AttentionShape,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    value_bias_amplitude: float = 0.0,
) -> AttentionInputs:
    """Create reproducible random Q/K/V tensors for an attention shape."""
    query = torch.randn(
        (shape.batch_size, shape.num_query_heads, shape.query_length, shape.head_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    key_shape = (
        shape.batch_size,
        shape.effective_num_key_value_heads,
        shape.key_value_length,
        shape.head_dim,
    )
    key = torch.randn(key_shape, device=device, dtype=dtype, generator=generator)
    value = torch.randn(key_shape, device=device, dtype=dtype, generator=generator)
    if value_bias_amplitude:
        bias = torch.linspace(
            -value_bias_amplitude,
            value_bias_amplitude,
            shape.head_dim,
            device=device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, shape.head_dim)
        value = (value.float() + bias).to(dtype)
    return query, key, value
