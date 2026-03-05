"""Stub xformers.ops module for systems without CUDA.

Provides fallback implementations that use standard PyTorch attention.
Audiocraft imports xformers but defaults to 'torch' backend, so these
stubs just need to exist for the import to succeed.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Any, Union


class AttentionBias:
    """Stub attention bias class."""
    pass


class LowerTriangularMask(AttentionBias):
    """Stub lower triangular mask."""
    pass


class BlockDiagonalMask(AttentionBias):
    """Stub block diagonal mask."""
    pass


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[AttentionBias] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Fallback to standard scaled dot-product attention."""
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=p if query.requires_grad else 0.0,
        scale=scale,
        is_causal=isinstance(attn_bias, LowerTriangularMask),
    )


def unbind(x: torch.Tensor, dim: int = 0) -> Tuple[torch.Tensor, ...]:
    """Unbind tensor - fallback to torch.unbind."""
    return torch.unbind(x, dim=dim)


# Alias for compatibility
fmha = type('fmha', (), {
    'memory_efficient_attention': memory_efficient_attention,
    'Inputs': type('Inputs', (), {}),
    'AttentionBias': AttentionBias,
    'LowerTriangularMask': LowerTriangularMask,
    'BlockDiagonalMask': BlockDiagonalMask,
})()


# Additional stubs that audiocraft might check for
class MemoryEfficientAttentionCutlassOp:
    pass


class MemoryEfficientAttentionFlashAttentionOp:
    pass


class MemoryEfficientAttentionCutlassFwdFlashBwOp:
    pass
