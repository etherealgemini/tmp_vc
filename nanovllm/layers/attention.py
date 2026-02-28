import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# ---------------------------------------------------------------------------
# Attention-weight capture utilities
# ---------------------------------------------------------------------------
# When capture mode is enabled, every LLM Attention.forward call appends a
# Tensor[num_heads, sq, sk] (CPU float32 softmax weights) to _ATTN_CAPTURE_STORE
# in layer order.  The store is cleared on enable and on retrieval.
# ---------------------------------------------------------------------------

_ATTN_CAPTURE_ENABLED: bool = False
_ATTN_CAPTURE_STORE: list = []


def set_attn_capture(enabled: bool) -> None:
    global _ATTN_CAPTURE_ENABLED
    _ATTN_CAPTURE_ENABLED = enabled
    if enabled:
        _ATTN_CAPTURE_STORE.clear()


def clear_attn_capture() -> None:
    _ATTN_CAPTURE_STORE.clear()


def get_attn_capture() -> list:
    """Return a copy of the captured attention-weight list and clear the store."""
    result = list(_ATTN_CAPTURE_STORE)
    _ATTN_CAPTURE_STORE.clear()
    return result


def _gather_kv_from_paged_cache(
    cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """Reconstruct a [seq_len, num_kv_heads, head_dim] tensor from paged cache.

    Args:
        cache: [num_blocks, block_size, num_kv_heads, head_dim]
        block_tables: [max_blocks] for a single sequence (1-D)
        seq_len: number of valid tokens
        block_size: tokens per block
    """
    chunks = []
    remaining = seq_len
    for blk_id in block_tables.tolist():
        if blk_id < 0 or remaining <= 0:
            break
        take = min(block_size, remaining)
        chunks.append(cache[blk_id, :take])   # [take, num_kv_heads, head_dim]
        remaining -= take
    return torch.cat(chunks, dim=0)            # [seq_len, num_kv_heads, head_dim]


def _compute_attn_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> list:
    """Compute softmax attention weights (for visualization only).

    Returns a list of Tensor[num_heads, sq, sk] — one entry per batch item.
    The computation is done in float32 for numerical stability.
    """
    cu_q = context.cu_seqlens_q
    cu_k = context.cu_seqlens_k
    block_tables = context.block_tables   # [batch, max_blocks] or None
    block_size = k_cache.shape[1] if (k_cache is not None and k_cache.numel() > 0) else 256

    batch = len(cu_q) - 1
    all_weights = []

    for b in range(batch):
        q_s = cu_q[b].item()
        q_e = cu_q[b + 1].item()
        k_s = cu_k[b].item()
        k_e = cu_k[b + 1].item()
        sq = q_e - q_s
        sk = k_e - k_s

        q_b = q[q_s:q_e].float()   # [sq, num_heads, head_dim]

        if block_tables is not None and k_cache is not None and k_cache.numel() > 0:
            k_b = _gather_kv_from_paged_cache(k_cache, block_tables[b], sk, block_size).float()
            v_b = _gather_kv_from_paged_cache(v_cache, block_tables[b], sk, block_size).float()
        else:
            k_b = k[k_s:k_e].float()   # [sk, num_kv_heads, head_dim]
            v_b = v[k_s:k_e].float()

        # Expand GQA groups
        if num_heads != num_kv_heads:
            g = num_heads // num_kv_heads
            k_b = k_b.unsqueeze(2).expand(-1, -1, g, -1).reshape(sk, num_heads, head_dim)
            v_b = v_b.unsqueeze(2).expand(-1, -1, g, -1).reshape(sk, num_heads, head_dim)

        # scores: [num_heads, sq, sk]
        q_t = q_b.permute(1, 0, 2)          # [nh, sq, hd]
        k_t = k_b.permute(1, 2, 0)          # [nh, hd, sk]
        scores = torch.bmm(q_t, k_t) * scale  # [nh, sq, sk]

        # Causal mask: query position i corresponds to absolute position (offset + i)
        # where offset = sk - sq (number of cached/prefix tokens)
        offset = sk - sq
        q_pos = torch.arange(sq, device=q.device).unsqueeze(1)   # [sq, 1]
        k_pos = torch.arange(sk, device=q.device).unsqueeze(0)   # [1, sk]
        causal = k_pos <= (q_pos + offset)                        # [sq, sk]
        scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))

        weights = F.softmax(scores, dim=-1)  # [nh, sq, sk]
        all_weights.append(weights.detach().cpu())

    return all_weights


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        # Capture attention weights for analysis when requested (prefill only).
        # Analysis runs are always single-sequence batches, so we store the first
        # (and only) element of weights_list as a Tensor[num_heads, sq, sk].
        if _ATTN_CAPTURE_ENABLED and context.is_prefill:
            weights_list = _compute_attn_weights(
                q, k, v, k_cache, v_cache, context,
                self.scale, self.num_heads, self.num_kv_heads, self.head_dim,
            )
            _ATTN_CAPTURE_STORE.append(weights_list[0])
        return o
