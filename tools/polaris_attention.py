# SPDX-License-Identifier: GPL-3.0-only
"""Inference attention with bounded query/head tiles.

Inspired by advanced-lvl-up's GPL ComfyUI jab-attention helper, rewritten to
preserve masks, precision and layout. No ComfyUI imports or model dependencies.
Boolean masks use True=attend. Rank-3 masks are [batch, query, key], as in ComfyUI;
rank-2 masks are [query, key]. Use [batch, 1, 1, key] for batched key padding.
"""
import torch


def _mask_view(mask, shape, device):
    if mask is None:
        return None
    if mask.device != device:
        raise ValueError('Attention mask and inputs must use the same device')
    if mask.dtype != torch.bool and not mask.is_floating_point():
        raise TypeError('Mask must be boolean or an additive floating-point bias')
    if mask.ndim == 1:
        mask = mask[None, None, None, :]
    elif mask.ndim == 2:
        mask = mask[None, None, :, :]
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]
    elif mask.ndim != 4:
        raise ValueError('Mask must have 1–4 dimensions')
    if any(size not in (1, target) for size, target in zip(mask.shape, shape)):
        raise ValueError('Mask is not broadcastable to [batch, heads, query, key]')
    return mask.expand(shape)  # View only; never materialize a full broadcast mask.


def attention(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
              skip_output_reshape=False, *, query_chunk_size=256,
              head_chunk_size=8, score_limit_bytes=32*1024*1024):
    """Return [B,Q,H*Dv] or [B,H,Q,Dv], retaining q's dtype.

    attn_precision=float32 upcasts before score multiplication, softmax and value
    projection, even inside autocast. FP16 inputs also use FP32 internally: the
    legacy backend lacks some sliced FP16 batched GEMMs. Output dtype is retained.
    Fully masked rows produce exact zeros.
    The budget limits one score tensor, not total VRAM; other intermediates exist.
    This implementation is for inference, not gradient-enabled training.
    """
    if not isinstance(heads, int) or heads < 1:
        raise ValueError('heads must be positive')
    if min(query_chunk_size, head_chunk_size, score_limit_bytes) < 1:
        raise ValueError('Tile sizes and score budget must be positive')
    if any(x.device != q.device for x in (k, v)):
        raise ValueError('Attention inputs must use one device')
    if q.dtype != k.dtype or any(x.dtype not in (torch.float16, torch.float32) for x in (q, k, v)):
        raise TypeError('Supported inputs are FP32/FP16 with matching query/key dtype')
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise ValueError('Use the original attention for gradient-enabled training')
    rank = 4 if skip_reshape else 3
    if any(x.ndim != rank for x in (q, k, v)):
        raise ValueError('Input rank does not match skip_reshape')
    if not skip_reshape:
        if any(x.shape[-1] % heads for x in (q, k, v)):
            raise ValueError('Channels must be divisible by heads')
        q, k, v = (x.reshape(x.shape[0], x.shape[1], heads, x.shape[2]//heads).transpose(1, 2)
                   for x in (q, k, v))
    batch, _, queries, depth = q.shape
    keys, value_depth = k.shape[2], v.shape[3]
    if depth < 1 or value_depth < 1 or any(x.shape[:2] != (batch, heads) for x in (q, k, v)):
        raise ValueError('Batch/head dimensions must agree and depths must be positive')
    if k.shape[3] != depth or v.shape[2] != keys:
        raise ValueError('Query/key depth and key/value token counts must agree')
    bias = _mask_view(mask, (batch, heads, queries, keys), q.device)
    if not queries or not keys or not batch:
        result = q.new_zeros((batch, heads, queries, value_depth))
        return result if skip_output_reshape else result.transpose(1, 2).reshape(batch, queries, heads*value_depth)
    upcast = attn_precision == torch.float32 or any(x.dtype == torch.float16 for x in (q, k, v))
    score_dtype = torch.float32 if upcast else q.dtype
    projection_dtype = torch.float32 if upcast else v.dtype
    element_bytes = 4 if score_dtype == torch.float32 else 2
    ht = min(heads, head_chunk_size, max(1, score_limit_bytes//(keys*element_bytes)))
    qt = min(queries, query_chunk_size, max(1, score_limit_bytes//(ht*keys*element_bytes)))
    if keys*element_bytes > score_limit_bytes:
        raise ValueError('One query/head exceeds the score budget; use the original KV-chunked attention')
    result = q.new_empty((batch, heads, queries, value_depth))
    with torch.autocast(device_type=q.device.type, enabled=False):
        for b in range(batch):
            for h in range(0, heads, ht):
                end = min(h+ht, heads)
                key_t = k[b, h:end].to(score_dtype).transpose(1, 2)
                value = v[b, h:end].to(projection_dtype)
                for start in range(0, queries, qt):
                    stop = min(start+qt, queries)
                    query = q[b, h:end, start:stop].to(score_dtype)
                    scores = torch.baddbmm(query.new_empty((1, 1, 1)), query,
                                          key_t, beta=0, alpha=depth**-0.5)
                    hidden = None
                    if bias is not None:
                        part = bias[b, h:end, start:stop]
                        if part.dtype == torch.bool:
                            scores.masked_fill_(~part, float('-inf'))
                            hidden = (~part).all(dim=-1, keepdim=True)
                        else:
                            part = part.to(score_dtype)
                            scores.add_(part)
                            hidden = (part == float('-inf')).all(dim=-1, keepdim=True)
                        scores.masked_fill_(hidden, 0)
                    probabilities = scores.softmax(dim=-1)
                    del scores
                    if hidden is not None:
                        probabilities.masked_fill_(hidden, 0)
                    output = torch.bmm(probabilities.to(projection_dtype), value)
                    if hidden is not None:
                        output.masked_fill_(hidden, 0)
                    result[b, h:end, start:stop].copy_(output)
                    del probabilities, output
    return result if skip_output_reshape else result.transpose(1, 2).reshape(batch, queries, heads*value_depth)
