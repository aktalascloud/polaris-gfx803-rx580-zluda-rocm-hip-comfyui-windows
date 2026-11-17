# SPDX-License-Identifier: GPL-3.0-only
"""Install the default bounded attention dispatcher into a stopped/fresh copy.

No model imports or GPU calls. Upstream bytes remain the prefix of attention.py;
rollback restores that verified prefix. No second backup of the source is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import stat

UPSTREAM_SHA256 = 'fae9969a9c80cae3a359af6032a5d77497836199df66fce46e22df89cd4a6cdc'
MARKER = '\n# POLARIS_BOUNDED_ATTENTION_V1\n'
HOOK = '''from .polaris_attention import attention as _polaris_tiled_attention
_polaris_upstream_sub_quad = attention_sub_quad

def attention_sub_quad(query, key, value, heads, mask=None, attn_precision=None,
                       skip_reshape=False, skip_output_reshape=False, **kwargs):
    q, k, v = query, key, value
    # The common small FP32 path needs no precision, allocator or training probes.
    if (mask is None and not kwargs and q.dtype == torch.float32 and
            k.dtype == torch.float32 and v.dtype == torch.float32 and
            q.ndim == k.ndim == v.ndim == (4 if skip_reshape else 3) and
            q.shape[-1] == v.shape[-1]):
        axis = 2 if skip_reshape else 1
        count = q.shape[0] * heads * q.shape[axis] * k.shape[axis] * 4
        if 0 < count < 8 * 1024 * 1024:
            return _polaris_upstream_sub_quad(q, k, v, heads, mask=mask,
                attn_precision=attn_precision, skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape)
    precision = get_attn_precision(attn_precision, q.dtype)
    training = torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v))
    supported = all(x.dtype in (torch.float16, torch.float32) for x in (q, k, v))
    if skip_reshape and q.ndim == 4 and k.ndim == 4:
        queries, keys = q.shape[2], k.shape[2]
    elif not skip_reshape and q.ndim == 3 and k.ndim == 3:
        queries, keys = q.shape[1], k.shape[1]
    else:
        supported = False
        queries = keys = 0
    # Keep the smaller stock path for ordinary small FP32 calls. Masked calls
    # use explicit boolean/additive semantics and zero fully masked rows.
    small = q.shape[0] * heads * queries * keys * 4 < 8 * 1024 * 1024
    stock_value_shape = v.ndim == q.ndim and v.shape[-1] == q.shape[-1]
    preserve_half = q.dtype == torch.float16 and args.dont_upcast_attention
    use_stock = (q.device.type != 'cuda' or training or not supported or kwargs or
                 preserve_half or (small and stock_value_shape and mask is None and all(x.dtype == torch.float32 for x in (q, k, v))
                                    and queries > 0 and keys > 0))
    if use_stock:
        return _polaris_upstream_sub_quad(q, k, v, heads, mask=mask,
            attn_precision=attn_precision, skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape, **kwargs)
    free, _ = model_management.get_free_memory(q.device, True)
    score_budget = min(32 * 1024 * 1024, max(1, int(free) // 8))
    if keys * 4 > score_budget:
        return _polaris_upstream_sub_quad(q, k, v, heads, mask=mask,
            attn_precision=attn_precision, skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape)
    return _polaris_tiled_attention(q, k, v, heads, mask=mask,
        attn_precision=precision, skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape, score_limit_bytes=score_budget)

if optimized_attention is _polaris_upstream_sub_quad:
    optimized_attention = attention_sub_quad
if optimized_attention_masked is _polaris_upstream_sub_quad:
    optimized_attention_masked = attention_sub_quad
'''


def normalized_digest(text):
    return hashlib.sha256(text.replace('\r\n', '\n').encode('utf-8')).hexdigest()


def status(root):
    folder = Path(root)/'comfy'/'ldm'/'modules'
    target = folder/'attention.py'
    if not target.exists():
        return dict(enabled=False, installed=False)
    text = target.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    original, separator, suffix = text.partition(MARKER)
    helper, receipt = folder/'polaris_attention.py', folder/'polaris_attention.json'
    info = json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else {}
    return dict(installed=bool(separator), enabled=bool(separator and suffix == HOOK),
                upstream_matches=normalized_digest(original) == UPSTREAM_SHA256,
                helper_matches=bool(helper.exists() and hashlib.sha256(helper.read_bytes()).hexdigest() == info.get('helper_sha256')))


def apply(root, rollback=False):
    raw = Path(root).absolute()
    folder = raw/'comfy'/'ldm'/'modules'
    target = folder/'attention.py'
    for item in [target, *target.parents]:
        if item.is_symlink() or getattr(item.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError('Reparse attention path refused')
        if item == raw:
            break
    text = target.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    original, separator, suffix = text.partition(MARKER)
    if normalized_digest(original) != UPSTREAM_SHA256:
        raise ValueError('Unsupported/modified attention.py; pinned ComfyUI source required')
    if separator and suffix != HOOK:
        raise ValueError('Existing attention hook differs; refusing overwrite')
    helper = folder/'polaris_attention.py'
    receipt = folder/'polaris_attention.json'
    for file in [helper, receipt]:
        if file.exists() and (file.is_symlink() or getattr(file.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError('Reparse attention artifact refused')
    payload = Path(__file__).with_name('polaris_attention.py').read_bytes()
    if helper.exists() and helper.read_bytes() != payload:
        raise ValueError('Conflicting attention helper; refusing overwrite')
    if not rollback:
        helper.write_bytes(payload)
    target.write_text(original if rollback else original+MARKER+HOOK, encoding='utf-8')
    info = dict(schema=1, enabled=not rollback, upstream_sha256=UPSTREAM_SHA256,
                helper_sha256=hashlib.sha256(payload).hexdigest(),
                policy='stock small FP32; tiled large/masked/FP16 inference; original explicit backends retained')
    receipt.write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['install', 'rollback', 'status'])
    parser.add_argument('root')
    args = parser.parse_args()
    print(json.dumps(status(args.root) if args.action == 'status' else apply(args.root, args.action == 'rollback')))
