# SPDX-License-Identifier: Apache-2.0
"""Native BF16 inverse RoPE together with empty-row writeback."""
from __future__ import annotations

import torch
import hashlib
from pathlib import Path

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

_NATIVE_SOURCE_SHA256 = '002a018af1965eb13b2ffb9fd527bb0f2860d2a4b4eee352a0979c2c9529297c'
_native_contract = None


def _native_matches():
    global _native_contract
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as native
    if _native_contract is None:
        if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != _NATIVE_SOURCE_SHA256:
            _native_contract = ()
        else:
            _native_contract = (native._inverse_rope_gptj_kernel, native.rocm_inverse_rope_rows_)
    return _native_contract == (native._inverse_rope_gptj_kernel, native.rocm_inverse_rope_rows_)


if triton is not None:
    @triton.jit
    def _mask_copy_inverse_rope(Result, Output, Main, Extra, Positions, CS,
                                HEADS: tl.constexpr, HAS_EXTRA: tl.constexpr):
        row = tl.program_id(0)
        head = tl.program_id(1)
        base = (row * HEADS + head) * 512
        count = tl.load(Main + row)
        if HAS_EXTRA:
            count += tl.load(Extra + row)
        n = tl.arange(0, 512)
        value = tl.load(Result + base + n, (n < 448) & (count > 0), other=0.)
        tl.store(Output + base + n, value, n < 448)
        pos = tl.load(Positions + row).to(tl.int64)
        k = tl.arange(0, 32)
        a = tl.load(Result + base + 448 + 2 * k, count > 0, other=0.).to(tl.float32)
        b = tl.load(Result + base + 448 + 2 * k + 1, count > 0, other=0.).to(tl.float32)
        cos = tl.load(CS + pos * 64 + k)
        sin = tl.load(CS + pos * 64 + 32 + k)
        # Keep the source-pinned native expression and default fused arithmetic.
        # Empty rows rotate positive BF16 zeros too, preserving signed zeros.
        even = a * cos + b * sin
        odd = b * cos - a * sin
        tl.store(Output + base + 448 + 2 * k, even.to(tl.bfloat16))
        tl.store(Output + base + 448 + 2 * k + 1, odd.to(tl.bfloat16))


def try_mask_copy_rope(result, output, main_lens, extra_lens, positions, cos_sin):
    if triton is None or positions is None or cos_sin is None:
        return False
    tensors = (result, output, main_lens, positions, cos_sin) + (() if extra_lens is None else (extra_lens,))
    if (result.ndim != 3 or not 0 < result.shape[0] <= 64
            or result.shape[1:] not in ((64, 512), (128, 512))
            or output is None or output.shape != result.shape
            or result.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
            or not all(t.is_cuda and t.device == result.device and t.is_contiguous() for t in tensors)
            or main_lens.shape != (result.shape[0],) or main_lens.dtype != torch.int32
            or (extra_lens is not None and (extra_lens.shape != main_lens.shape or extra_lens.dtype != torch.int32))
            or positions.shape != main_lens.shape or positions.dtype not in (torch.int32, torch.int64)
            or cos_sin.ndim != 2 or cos_sin.shape[0] == 0
            or cos_sin.shape[1] != 64 or cos_sin.dtype != torch.float32):
        return False
    if getattr(torch.cuda.get_device_properties(result.device), 'gcnArchName', '').split(':')[0] != 'gfx938':
        return False
    if not _native_matches():
        return False
    _mask_copy_inverse_rope[(result.shape[0], result.shape[1])](
        result, output, main_lens, extra_lens, positions, cos_sin,
        result.shape[1], extra_lens is not None, num_warps=4)
    return True
