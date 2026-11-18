# SPDX-License-Identifier: Apache-2.0
"""Prototype fresh, source-pinned HCU sparse decode scheduling metadata."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

_INTERFACE_SHA = '8ea8f6205ec41b1cd0d42bd231156ed26751fa4862dab15b81e3ec11ff1fd5e4'
_BINARY_SHA = '7ad05bfbb2fa67148b3e6bd6cf288f54ea1baa074438dc457d9d69b90bb92301'
_contract = None


def _native_matches(provider, factory, metadata):
    global _contract
    from flash_mla import flash_mla_interface as native
    current = (native.flash_mla_with_kvcache, native.get_mla_metadata,
               native.FlashMLASchedMeta, native.flash_mla_cuda.sparse_decode_fwd)
    if _contract is None:
        good = (hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() == _INTERFACE_SHA
                and hashlib.sha256(Path(native.flash_mla_cuda.__file__).read_bytes()).hexdigest() == _BINARY_SHA)
        _contract = current if good else ()
    return (_contract == current and provider is current[0]
            and factory is current[1] and type(metadata) is current[2])


if triton is not None:
    @triton.jit
    def _fresh_native_plan(Main, Extra, Plan, Splits,
                           ROWS: tl.constexpr, HAS_EXTRA: tl.constexpr):
        lane = tl.arange(0, 64)
        main = tl.load(Main + lane, lane < ROWS, other=0)
        blocks = tl.cdiv(tl.maximum(main, 1), 64)
        if HAS_EXTRA:
            extra = tl.load(Extra + lane, lane < ROWS, other=0)
            blocks += tl.cdiv(extra, 64)
        blocks = tl.where(lane < ROWS, blocks, 0)
        total = tl.sum(tl.where(lane < ROWS, blocks + 5, 0), 0)
        payload = tl.cdiv(total, 64) + 5
        smallest = tl.min(tl.where(lane < ROWS, blocks, 2147483647), 0)
        largest = tl.max(blocks, 0)
        capacity = payload - 5
        one_per = (capacity >= smallest) & (capacity - smallest <= 5)
        whole_chunks = (capacity < smallest) & (smallest % capacity == 0)
        if (smallest == largest) & (one_per | whole_chunks):
            chunks = tl.where(one_per, 1, smallest // capacity)
            chunk_size = tl.where(one_per, smallest, capacity)
            uniform_index = tl.arange(0, 512)
            uniform_part, uniform_field = uniform_index // 8, uniform_index % 8
            uniform_row, uniform_offset = uniform_part // chunks, uniform_part % chunks
            uniform_active = uniform_row < ROWS
            uniform_begin_req = tl.where(uniform_active, uniform_row, ROWS)
            uniform_end_req = tl.where(uniform_active, uniform_row, ROWS - 1)
            uniform_begin_block = tl.where(uniform_active, uniform_offset * chunk_size, 0)
            uniform_end_block = tl.where(uniform_active, (uniform_offset + 1) * chunk_size, smallest)
            uniform_split_idx = tl.where(uniform_active, uniform_offset, 0)
            uniform_split_flag = (uniform_active & (chunks > 1)).to(tl.int32)
            uniform_value = tl.where(uniform_field == 0, uniform_begin_req, 0)
            uniform_value = tl.where(uniform_field == 1, uniform_end_req, uniform_value)
            uniform_value = tl.where(uniform_field == 2, uniform_begin_block, uniform_value)
            uniform_value = tl.where(uniform_field == 3, uniform_end_block, uniform_value)
            uniform_value = tl.where(uniform_field == 4, uniform_split_idx, uniform_value)
            uniform_value = tl.where((uniform_field == 5) | (uniform_field == 6), uniform_split_flag, uniform_value)
            tl.store(Plan + uniform_index, uniform_value)
            uniform_row_index = tl.arange(0, 128)
            tl.store(Splits + uniform_row_index, uniform_row_index * chunks, uniform_row_index <= ROWS)
        else:
            req, block, split, cumulative = 0, 0, 0, 0
            tl.store(Splits, 0)
            for part in range(64):
                first_req, first_block, first_split = req, block, split
                starts_split = block != 0
                available = payload
                while (req < ROWS) & (available > 0):
                    index = tl.full((1,), 0, tl.int32) + req
                    size = tl.sum(tl.gather(blocks, index, 0), 0)
                    remaining = size - block
                    if available >= remaining + 5:
                        cumulative += split + 1
                        tl.store(Splits + req + 1, cumulative)
                        available -= remaining + 5
                        req += 1
                        block, split = 0, 0
                    else:
                        if available > 5:
                            block += available - 5
                            split += 1
                        available = 0
                end_req = tl.where(block > 0, req, req - 1)
                prev_index = tl.full((1,), 0, tl.int32) + tl.maximum(req - 1, 0)
                prev_size = tl.sum(tl.gather(blocks, prev_index, 0), 0)
                end_block = tl.where(block > 0, block, prev_size)
                end_index = tl.full((1,), 0, tl.int32) + tl.maximum(end_req, 0)
                end_size = tl.sum(tl.gather(blocks, end_index, 0), 0)
                ends_split = end_block != end_size
                shared_split = starts_split | ends_split
                starts_split = tl.where(first_req == end_req, shared_split, starts_split)
                ends_split = tl.where(first_req == end_req, shared_split, ends_split)
                field = tl.arange(0, 8)
                value = tl.where(field == 0, first_req, 0)
                value = tl.where(field == 1, end_req, value)
                value = tl.where(field == 2, first_block, value)
                value = tl.where(field == 3, end_block, value)
                value = tl.where(field == 4, first_split, value)
                value = tl.where(field == 5, starts_split.to(tl.int32), value)
                value = tl.where(field == 6, ends_split.to(tl.int32), value)
                tl.store(Plan + part * 8 + field, value)


def prepare_native_plan(main, extra, native_heads, main_width, extra_width, provider, factory, cache_blocks):
    """Keep a fresh native object; fill supported plans from current counts."""
    metadata = factory()[0]
    if (triton is None or main.ndim != 1 or not 0 < main.numel() <= 64
            or native_heads != 64 or not 0 < main_width <= 2048 or main_width % 64
            or len(cache_blocks) != 2 or cache_blocks[0] not in (32, 64)
            or cache_blocks[1] not in ((0,) if extra is None else (32, 64))
            or (extra is None) != (extra_width == 0)
            or (extra is not None and (extra.shape != main.shape or not 0 < extra_width <= 2048 or extra_width % 64))):
        return metadata
    tensors = (main,) if extra is None else (main, extra)
    if not all(x.is_cuda and x.device == main.device and x.dtype == torch.int32 and x.is_contiguous() for x in tensors):
        return metadata
    props = torch.cuda.get_device_properties(main.device)
    if (getattr(props, 'gcnArchName', '').split(':')[0] != 'gfx938'
            or props.multi_processor_count != 64 or not _native_matches(provider, factory, metadata)):
        return metadata
    plan = torch.empty((64, 8), device=main.device, dtype=torch.int32)
    splits = torch.empty((main.numel() + 1,), device=main.device, dtype=torch.int32)
    _fresh_native_plan[(1,)](main, extra, plan, splits, main.numel(), extra is not None, num_warps=1)
    metadata.tile_scheduler_metadata, metadata.num_splits = plan, splits
    return metadata
