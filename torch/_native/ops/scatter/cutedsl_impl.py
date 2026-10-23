from functools import partial

import torch

from ... import cutedsl_utils as cu
from ..scatter_add.cutedsl_impl import (
    _alignment_contract_ok,
    _base_cond_ok,
    _is_vec_scatter_supported,
    _make_impls,
    _normalize_dim,
    _vs_kernel,
)


def _cond(self, dim, index, src):
    if not _base_cond_ok(self, index, src):
        return False
    if not _is_vec_scatter_supported(self, dim, index, src):
        return False
    if self.numel() == 0 or index.numel() == 0:
        return False
    d = _normalize_dim(dim, self.ndim)
    if any(
        index.size(i) > src.size(i) or (i != d and index.size(i) > self.size(i))
        for i in range(self.ndim)
    ):
        return False
    if self.stride(d) < self.numel() // self.size(d):
        return False
    if torch._C._overlaps(self, src) or torch._C._overlaps(self, index):
        return False
    return _alignment_contract_ok(self, src, d, dst_ptr_align=16)


_, _, _inplace_impl = _make_impls(lambda: partial(_vs_kernel(), accumulate=False))


def register_to_dispatch() -> None:
    cu.register_op_override(
        "aten", "scatter_.src", "CUDA", cond=_cond, impl=_inplace_impl
    )
