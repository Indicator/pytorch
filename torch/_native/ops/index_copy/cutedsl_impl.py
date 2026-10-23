from functools import partial

import torch

from ... import cutedsl_utils as cu
from ..scatter_add.cutedsl_impl import (
    _base_cond_ok,
    _is_vec_scatter_supported,
    _prepare_kernel_inputs,
    _scatter_add_eligibility,
    _vs_kernel,
)


def _cond(self, dim, index, source, *, out=None, functional=False):
    dst = self if out is None else out
    if not _base_cond_ok(self, index, source, dst):
        return False
    if self.ndim < 2 or dim not in (0, -self.ndim):
        return False
    if source.ndim != self.ndim or self.shape[1:] != source.shape[1:]:
        return False
    if index.ndim > 1 or index.dtype != torch.int64 or not index.is_contiguous():
        return False
    if index.numel() != source.shape[0] or source.numel() == 0 or self.numel() == 0:
        return False
    if dst.shape != self.shape or dst.dtype != self.dtype:
        return False
    if any(t.device != self.device for t in (index, source, dst)):
        return False
    # The functional output is contiguous, so its slices must preserve the
    # memory order approved by TensorIterator for self and source.
    if functional and not dst.select(0, 0).is_contiguous():
        return False
    if dst.stride(0) < dst.select(0, 0).numel():
        return False
    if torch._C._overlaps(dst, source) or torch._C._overlaps(dst, index):
        return False
    expanded = index.view(-1, *([1] * (source.ndim - 1))).expand_as(source)
    return _is_vec_scatter_supported(dst, 0, expanded, source, accumulate=False)


def _inplace_impl(self, dim, index, source):
    expanded = index.view(-1, *([1] * (source.ndim - 1))).expand_as(source)
    it = _scatter_add_eligibility(self, 0, expanded, source)
    if it is None:
        raise RuntimeError("index_copy cutedsl: cond approved but iter rebuild failed")
    dst, index_1d, src = _prepare_kernel_inputs(self, 0, expanded, source, it)
    _vs_kernel()(dst, index_1d, src, accumulate=False)
    return self


def _impl(self, dim, index, source):
    out = self.clone(memory_format=torch.contiguous_format)
    return _inplace_impl(out, dim, index, source)


def _out_impl(self, dim, index, source, *, out):
    if out is not self:
        out.copy_(self)
    return _inplace_impl(out, dim, index, source)


def register_to_dispatch() -> None:
    for op, fn in (
        ("index_copy", _impl),
        ("index_copy.out", _out_impl),
        ("index_copy_", _inplace_impl),
    ):
        cond = partial(_cond, functional=True) if op == "index_copy" else _cond
        cu.register_op_override("aten", op, "CUDA", cond=cond, impl=fn)
