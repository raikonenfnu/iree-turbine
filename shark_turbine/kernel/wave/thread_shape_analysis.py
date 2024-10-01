# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ...support.logging import get_logger
from shark_turbine.kernel._support.tracing import CapturedTrace
import torch.fx as fx
from ..ops.wave_ops import *
from ..lang.global_symbols import *
from .utils import capture_forward_slice, capture_backward_slice

logger = get_logger("turbine.wave.hoisting")


def get_allocs(graph: fx.Graph) -> list[CustomOp]:
    return [
        custom_node
        for node in graph.nodes
        if isinstance((custom_node := get_custom(node)), Allocate)
    ]


@dataclass(order=True)
class IndexSize:
    index: IndexSymbol
    size: int

    def __hash__(self):
        return hash((self.index, self.size))


def get_index_sizes(indices: list[IndexSequence]):
    dims = frozenset([IndexSize(index, seq.size) for index, seq in indices.items()])
    return dims


def get_custom_index_sizes(custom: CustomOp):
    return get_index_sizes(custom.index)


def set_index_size(custom: CustomOp, target_index_sizes: IndexSize):
    for target in target_index_sizes:
        if target.index not in custom.index:
            raise NotImplementedError(
                "NYI: Handle when source target index size is not found in target/user index."
            )
        custom.index[target.index].size = target.size


def determine_thread_shapes(trace: CapturedTrace):
    """Insert broadcasts to binary ops operands that requires it."""

    # Anchor ops are ops who's thread shape are predetermined.
    anchorOpTypes = (Read, Write, MMA, ReduceOp)
    noHandleTypes = (Placeholder, Output, Broadcast, ExtractSlice, Allocate)
    nonPropagatableTypes = anchorOpTypes + noHandleTypes

    def is_anchor_op(node: fx.Node):
        return isinstance(get_custom(node), anchorOpTypes)

    def propagatable_op(node: fx.Node):
        return not isinstance(get_custom(node), nonPropagatableTypes)

    anchor_ops = trace.walk(is_anchor_op)
    thread_size_to_ops_map = {}
    for anchor_op in anchor_ops:
        custom = get_custom(anchor_op)
        index_sizes = get_custom_index_sizes(custom)
        if isinstance(custom, (Read, ReduceOp)):
            fwd_slice = capture_forward_slice(custom.fx_node, propagatable_op)
            fwd_slice.remove(custom.fx_node)
            thread_size_to_ops_map[index_sizes] = thread_size_to_ops_map.get(
                index_sizes, set([])
            ).union(fwd_slice)
        elif isinstance(custom, Write):
            bwd_slice = capture_backward_slice(custom.fx_node, propagatable_op)
            bwd_slice.remove(custom.fx_node)
            thread_size_to_ops_map[index_sizes] = thread_size_to_ops_map.get(
                index_sizes, set([])
            ).union(bwd_slice)
        elif isinstance(custom, MMA):
            lhs_bwd_slice = capture_backward_slice(custom.lhs, propagatable_op)
            rhs_bwd_slice = capture_backward_slice(custom.rhs, propagatable_op)
            acc_slice = capture_forward_slice(custom.acc, propagatable_op)
            acc_slice.union(capture_backward_slice(custom.acc, propagatable_op))
            acc_index = get_index_sizes(custom.acc_index)
            lhs_index = get_index_sizes(custom.lhs_index)
            rhs_index = get_index_sizes(custom.rhs_index)
            thread_size_to_ops_map[acc_index] = thread_size_to_ops_map.get(
                acc_index, set([])
            ).union(acc_slice)
            thread_size_to_ops_map[lhs_index] = thread_size_to_ops_map.get(
                lhs_index, set([])
            ).union(lhs_bwd_slice)
            thread_size_to_ops_map[rhs_index] = thread_size_to_ops_map.get(
                rhs_index, set([])
            ).union(rhs_bwd_slice)

    # Apply supposedly indecummulative_setpedent sets by their bucketed thread shape.
    cummulative_set = set()
    for target_index_size, target_ops in thread_size_to_ops_map.items():
        # Ensure that we do not have any conflicts.
        if not cummulative_set.isdisjoint(target_ops):
            raise NotImplementedError("NYI: Handling of conflicting thread shape.")
        cummulative_set = cummulative_set.union(target_ops)
        for user in target_ops:
            custom_user = get_custom(user)
            set_index_size(custom_user, target_index_size)
            if isinstance(custom_user, IterArg):
                init_args = custom_user.get_src_reduction().init_args[
                    custom_user.get_iter_idx()
                ]
                set_index_size(get_custom(init_args), target_index_size)
