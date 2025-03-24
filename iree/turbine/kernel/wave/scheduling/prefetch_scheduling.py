# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx
from ...ops.wave_ops import get_custom
from .graph_utils import Edge
from .resources import Operation, get_custom_operation_type
from collections import deque
from enum import Enum
from typing import Callable
import math


def get_root_nodes(edges: list[Edge]) -> list[fx.Node]:
    """
    Given scheduling edges, returns a list of node that
    do not have a ancestor, i.e are root nodes of the graph.
    """
    source_nodes = set()
    dst_nodes = set()
    for edge in edges:
        source_nodes.add(edge._from)
        dst_nodes.add(edge._to)
    root_nodes = source_nodes.difference(dst_nodes)
    return root_nodes


class PrefetchStage(Enum):
    GLOBAL_LOAD = 0
    LOCAL_STORE = 1
    LOCAL_LOAD = 2
    COMPUTE = 3

    def next(self):
        if self.value == 3:
            return PrefetchStage(3)
        v = self.value + 1
        return PrefetchStage(v)


operation_stage_table = {
    Operation.READ_SHARED: PrefetchStage.LOCAL_LOAD,
    Operation.WRITE_SHARED: PrefetchStage.LOCAL_STORE,
    Operation.READ_GLOBAL: PrefetchStage.GLOBAL_LOAD,
    Operation.MMA: PrefetchStage.COMPUTE,
    Operation.NOOP: PrefetchStage.COMPUTE,
    Operation.VALU: PrefetchStage.COMPUTE,
    Operation.SHUFFLE: PrefetchStage.COMPUTE,
}


def get_scheduling_stage(op: fx.Node) -> Operation:
    op_ty = get_custom_operation_type(get_custom(op))
    if op_ty not in operation_stage_table:
        raise NotImplementedError(f"Cannot find {op_ty} in operation_stage_table")
    return operation_stage_table[op_ty]


class PrefetchScheduler:
    """
    Prefetch Scheduler

    Convert vanilla schedule of:
        for i = 0 to N:
            a = READ_GLOBAL i
            WRITE_SHARED a
            barrier
            b = READ_SHARED
            COMPUTE b

    into prefetch schedule:
        a_0 = READ_GLOBAL 0
        WRITE_SHARED a_0
        for i = 0 to N - 1:
            a_{i+1} = READ_GLOBAL i + 1
            // a_{i+1} is NOT blocked by this barrier because barriers only block shared memory transfers
            barrier
            b_i = READ_SHARED
            COMPUTE b_i
            barrier
            WRITE_SHARED a_{i+1}
        barrier
        b_N = READ_SHARED
        COMPUTE b_N
    """

    def __init__(
        self,
        graph: fx.Graph,
        edges: list[Edge],
        resources: list[int],
    ) -> None:
        self.graph = graph
        self.edges = edges
        self.resources = resources
        self.seed = 2024

    def weighted_topological_sort(self, graph, edges):
        """
        Sort nodes based on scheduling weight.

        Where scheduling weight is defined as:
        scheduling_weight[node] = sum(edge.weight + scheduling_weight[edge.source] for edge in edges)

        scheduling weight is suppose to quantify how early or late a node should live in a graph.
        Similar to topoological ordering but more robust.

        This is achieved by setting up a workqueue which starts with root of graph.
        we then explore successors of nodes in workqueue and compute it's scheduling
        weight as defined in above formula. If ancestor/producer to current node has
        no value yet, we move current node to end of queue to try again later.
        """
        schedule_weight = {}
        root_nodes = get_root_nodes(edges)
        workqueue = deque(root_nodes)
        non_solved_counter = 0
        t_counter = 0
        while len(workqueue) > 0:
            node = workqueue.popleft()
            is_producer_edge = lambda edge: edge._to == node
            producers_edges = self.find_edges(is_producer_edge)
            filter_producer_edge = [
                edge
                for edge in producers_edges
                if edge.weight.iteration_difference == 0
            ]

            # Save for later if producer not registered yet.
            if any(
                [edge._from not in schedule_weight for edge in filter_producer_edge]
            ):
                # If we went over entire workqueue and still cannot find producer,
                # means it is missing producer from the edges.
                non_solved_counter += 1
                t_counter += 1
                if non_solved_counter >= len(workqueue):
                    raise ValueError(
                        "Cannot find producer(s) for remaining item in workqueue."
                    )
                print(t_counter)
                workqueue.append(node)
                continue

            non_solved_counter = 0
            schedule_weight[node] = sum(
                [
                    schedule_weight[edge._from] + edge.weight.delay
                    for edge in filter_producer_edge
                ]
            )
            is_consumer_edge = lambda edge: edge._from == node
            consumer_edges = self.find_edges(is_consumer_edge)
            consumer_nodes = [
                edge._to
                for edge in consumer_edges
                if edge.weight.iteration_difference == 0
            ]
            edge_weight = {edge._to: edge.weight.delay for edge in consumer_edges}
            breakpoint()
            consumer_nodes.sort(key=lambda item: (edge_weight[item], item))
            workqueue.extend(consumer_nodes)
        return sorted(graph.nodes, key=lambda x: schedule_weight[x])

    def prefetch_scheduling(self, graph: fx.Graph, edges: list[Edge]):
        """
        Classify node to different stages. Based on it's stage,
        program schedules clock for each node. This function also checks
        that sorted node "contiguously" move between stages.
        """
        sorted_nodes = self.weighted_topological_sort(graph, edges)
        schedule = {}
        current_stage = PrefetchStage.GLOBAL_LOAD
        for node in sorted_nodes:
            node_stage = get_scheduling_stage(node)
            next_stage = current_stage.next()
            if node_stage == current_stage:
                schedule[node] = current_stage.value
            elif node_stage == next_stage:
                schedule[node] = next_stage.value
                current_stage = next_stage
            else:
                # Node do not move contigously through stages.
                return {}, False
        return schedule, True

    def schedule_graph(self) -> tuple[dict[fx.Node, int], bool]:
        """
        1. Identify which nodes are part of the global_read/local_write/local_read/compute phase
        2. Set nodes to clock (0,1,2,3) based on phase.
        2. Set initiation interval to generate valid 2 stage prefetch.
        """
        self.schedule, success = self.prefetch_scheduling(self.graph, self.edges)
        self._initiation_interval = 2
        if self.num_stages != self._initiation_interval:
            return {}, False
        return self.schedule, success

    def find_edges(self, filter: Callable[[Edge], bool]) -> list[Edge]:
        filtered = []
        for edge in self.edges:
            if filter(edge):
                filtered.append(edge)
        return filtered

    @property
    def initiation_interval(self) -> int:
        """
        Returns the initiation interval of the schedule.
        """
        return self._initiation_interval

    @property
    def num_stages(self) -> int:
        """
        Returns the number of stages in the kernel of the pipelined loop.
        """
        max_cycle = max([t for t in self.schedule.values()])
        return math.ceil(max_cycle / self.initiation_interval)
