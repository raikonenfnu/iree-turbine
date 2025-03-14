# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx
from ....support.logging import get_logger
from .graph_utils import Edge
from collections import deque
from functools import cmp_to_key
from typing import Callable
import numpy as np
import math

logger = get_logger("turbine.wave.modulo_scheduling")


def get_root_nodes(edges: list[Edge]) -> list[fx.Node]:
    source_nodes = set()
    dst_nodes = set()
    for edge in edges:
        source_nodes.add(edge._from)
        dst_nodes.add(edge._to)
    root_nodes = source_nodes.difference(dst_nodes)
    return root_nodes


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
        self.T0 = math.ceil(self.compute_resource_ii())

    def weighted_topological_sort(self, graph, edges):
        """Sort based on weight and then topological"""
        schedule_weight = {}
        root_nodes = get_root_nodes(edges)
        workqueue = deque(root_nodes)
        non_solved_counter = 0
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
                if non_solved_counter >= workqueue:
                    raise ValueError(
                        "Cannot find producer(s) for remaining item in workqueue."
                    )
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
            workqueue.extend(consumer_nodes)
        return sorted(graph.nodes, key=lambda x: schedule_weight[x])

    def earliest_start_fn(self, schedule, RT, node):
        # Check for earliest schedule based on data dependence.
        is_producer_edge = lambda edge: edge._to == node
        producers_edges = self.find_edges(is_producer_edge)
        filter_producer_edge = [
            edge for edge in producers_edges if edge.weight.iteration_difference == 0
        ]
        producer_starts = [
            schedule[edge._from] + edge.weight.delay for edge in filter_producer_edge
        ] or [0]
        earliest_data_start = max(producer_starts)

        # Apply resource constraints.
        earliest_start = None
        for i in range(self.T0):
            cur_start = earliest_data_start + i
            cur_resource = RT[cur_start : cur_start + node.rrt.shape[0]]
            candidate_resource = cur_resource + node.rrt
            if np.all(candidate_resource <= self.resources):
                earliest_start = cur_start
                break

        # Check if found valid solution.
        if earliest_start == None:
            raise ValueError(
                "Cannot find start loc for node for prefetch, try other scheduling."
            )

        return earliest_start

    def list_scheduling(self, graph: fx.Graph, edges: list[Edge]):
        sorted_nodes = self.weighted_topological_sort(graph, edges)
        RT = np.zeros((len(sorted_nodes), len(self.resources)))
        schedule = {}
        for node in sorted_nodes:
            s = self.earliest_start_fn(schedule, RT, node)
            RT[s : s + node.rrt.shape[0]] += node.rrt
            schedule[node] = s
        return schedule

    def schedule_graph(self) -> tuple[dict[fx.Node, int], bool]:
        """
        1. Resource List scheduling to get clock
        2. Identify which clocks is part of the read/write/compute phase
        3. Set initiation interval to be S(write_shared.last).
        """
        self.schedule = self.list_scheduling(self.graph, self.edges)
        # TODO: Cluster ops into 4 stage of scheduling:
        #       1. GLOBAL_LOAD stage
        #       2. LOCAL_STORE stage
        #       3. LOCAL_LOAD stage
        #       4. COMPUTE stage
        # TODO: Implement initiation interval extractor based on the bucketing.
        # self._initiation_interval = self.find_ii_for_two_stage_prefetch()
        self._initiation_interval = 3
        success = True
        return self.schedule, success

    def compute_resource_ii(self) -> int:
        """
        Compute the resource constrained initiation interval.
        """
        usage = np.zeros(len(self.resources))
        for node in self.graph.nodes:
            usage += np.sum(node.rrt, axis=0)
        usage /= self.resources
        logger.debug(f"Resource constrained initiation interval: {np.max(usage)}.")
        return np.max(usage)

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
