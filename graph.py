#!/usr/bin/env python
# coding: utf-8
"""
Subgraph of different tasks
"""

import random

import networkx as nx
import numpy as np


def genSubgraphSP(graph, num_edges, seed=42):
    """
    Generate subgraphs for shortest path
    """
    # random seed
    random.seed(seed)
    while True:
        # sample edges
        edges = random.sample(list(graph.edges), k=num_edges)
        # subgraph
        subgraph = graph.edge_subgraph(edges)
        # check connectivity
        if (len(subgraph) == len(graph.nodes)) and \
           (len([subgraph.subgraph(v) for v in nx.connected_components(subgraph)]) == 1):
            while True:
                # random source and target
                s, t = random.sample(list(subgraph.nodes), 2)
                # check length
                if len(nx.shortest_path(subgraph, source=s, target=t)) > 3:
                    break
            yield subgraph, edges, s, t


def genSubgraphTSP(graph, seed=42):
    """
    Generate complete subgraphs for TSP
    """
    # random seed
    random.seed(seed)
    np.random.seed(seed)
    # nodes num
    m = len(graph.nodes)
    while True:
        # random size
        ms = random.randint(m//2, m*3//4)
        # nodes sampling
        nodes = np.sort(np.random.choice(m, ms, replace=False))
        # subgraph
        subgraph = graph.subgraph(nodes)
        yield subgraph, nodes
