#!/usr/bin/env python
# coding: utf-8
"""
Shortest Path
"""

import networkx as nx
import numpy as np
import gurobipy as gp

from pyepo.model.opt import optModel


class shortestPathModelDijkstra(optModel):

    def __init__(self, graph, edges, source, target):
        """
        Args:
            graph (nx.graph): orignal complete graph
            edges (list(tuple)): remaining edges on the graph
            source (int): source node index
            target (int): target node index
        """
        self.graph = graph
        self.edges = edges
        self.source = source
        self.target = target
        super().__init__()

    def _getModel(self):
        """
        A method to build model

        Returns:
            tuple: optimization model and variables
        """
        # build graph as optimization model
        g = self.graph.edge_subgraph(self.edges)
        # init cost
        nx.set_edge_attributes(g, values=0, name="cost")
        return self.graph, self.graph.edges

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (ndarray): cost of objective function
        """
        m = len(self.graph.nodes)
        k = 0
        for i in range(m):
            for j in range(i+1,m):
                self.graph.edges[(i,j)]["cost"] = c[k]
                k += 1
        self._model = self.graph.edge_subgraph(self.edges)

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        # dijkstra
        path = nx.shortest_path(self._model, weight="cost", source=self.source, target=self.target, method="dijkstra")
        # convert path into active edges
        edges = []
        u = self.source
        for v in path[1:]:
            if u < v:
                edges.append((u,v))
            if u > v:
                edges.append((v,u))
            u = v
        # init sol & obj
        sol = np.zeros(len(self.graph.edges))
        obj = 0
        # convert active edges into solution and obj
        for i, e in enumerate(self.graph.edges):
            if e in edges:
                sol[i] = 1 # active edge
                obj += self._model.edges[e]["cost"] # cost of active edge
        return sol, obj


class shortestPathModelLP(optModel):

    def __init__(self, graph, edges, source, target):
        """
        Args:
            graph (nx.graph): orignal complete graph
            edges (list(tuple)): remaining edges on the graph
            source (int): source node index
            target (int): target node index
        """
        self.graph = graph
        self.edges = edges
        self.source = source
        self.target = target
        super().__init__()

    def _getModel(self):
        """
        A method to build model

        Returns:
            tuple: optimization model and variables
        """
        # build graph
        g = self.graph.edge_subgraph(self.edges).to_directed()
        # ceate a model
        m = gp.Model("shortest path")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        x = m.addVars(g.edges, name="x", ub=1) # extra 'ub=1' to avoid negative cycle
        for v in g.nodes:
            neighbors = [n for n in g.neighbors(v)]
            expr = gp.quicksum(x[u,v] for u in neighbors) - gp.quicksum(x[v,u] for u in neighbors)
            if v == self.source:
                m.addConstr(expr == -1)
            elif v == self.target:
                m.addConstr(expr == 1)
            else:
                m.addConstr(expr == 0)
        return m, x

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (ndarray): cost of objective function
        """
        # add cost
        m = len(self.graph.nodes)
        k = 0
        for i in range(m):
            for j in range(i+1,m):
                self.graph.edges[(i,j)]["cost"] = c[k]
                k += 1
        # subgraph
        g = self.graph.edge_subgraph(self.edges).to_directed()
        # objective function
        obj = 0
        for e in g.edges:
            obj += g.edges[e]["cost"] * self.x[e]
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.update()
        self._model.optimize()
        # sol
        sol = []
        for u, v in self.graph.edges:
            if (u,v) not in self.x:
                sol.append(0)
            else:
                sol.append(max(self.x[u,v].x, self.x[v,u].x))
        return sol, self._model.objVal
