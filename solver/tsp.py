#!/usr/bin/env python
# coding: utf-8
"""
Traveling salesperson
"""

from collections import defaultdict
from itertools import combinations
import copy

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from pyepo.model.opt import optModel

class tspModel(optModel):

    def __init__(self, graph, ind):
        """
        Args:
            graph (nx.graph): orignal complete graph
            ind (list(int)): nodes to obtain a TSP tour
        """
        self.graph = graph
        self.ind = list(ind)
        super().__init__()

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # get subgraph
        g = self.graph.subgraph(self.ind)
        # ceate a model
        m = gp.Model("tsp")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        x = m.addVars(g.edges, name="x", vtype=GRB.BINARY)
        for i, j in g.edges:
            x[j, i] = x[i, j]
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(x.sum(i, "*") == 2 for i in g.nodes)  # 2 degree
        # activate lazy constraints
        m._x = x
        m._ind = self.ind
        m._n = len(g.nodes)
        m.Params.lazyConstraints = 1
        return m, x

    @staticmethod
    def _subtourelim(model, where):
        """
        A static method to add lazy constraints for subtour elimination
        """
        def subtour(selected, ind):
            """
            find shortest cycle
            """
            unvisited = copy.deepcopy(ind)
            # init dummy longest cycle
            cycle = ind + [ind[0]]
            while unvisited:
                thiscycle = []
                neighbors = unvisited
                while neighbors:
                    current = neighbors[0]
                    thiscycle.append(current)
                    unvisited.remove(current)
                    neighbors = [
                        j for i, j in selected.select(current, "*")
                        if j in unvisited
                    ]
                if len(cycle) > len(thiscycle):
                    cycle = thiscycle
            return cycle

        if where == GRB.Callback.MIPSOL:
            # selected edges
            xvals = model.cbGetSolution(model._x)
            selected = gp.tuplelist(
                (i, j) for i, j in model._x.keys() if xvals[i, j] > 1e-2)
            # shortest cycle
            tour = subtour(selected, model._ind)
            # add cuts
            if len(tour) < len(model._ind):
                model.cbLazy(gp.quicksum(model._x[i, j] for i, j in combinations(tour, 2)) <= len(tour) - 1)

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (list): cost vector
        """
        obj = gp.quicksum(c[i] * self.x[e] for i, e in enumerate(self.graph.edges) if e in self.x)
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model
        """
        self._model.update()
        self._model.optimize(self._subtourelim)
        sol = np.zeros(len(self.graph.edges), dtype=np.uint8)
        for i, e in enumerate(self.graph.edges):
            if e not in self.x:
                continue
            if self.x[e].x > 1e-2:
                sol[i] = 1
        return sol, self._model.objVal

    def getTour(self, sol):
        """
        A method to get a tour from solution

        Args:
            sol (list): solution

        Returns:
            list: a TSP tour
        """
        # active edges
        edges = defaultdict(list)
        for i, (j, k) in enumerate(self.graph.edges):
            if sol[i] > 1e-2:
                edges[j].append(k)
                edges[k].append(j)
        # get tour
        visited = {list(edges.keys())[0]}
        tour = [list(edges.keys())[0]]
        while len(visited) < len(edges):
            i = tour[-1]
            for j in edges[i]:
                if j not in visited:
                    tour.append(j)
                    visited.add(j)
                    break
        if 0 in edges[tour[-1]]:
            tour.append(0)
        return tour
