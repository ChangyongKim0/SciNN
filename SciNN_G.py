# %%
import json
import copy
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math
from collections import deque
from itertools import chain


matplotlib.use("Agg")


class SciNN_G:
    def __init__(self, length_list, eta_matrix, eta_BCM_matrix, input_index, output_index):
        self.V = range(len(length_list))
        self.n = length_list
        self.eta = eta_matrix
        self.eta_BCM = eta_BCM_matrix
        self.i = input_index
        self.o = output_index
        self.h = [[0]*n_v for n_v in self.n]
        self.h_state = [[0]*n_v for n_v in self.n]
        self.W = [[[[0]*n_a
                    for _ in range(n_b)] for n_a in self.n] for n_b in self.n]
        self.E = [[(self.eta[b][a] > 0 or self.eta_BCM[b][a] > 0)
                   for a in self.V] for b in self.V]
        self.h_prev = [[0]*n_v for n_v in self.n]
        self.W_prev = [[[[0]*n_a
                         for _ in range(n_b)] for n_a in self.n] for n_b in self.n]
        self.rho = 0.8
        self.d = [self._getGraphDistance(
            self.V, self.E, v, self.o) for v in self.V]
        self.layer_depth = self._getGraphMinDepth(self.V, self.d)
        self.initial_range = .5
        self.max_range = 1
        self.V_sorted = self._getGraphOrder(self.V, self.d)

    def _getGraphMinDepth(self, V,  d):
        min_depth = 0
        for v in V:
            min_depth = max(min_depth, d[v])
        return min_depth + 1

    def _getGraphDistance(self, V, E, start, end, error_value=-1):
        queue = deque([start])
        visited = {start: None}

        while queue:
            current_node = queue.popleft()
            if current_node == end:
                break
            for v in V:
                if E[v][current_node] and v not in visited:
                    visited[v] = current_node
                    queue.append(v)
                    # print(start, end, v, current_node, E[v][current_node], visited)

        path = []

        while end != None:
            path.append(end)
            end = visited[end]
            # print(end, path)

        return len(path)-1 if path else error_value

    # def _topologicalSortUtil(self, V, E, v, visited, stack):
    #     visited[v] = True
    #     for w in V:
    #         if not visited[w] and E[w][v]:
    #             self._topologicalSortUtil(V, E, w, visited, stack)
    #     stack.append(v)

    # def _getTopOrder(self, V, E):
    #     visited = [False]*len(V)
    #     stack = []
    #     for v in V:
    #         if not visited[v]:
    #             self._topologicalSortUtil(V, E, v, visited, stack)
    #     return stack[::-1]

    def _getGraphOrder(self, V, d):
        temp = [[] for _ in range(self.layer_depth)]
        for v in V:
            temp[d[v]].append(v)
            # print(temp)
        return list(chain(*temp))[::-1]

    def _deleteDiagonal(self):
        for v in self.V:
            for i in range(len(self.h[v])):
                self.W[v][v][i][i] = 0

    def _getInternalSynapseNo(self, sparsity, N):
        return sparsity*math.pow(N, 4/3)

    def _getInternalSynapseDensity(self, sparsity, N):
        return self._getInternalSynapseNo(sparsity, N)/N/N

    def _getDirectionalSynapseDensity(self, sparsity, N1, N2):
        return (self._getInternalSynapseNo(sparsity, N1+N2) - self._getInternalSynapseNo(sparsity, N1) - self._getInternalSynapseNo(sparsity, N2))/N1/N2/2

    def _getPreLayer(self, b, V, E):
        pre_layer_list_of_b = []
        for a in V:
            if E[b][a] and b != a:
                pre_layer_list_of_b.append(a)
        return pre_layer_list_of_b

    def initializeWeight(self, sparsity=2.27):
        for b in self.V:
            for a in self.V:
                if b == a:
                    synapse_density = self._getInternalSynapseDensity(
                        sparsity, len(self.h[b]))
                elif self.E[b][a]:
                    synapse_density = self._getDirectionalSynapseDensity(
                        sparsity, len(self.h[b]), len(self.h[a]))
                if self.E[b][a]:
                    for j in range(len(self.h[a])):
                        if self.h_state[a][j] == 0:
                            h_state_a_j = 1 if random.random() < self.rho else -1
                        else:
                            h_state_a_j = self.h_state[a][j]
                        for i in range(len(self.h[b])):
                            if random.random() < synapse_density:
                                self.h_state[a][j] = h_state_a_j
                                self.W[b][a][i][j] = random.random(
                                )*self.initial_range*h_state_a_j
        self._deleteDiagonal()

    def _makeDataFrame(self, V, h, W, h_prev):
        temp_data = [{'from': 't', 'to': 't', 'weight': None}]
        temp_data = [{'from': 't', 'to': 't-1', 'weight': None}]
        for v in V:
            for i in range(len(h[v])):
                temp_data.append(
                    {'from': f'{v}-{i}', 'to': 't-1', 'weight':  self.max_range/2, 'strength': h_prev[v][i]})
                temp_data.append(
                    {'from': f'{v}-{i}', 'to': 't', 'weight':  self.max_range/2, 'strength': h[v][i]})
                temp_data.append(
                    {'from': 't', 'to': f'{v}-{i}', 'weight':  self.max_range/2, 'strength': h[v][i]})
        for b in V:
            for a in V:
                for i in range(len(h[b])):
                    for j in range(len(h[a])):
                        temp_data.append({'from': f'{a}-{j}', 'to': f'{b}-{i}', 'weight': abs(
                            W[b][a][i][j]), 'strength': W[b][a][i][j]*h_prev[a][j]})
        return pd.DataFrame(temp_data)

    def drawHeatmap(self):
        # plt.close()
        data = self._makeDataFrame(self.V, self.h, self.W, self.h_prev)
        g = sns.relplot(x="from", y="to",
                        hue="strength", size="weight",
                        hue_norm=(-self.max_range, self.max_range), size_norm=(0, self.max_range),
                        palette="coolwarm",
                        sizes=(0, 3000/(sum([len(self.h[l])
                               for l in range(len(self.h))]))),
                        marker="s", linewidth=0, legend=False,
                        aspect=1.25, data=data)
        g.ax.invert_yaxis()
        ghost = g.ax.scatter([], [], c=[], vmin=-self.max_range,
                             vmax=self.max_range, cmap="coolwarm")
        g.figure.colorbar(ghost)
        return g

    def saveHeatmap(self, file_name):
        g = self.drawHeatmap()
        g.savefig(file_name)
        plt.close('all')
        plt.clf()

    def saveCurrent(self, file_name):
        to_save = {"h": self.h, "W": self.W, "h_prev": self.h_prev,
                   "W_prev": self.W_prev, "h_state": self.h_state}
        with open("data/"+file_name+".json", "w") as f:
            json.dump(to_save, f)

    def loadData(self, file_name):
        with open("data/"+file_name+".json", "r") as f:
            to_load = json.load(f)
        self.h, self.W, self.h_prev, self.W_prev = to_load[
            "h"], to_load["W"], to_load["h_prev"], to_load["W_prev"]
        self.h_state = to_load["h_state"]


# a = SciNN(3)
# a.addLayer(4)
# a.initializeWeight()

# %%
