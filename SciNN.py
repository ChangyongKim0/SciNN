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

matplotlib.use("Agg")


class SciNN:
    def __init__(self, input_length):
        self.h = [[0]*input_length]
        self.W = []
        self.h_prev = [[0]*input_length]
        self.W_prev = []
        self.initial_range = .5
        self.lr = 0.05
        self.layer_depth = -1

    def addLayer(self, layer_length):
        self.W.append([[0 for _ in range(len(self.h[-1])+layer_length)]
                      for _ in range(layer_length)])
        self.h.append(
            [random.random()*self.initial_range for _ in range(layer_length)])
        self.W_prev.append(
            [[0 for _ in range(len(self.h[-1])+layer_length)] for _ in range(layer_length)])
        self.h_prev.append(
            [random.random()*self.initial_range for _ in range(layer_length)])
        self.layer_depth += 1

    def _deleteDiagonal(self):
        for W_l in self.W:
            for i in range(len(W_l)):
                for j in range(len(W_l[i])-len(W_l), len(W_l[i])):
                    W_l[j-(len(W_l[i])-len(W_l))][j] = 0

    def initializeWeightByDensity(self, density=0.5):
        for W_l in self.W:
            for W_l_i in W_l:
                for j in range(len(W_l_i)):
                    if random.random() < density:
                        W_l_i[j] = random.random()*2*self.initial_range - \
                            self.initial_range
                    else:
                        W_l_i[j] = 0
        for W_Lplus1_i in self.W[self.layer_depth]:
            for j in range(len(W_Lplus1_i)-len(self.W[self.layer_depth]), len(W_Lplus1_i)):
                W_Lplus1_i[j] = 0
        self._deleteDiagonal()

    def _getInternalSynapseNo(self, sparsity, N):
        return sparsity*math.pow(N, 4/3)

    def _getInternalSynapseDensity(self, sparsity, N):
        return self._getInternalSynapseNo(sparsity, N)/N/N

    def _getDirectionalSynapseDensity(self, sparsity, N1, N2):
        return (self._getInternalSynapseNo(sparsity, N1+N2) - self._getInternalSynapseNo(sparsity, N1) - self._getInternalSynapseNo(sparsity, N2))/N1/N2

    def initializeWeight(self, sparsity=2.27):
        for W_l in self.W:
            for W_l_i in W_l:
                for j in range(len(W_l_i)):
                    if j < len(W_l_i) - len(W_l):
                        synapse_density = self._getDirectionalSynapseDensity(
                            sparsity, len(W_l_i) - len(W_l), len(W_l))
                    else:
                        synapse_density = self._getInternalSynapseDensity(
                            sparsity, len(W_l_i) - len(W_l))
                    if random.random() < synapse_density:
                        W_l_i[j] = random.random()*2*self.initial_range - \
                            self.initial_range
                    else:
                        W_l_i[j] = 0
        for W_Lplus1_i in self.W[self.layer_depth]:
            for j in range(len(W_Lplus1_i)-len(self.W[self.layer_depth]), len(W_Lplus1_i)):
                W_Lplus1_i[j] = 0
        self._deleteDiagonal()

    def _makeDataFrame(self, h, W, h_prev, W_prev):
        temp_data = [{'from': 't', 'to': 't', 'weight': None}]
        temp_data = [{'from': 't', 'to': 't-1', 'weight': None}]
        for l in range(len(h)):
            for i in range(len(h[l])):
                temp_data.append(
                    {'from': f'{l}-{i}', 'to': 't-1', 'weight': self.initial_range/2, 'strength': h_prev[l][i]})
                temp_data.append(
                    {'from': f'{l}-{i}', 'to': 't', 'weight': self.initial_range/2, 'strength': h[l][i]})
                temp_data.append(
                    {'from': 't', 'to': f'{l}-{i}', 'weight': self.initial_range/2, 'strength': h[l][i]})
        for l in range(len(W)):
            for i in range(len(W[l])):
                for j in range(len(W[l][i])):
                    if j >= len(W[l][i])-len(W[l]):
                        temp_data.append({'from': f'{l+1}-{j-(len(W[l][i])-len(W[l]))}', 'to': f'{l+1}-{i}', 'weight': abs(
                            W[l][i][j]), 'strength': W[l][i][j]*h_prev[l+1][j-(len(W[l][i])-len(W[l]))]})
                    else:
                        temp_data.append({'from': f'{l}-{j}', 'to': f'{l+1}-{i}', 'weight': abs(
                            W[l][i][j]), 'strength': W[l][i][j]*h[l][j]})
        return pd.DataFrame(temp_data)

    def drawHeatmap(self):
        # plt.close()
        data = self._makeDataFrame(self.h, self.W, self.h_prev, self.W_prev)
        g = sns.relplot(x="from", y="to",
                        hue="strength", size="weight",
                        hue_norm=(-self.initial_range, self.initial_range), size_norm=(0, self.initial_range),
                        palette="coolwarm",
                        sizes=(0, 3000/(sum([len(self.h[l])
                               for l in range(len(self.h))]))),
                        marker="s", linewidth=0, legend=False,
                        aspect=1.25, data=data)
        g.ax.invert_yaxis()
        ghost = g.ax.scatter([], [], c=[], vmin=-self.initial_range,
                             vmax=self.initial_range, cmap="coolwarm")
        g.fig.colorbar(ghost)
        return g

    def saveHeatmap(self, file_name):
        g = self.drawHeatmap()
        g.savefig(file_name)
        plt.close('all')
        plt.clf()

    def saveCurrent(self, file_name):
        to_save = {"h": self.h, "W": self.W, "h_prev": self.h_prev,
                   "W_prev": self.W_prev, "layer_depth": self.layer_depth}
        with open("data/"+file_name+".json", "w") as f:
            json.dump(to_save, f)

    def loadData(self, file_name):
        with open("data/"+file_name+".json", "r") as f:
            to_load = json.load(f)
        self.h, self.W, self.h_prev, self.W_prev = to_load[
            "h"], to_load["W"], to_load["h_prev"], to_load["W_prev"]
        self.layer_depth = to_load["layer_depth"]


# a = SciNN(3)
# a.addLayer(4)
# a.initializeWeight()

# %%
