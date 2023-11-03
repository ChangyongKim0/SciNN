# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import copy


class SciNN:
    def __init__(self, input_length):
        self.h = [[0]*input_length]
        self.W = []
        self.h_prev = [[0]*input_length]
        self.W_prev = []
        self.initial_range = 1
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

    def initializeWeight(self):
        for W_l in self.W:
            for W_l_i in W_l:
                for j in range(len(W_l_i)):
                    W_l_i[j] = random.randrange(
                        0, 2)*(random.random()*2*self.initial_range-self.initial_range)
        for W_Lplus1_i in self.W[self.layer_depth]:
            for j in range(len(W_Lplus1_i)-len(self.W[self.layer_depth]), len(W_Lplus1_i)):
                W_Lplus1_i[j] = 0

    def setActivationFunction(self, fn_type):
        if fn_type == "tanh":
            self.actFn = lambda x: np.tanh(x)
            self.actFnDer = lambda y: 1 - y**2
        elif fn_type == "sigmoid":
            self.actFn = lambda x: 1/(1+np.exp(x))
            self.actFnDer = lambda y: y*(1 - y)
        elif fn_type == "relu":
            self.actFn = lambda x: x if x > 0 else 0
            self.actFnDer = lambda y: 1 if y > 0 else 0
        else:
            _

    def setLossFunction(self, fn_type):
        def _softmax(output):
            output_sum = sum([np.exp(output[i]) for i in range(len(output))])
            return [np.exp(output[i])/output_sum for i in range(len(output))]
        if fn_type == "cross entropy":
            self.lossFn = lambda output, target: -1 * \
                sum([target[i]*np.log(_softmax(output)[i])
                    for i in range(len(output))])
            self.lossFnDer = lambda output, target: [-1*target[i]*(
                1-_softmax(output)[i]) for i in range(len(output))]
        elif fn_type == "mse":
            self.lossFn = lambda output, target: sum(
                [(output[i]-target[i])**2 for i in range(len(output))])/len(output)
            self.lossFnDer = lambda output, target: [
                2*(output[i] - target[i])/len(output) for i in range(len(output))]
        else:
            _

    def _propagateForward(self, input_vector):
        if len(input_vector) != len(self.h[0]):
            print("ERROR: wrong input length")
            return False
        else:
            self.h_prev = copy.deepcopy(self.h)
            for l in range(len(self.h)):
                if l == 0:
                    self.h[l] = copy.deepcopy(input_vector)
                else:
                    for i in range(len(self.h[l])):
                        self.h[l][i] = self.actFn(
                            np.matmul(self.W[l-1][i], np.concatenate([self.h[l-1], self.h_prev[l]])))
                        # remove negative voltage
                        if self.h[l][i] < 0:
                            self.h[l][i] = 0

    def _propagateBackward(self, target_vector):
        if len(target_vector) != len(self.h[-1]):
            print("ERROR: wrong input length")
            return False
        else:
            dL_dh = []
            self.W_prev = copy.deepcopy(self.W)
            for l in [len(self.W) - l - 1 for l in range(len(self.W))]:
                if l == len(self.W) - 1:
                    dL_dh = self.lossFnDer(self.h[l+1], target_vector)
                else:
                    dL_dh = [sum([self.actFnDer(self.h[l+1][j])*self.W[l][j][i]*dL_dh[j]
                                 for j in range(len(dL_dh))]) for i in range(len(self.h[l+1]))]
                for i in range(len(self.W[l])):
                    # print(i, len(self.W[l]), len(self.W[l][i]), len(self.h[l+1]), len(self.h[l]), len(dL_dh))
                    self.W[l][i] = [self.W[l][i][j] - self.lr*self.actFnDer(self.h[l+1][i])*np.concatenate(
                        [self.h[l], self.h_prev[l+1]])[j]*dL_dh[i] for j in range(len(self.W[l][i]))]
        for W_Lplus1_i in self.W[self.layer_depth]:
            for j in range(len(W_Lplus1_i)-len(self.W[self.layer_depth]), len(W_Lplus1_i)):
                W_Lplus1_i[j] = 0

    def _makeDataFrame(self, h, W, h_prev, W_prev):
        temp_data = [{'from': '', 'to': '', 'weight': None}]
        for l in range(len(h)):
            for i in range(len(h[l])):
                temp_data.append({'from': f'{
                                 l}-{i}', 'to': '', 'weight': self.initial_range/2, 'strength': h_prev[l][i]})
                temp_data.append({'from': '', 'to': f'{
                                 l}-{i}', 'weight': self.initial_range/2, 'strength': h[l][i]})
        for l in range(len(W)):
            for i in range(len(W[l])):
                for j in range(len(W[l][i])):
                    if j >= len(W[l][i])-len(W[l]):
                        temp_data.append({'from': f'{l+1}-{j-(len(W[l][i])-len(W[l]))}', 'to': f'{l+1}-{
                                         i}', 'weight': abs(W[l][i][j]), 'strength': W[l][i][j]*h_prev[l+1][i]})
                    else:
                        temp_data.append({'from': f'{l}-{j}', 'to': f'{l+1}-{i}', 'weight': abs(
                            W[l][i][j]), 'strength': W[l][i][j]*h_prev[l+1][i]})
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
        plt.close()


def printAfterErase(*args):
    print("\r", end="")
    print(*args, end="")


def fillZero(num, length):
    return str(num).rjust(length, "0")

# %%
