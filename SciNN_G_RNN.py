import SciNN_G as sg
import copy
import numpy as np
import math
import random
from collections import deque


class SciNN_G_RNN(sg.SciNN_G):
    def __init__(self, length_list, eta_matrix, eta_BCM_matrix, input_index, output_index):
        super().__init__(length_list, eta_matrix, eta_BCM_matrix, input_index, output_index)
        self.delta = .01
        self.epsilon = .01
        self.lr = 0.05
        self.theta = [[0]*n_v for n_v in self.n]
        self.actFn = lambda x: math.tanh(x) if x > 0 else -0.05*x
        self.actFnDer = lambda y: 1 - y**2 if y > 0 else -0.05
        self.feedSelf = lambda output: output
        # objects below will be used in BP only
        # (i)-th index means the value at time (t-i)
        self.hs = [copy.deepcopy(self.h) for _ in range(self.layer_depth+1)]
        self.Ws = [copy.deepcopy(self.W) for _ in range(self.layer_depth+1)]
        self.loss = -1
        self.target_vector = [0]*self.n[self.o]
        self.max_tanh_value = 0.99

    def _deleteDiagonal(self):
        super()._deleteDiagonal()
        for d in range(self.layer_depth):
            for v in self.V:
                for i in range(len(self.h[v])):
                    self.Ws[d][v][v][i][i] = 0

    def initializeWeight(self, sparsity=2.27):
        super().initializeWeight(sparsity)
        self.hs = [copy.deepcopy(self.h) for _ in range(self.layer_depth+1)]
        self.Ws = [copy.deepcopy(self.W) for _ in range(self.layer_depth+1)]

    def decodeWeightOfBackprop(self):
        self.W = [[[[np.arctanh(self.W[b][a][i][j]) for j in range(
            self.n[a])] for i in range(self.n[b])] for a in self.V] for b in self.V]
        self.W_prev = self.W
        self.hs = [copy.deepcopy(self.h) for _ in range(self.layer_depth+1)]
        self.Ws = [copy.deepcopy(self.W) for _ in range(self.layer_depth+1)]

    def setFeedSelf(self, feedSelf):
        self.feedSelf = feedSelf

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
            0

    def propagateBP(self, input_vector, target_vector):
        self.propagateBPConsideringOutput(input_vector, lambda: target_vector)

    def propagateBPConsideringOutput(self, input_vector, getTargetVectorFromOutput):
        if len(input_vector) != len(self.h[self.i])-len(self.feedSelf(self.h[self.o])):
            print("ERROR: wrong input length")
            return False
        else:
            # at t, forward
            for b in self.V_sorted:
                self.h_prev[b] = copy.deepcopy(self.h[b])
                if b != self.i:
                    pre_of_b = self._getPreLayer(b, self.V, self.E)
                    self.hs[self.d[b]][b] = [self.actFn(sum([np.matmul(
                        self.Ws[self.d[b]+1][b][a], self.hs[self.d[b]+1][a])[i] for a in pre_of_b])) for i in range(self.n[b])]
                self.h[b] = copy.deepcopy(self.hs[self.d[b]][b])
            # at t, backward
            dL_dh = [[0]*n_v for n_v in self.n]
            self.target_vector = getTargetVectorFromOutput(self.h[self.o])
            self.loss = self.lossFn(self.hs[0][self.o], self.target_vector)
            for a in self.V_sorted[::-1]:
                if a == self.o:
                    dL_dh[a] = self.lossFnDer(
                        self.hs[0][self.o], self.target_vector)
                else:
                    for b in self.V:
                        if self.E[b][a] and self.d[b]+1 == self.d[a]:
                            for j in range(self.n[a]):
                                dL_dh[a][j] += sum([self.Ws[self.d[a]][b][a][i][j]*self.actFnDer(
                                    self.hs[self.d[b]][b][i])*dL_dh[b][i] for i in range(self.n[b])])
            for a in self.V:
                for b in self.V:
                    if self.E[b][a]:
                        self.W_prev[b][a] = copy.deepcopy(self.W[b][a])
                        for i in range(self.n[b]):
                            for j in range(self.n[a]):
                                self.Ws[self.d[b]
                                        ][b][a][i][j] = self.Ws[self.d[b]+1][b][a][i][j]-self.eta[b][a] * \
                                    self.hs[self.d[b]+1][a][j] * \
                                    self.actFnDer(
                                        self.hs[self.d[b]][b][i])*dL_dh[b][i]
                                if self.Ws[self.d[b]][b][a][i][j] != 0 and self.h_state[a][j] == 0:
                                    self.h_state[a][j] = 1 if self.Ws[self.d[b]
                                                                      ][b][a][i][j] > 0 else -1
                                if self.Ws[self.d[b]][b][a][i][j] * self.h_state[a][j] < 0:
                                    self.Ws[self.d[b]][b][a][i][j] = 0
                                if self.Ws[self.d[b]][b][a][i][j] * self.h_state[a][j] >= self.max_tanh_value:
                                    self.Ws[self.d[b]][b][a][i][j] = self.max_tanh_value * \
                                        self.h_state[a][j]

                        self.W[b][a] = copy.deepcopy(self.Ws[self.d[b]][b][a])
            self._deleteDiagonal()
            # at t+1, backward
            self.Ws.insert(0, [[[[0]*n_a
                                 for _ in range(n_b)] for n_a in self.n] for n_b in self.n])
            self.Ws.pop(-1)
            # at t+1, forward
            self.hs.insert(0, [[0]*n_v for n_v in self.n])
            self.hs[0][self.i] = [
                *input_vector, *self.feedSelf(self.hs[1][self.o])]
            self.hs.pop(-1)

    def propagateBCM(self, input_vector):
        if len(input_vector) != len(self.h[self.i])-len(self.feedSelf(self.h[self.o])):
            print("ERROR: wrong input length")
            return False
        else:
            self.h_prev = copy.deepcopy(self.h)
            self.W_prev = copy.deepcopy(self.W)
            for b in self.V:
                if b == self.i:
                    self.h[b] = [*input_vector, *self.h_prev[self.o]]
                else:
                    pre_of_b = self._getPreLayer(b, self.V, self.E)
                    v_b = sum([np.matmul(self.W_prev[b][a], self.h_prev[a])
                              for a in pre_of_b])
                    # tilde_w_b = sum(
                    #     [np.matmul(self.W_prev[b][a], [1]*len(self.h_prev[a])) for a in pre_of_b])
                    for i in range(len(self.h[b])):
                        self.h[b][i] = max(0, math.tanh(sum([sum([self.h_prev[a][j]*math.tanh(
                            self.W_prev[b][a][i][j]) for j in range(self.n[a])]) for a in pre_of_b])))
                        self.theta[b][i] = (
                            1-self.delta)*self.theta[b][i]+self.delta*math.pow(v_b[i], 3)
                        for a in pre_of_b:
                            phi = v_b[i] * (v_b[i] - self.theta[b][i])
                            for j in range(len(self.h[a])):
                                if self.W_prev[b][a][i][j] != 0:
                                    self.W[b][a][i][j] = (
                                        1 if self.W_prev[b][a][i][j] > 0 else -1)*max(0, abs(self.W_prev[b][a][i][j])+self.eta_BCM[b][a]*phi*self.h[a][j])
                                else:
                                    if self.h_state[a][j] == 0:
                                        r = random.random()
                                        if r < self.epsilon*self.rho:
                                            self.h_state[a][j] = 1
                                        elif r < self.epsilon:
                                            self.h_state[a][j] = -1
                                    self.W[b][a][i][j] = self.h_state[a][j]*max(
                                        0, self.eta_BCM[b][a]*phi*self.h[a][j])
        self._deleteDiagonal()
