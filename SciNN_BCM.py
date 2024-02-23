import SciNN as s
import copy
import numpy as np
import math
import random

GLOBAL_ETA = 0.001


class SciNN_BCM(s.SciNN):
    def __init__(self, input_length):
        self.delta = .01
        self.epsilon = .01
        self.rho = 0.5
        super().__init__(input_length)
        self.v = [[0]*input_length]
        self.theta = [[0]*input_length]
        self.hat_w = [[0]*input_length]
        self.eta = [[GLOBAL_ETA]*input_length]
        self.actFn = math.tanh

    def addLayer(self, layer_length):
        self.v.append([0]*layer_length)
        self.theta.append([0]*layer_length)
        self.hat_w.append([0]*layer_length)
        self.eta.append([GLOBAL_ETA]*layer_length)
        super().addLayer(layer_length)

    def decodeWeightOfBackprop(self):
        for l in range(len(self.W)):
            for i in range(len(self.W[l])):
                # tilde_hat_w = sum(self.W[l][i])
                # print(tilde_hat_w)
                for j in range(len(self.W[l][i])):
                    self.W[l][i][j] = np.arctanh(self.W[l][i][j])
        self.W_prev = self.W

    def propagate(self, input_vector):
        if len(input_vector) != len(self.h[0]):
            print("ERROR: wrong input length")
            return False
        else:
            self.h_prev = copy.deepcopy(self.h)
            for l in range(len(self.h)):

                if l == 0:
                    self.h[l] = copy.deepcopy(input_vector)
                else:
                    # self.hat_w[l-1] = np.matmul(self.W[l-1],
                    #                             [1]*len(self.W[l-1][0]))
                    h = np.concatenate([self.h[l-1], self.h_prev[l]])
                    W_mod = [[math.tanh(self.W[l-1][i][j]) for j in range(len(self.W[l-1][i]))]
                             for i in range(len(self.W[l-1]))]
                    self.v[l] = np.matmul(W_mod, h)
                    # self.v[l] = [max(0, self.v[l][i])
                    #              for i in range(len(self.h[l]))]
                    for i in range(len(self.h[l])):
                        # self.h[l][i] = max(0, self.actFn(
                        #     self.v[l][i]*self.actFn(self.hat_w[l-1][i])/self.hat_w[l-1][i]))
                        self.h[l][i] = max(0, self.actFn(self.v[l][i]))
                        self.theta[l][i] *= 1-self.delta
                        self.theta[l][i] += self.delta * \
                            self.v[l][i]*self.v[l][i]*self.v[l][i]
                        for j in range(len(self.W[l-1][i])):
                            phi = self.v[l][i] * \
                                (self.v[l][i] - self.theta[l][i])
                            if self.W[l-1][i][j] != 0:
                                temp = self.W[l-1][i][j]
                                self.W[l-1][i][j] = (
                                    1 if temp > 0 else -1)*max(0, abs(temp)+self.eta[l][i]*phi*h[j])
                            else:
                                r = random.random()
                                if r < self.epsilon*self.rho:
                                    self.W[l-1][i][j] = max(
                                        0, self.eta[l][i]*phi * h[j])
                                elif r < self.epsilon:
                                    self.W[l-1][i][j] = -1*max(0, self.eta[l][i] *
                                                               phi*h[j])
        self._deleteDiagonal()
