# %%
import SciNN as s
import copy
import numpy as np


class SciNN_backprop(s.SciNN):
    def __init__(self, input_length):
        self.initial_range = 1
        self.lr = 0.05
        super().__init__(input_length)

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
            0

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

    def propagateForward(self, input_vector):
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

    def propagateBackward(self, target_vector):
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
                    for j in range(len(self.W[l][i])):
                        if abs(self.W[l][i][j]) >= 1:
                            self.W[l][i][j] = (
                                1 if self.W[l][i][j] else -1)*0.99
        for W_Lplus1_i in self.W[self.layer_depth]:
            for j in range(len(W_Lplus1_i)-len(self.W[self.layer_depth]), len(W_Lplus1_i)):
                W_Lplus1_i[j] = 0
        self._deleteDiagonal()

# %%
