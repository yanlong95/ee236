"""
The code use to find the parameters (W, w) for binary classification (1, 7).
W = (784*1), w = (784*1)

example: https://www.cvxpy.org/examples/machine_learning/svm.html
"""

import pandas as pd
import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt
from binary_data import bn_data
import time


class MyClassifier:
    def __init__(self, K, M):
        self.K = K  #Number of classes
        self.M = M  #Number of features
        self.W = []
        self.w = []

    def train(self, p, train_data, train_label):
        N_train = train_data.shape[0]
        erase = np.random.choice([0, 1], train_data.shape, p=[p, 1 - p])
        train_data = train_data * erase
        train_label = np.expand_dims(train_label, axis=1)

        W = cp.Variable((self.M, 1))
        b = cp.Variable()
        loss = cp.sum(cp.pos(1 - cp.multiply(train_label, train_data @ W + b)))
        reg = cp.norm(W, 1)

        lambd = 0.001
        prob = cp.Problem(cp.Minimize(loss/N_train + lambd * reg))
        prob.solve()

        self.W = W.value
        self.w = b.value

    def f(self, input):
        return np.sign(input @ self.W + self.w)

    def classify(self, test_data):
        return self.f(test_data)

    def TestCorrupted(self, p, test_data):
        erase = np.random.choice([0, 1], test_data.shape, p=[p, 1 - p])
        return self.classify(erase * test_data)


if __name__ == '__main__':
    # the labels for 7 are changed to -1 in order to train in a hinge loss {1: 1, 7: -1}
    t1 = time.time()

    p = 0.4
    train_data, train_label, test_data, test_label = bn_data()
    bn = MyClassifier(2, 784)
    bn.train(p, train_data, train_label)

    accuracy = np.sum(bn.TestCorrupted(p, test_data).flatten() == test_label) / test_data.shape[0]
    print('Accuracy: ', accuracy)

    t2 = time.time()
    print('Time: ', t2 - t1)


