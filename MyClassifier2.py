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


class MyClassifier:
    def __init__(self, K, M):
        self.K = K  #Number of classes
        self.M = M  #Number of features
        self.W = []
        self.w = []

    def train(self, p, train_data, train_label):
        return

    def f(self, input):
        return

    def classify(self, test_data):
        return

    def TestCorrupted(self, p, test_data):
        return



if __name__ == '__main__':
    # the labels for 7 are changed to -1 in order to train in a hinge loss {1: 1, 7: -1}
    train_data, train_label, test_data, test_label = bn_data()
