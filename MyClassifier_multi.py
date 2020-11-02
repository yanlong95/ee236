# -*- coding: utf-8 -*-
"""
ECE 236A Project 1, MyClassifier.py template. Note that you should change the
name of this file to MyClassifier_{groupno}.py
"""
import pandas as pd
import numpy as np
import cvxpy as cp

from keras.datasets import mnist


def select_data(data, label, idx1=1, idx2=7):
    selected = np.where((label == idx1) | (label == idx2))
    selected_data = data[selected]
    selected_label = label[selected]
    selected_label = np.where(selected_label == idx1, 1, selected_label)
    selected_label = np.where(selected_label == idx2, -1, selected_label)
    return selected_data, selected_label


class MyClassifier:
    def __init__(self, K, M):
        self.K = K  # Number of classes
        self.M = M  # Number of features
        self.W = []
        self.w = []

    def train(self, p, train_data, train_label):
        self.W = np.zeros((self.M, (self.K-1)*self.K//2))
        self.w = np.zeros((1, (self.K-1)*self.K//2))

        for i in range(self.K):
            for j in range(i + 1, self.K):
                x_train, y_train = select_data(train_data, train_label, i, j)
                N_train = y_train.shape[0]
                erase = np.random.choice([0, 1], x_train.shape, p=[p, 1 - p])
                x_train_erase = x_train * erase
                y_train = np.expand_dims(y_train, axis=1)

                W = cp.Variable((self.M, 1))
                w = cp.Variable()
                loss = cp.sum(cp.pos(1 - cp.multiply(y_train, x_train_erase @ W + w)))
                reg = cp.norm(W, 1)
                lambd = 0.001   # need to tune
                prob = cp.Problem(cp.Minimize(loss / N_train + lambd * reg))

                prob.solve()
                self.W[:, (2*self.K-i-1)*i//2+j-i-1] = W.value.flatten()
                self.w[:, (2*self.K-i-1)*i//2+j-i-1] = w.value.flatten()

                print(int(((2*self.K-i-1)*i//2+j-i) / ((self.K-1)*self.K//2) * 100), 'percent done.')

    def f(self, input):
        choice = np.sign(input @ self.W + self.w)
        choice = choice.astype(int)
        for i in range(self.K):
            for j in range(i+1, self.K):
                col = (2 * self.K - i - 1) * i // 2 + j - i - 1
                choice[:, col] = np.where(choice[:, col] == 1, i, choice[:, col])
                choice[:, col] = np.where(choice[:, col] == -1, j, choice[:, col])
        return choice

    def classify(self, test_data):
        choice = self.f(test_data)
        y_pred = np.zeros((test_data.shape[0]), dtype=int)
        for i in range(choice.shape[0]):
            y_pred[i] = np.bincount(choice[i, :]).argmax()

        return y_pred

    def TestCorrupted(self, p, test_data):
        erase = np.random.choice([0, 1], test_data.shape, p=[p, 1 - p])
        return self.classify(erase * test_data)


if __name__ == '__main__':
    # mnist_train = pd.read_csv("~\Desktop\proj1\data\mnist_train.csv")
    # mnist_test = pd.read_csv("~\Desktop\proj1\data\mnist_test.csv")

    # load data as numpy array, data = (n, 784), label = (n,)
    (X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()
    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)

    # train and test
    p = 0.6
    hyperplane = MyClassifier(10, 784)
    hyperplane.train(p, X_train, y_train)
    accuracy = np.sum(hyperplane.TestCorrupted(p, X_test) == y_test) / X_test.shape[0]
    print('Accuracy: ', accuracy)
