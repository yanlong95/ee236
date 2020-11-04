# -*- coding: utf-8 -*-
"""
ECE 236A Project 1, MyClassifier.py template. Note that you should change the
name of this file to MyClassifier_{groupno}.py
"""
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from keras.datasets import mnist


def select_data(data, label, idx1, idx2):
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


        x_train, y_train = select_data(train_data, train_label, 1, 7)
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
        self.W[:, 0] = W.value.flatten()
        self.w[:, 0] = w.value.flatten()


    def f(self, input):
        choice = np.sign(input @ self.W + self.w)
        choice = choice.astype(int)
        col = 0
        choice[:, col] = np.where(choice[:, col] == 1, 1, choice[:, col])
        choice[:, col] = np.where(choice[:, col] == -1, 7, choice[:, col])    
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
    mnist_train = pd.read_csv("~\mnist_train.csv")
    mnist_test = pd.read_csv("~\mnist_test.csv")

    # load data as numpy array, data = (n, 784), label = (n,)
    (X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()
    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)

    # train and test

    hyperplane = MyClassifier(2, 784)
    X_test_filtered,y_test_filtered = select_data(X_test, y_test, 1, 7)

    t = np.arange(1,11)
    plt.axis([0, 10, 95, 100])
    plt.title('Classification Accuracy vs. Erasure Probability')
    plt.xlabel('Experiment #')
    plt.ylabel('% Accuracy')
    plt.xticks(t)
    plt.text(8, 96, r'${p = 0.4}$', color = 'red')
    plt.text(8, 95.6, r'${p = 0.6}$', color = 'green')
    plt.text(8, 95.2, r'${p = 0.8}$', color = 'blue')    
    accuracy_04 = []
    p = 0.4
    for i in range (0,10):
        hyperplane.train(p, X_train, y_train)
        accuracy = np.sum(hyperplane.TestCorrupted(p, X_test) == y_test) / X_test_filtered.shape[0]
        print('Accuracy: ', accuracy)
        accuracy_04.append(accuracy*100)
    plt.plot(t,accuracy_04, 'ro')
    
    accuracy_06 = []
    p = 0.6
    for i in range (0,10):
        hyperplane.train(p, X_train, y_train)
        accuracy = np.sum(hyperplane.TestCorrupted(p, X_test) == y_test) / X_test_filtered.shape[0]
        print('Accuracy: ', accuracy)
        accuracy_06.append(accuracy*100)
    plt.plot(t,accuracy_06, 'g^')
    
    accuracy_08 = []
    p = 0.8
    for i in range (0,10):
        hyperplane.train(p, X_train, y_train)
        accuracy = np.sum(hyperplane.TestCorrupted(p, X_test) == y_test) / X_test_filtered.shape[0]
        print('Accuracy: ', accuracy)
        accuracy_08.append(accuracy*100)
    plt.plot(t,accuracy_08, 'bs')
    
    plt.savefig('Classification Accuracy vs. Erasure Probability.png')
 # red dashes, blue squares and green triangles


