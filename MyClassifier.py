# # -*- coding: utf-8 -*-
# """
# ECE 236A Project 1, MyClassifier.py template. Note that you should change the
# name of this file to MyClassifier_{groupno}.py
# """
import pandas as pd
import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt
import argparse
from keras.datasets import mnist


def softmax(x):
    shift_constant = np.expand_dims(-np.amax(x, axis=1), axis=1)
    num = np.exp(x + shift_constant)
    den = np.expand_dims(np.sum(num, axis=1), axis=1)
    return num / den


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_accuracy(output, labels):
    accuracy = (np.argmax(output, axis=1) == np.argmax(labels, axis=1)).sum() * 1. / labels.shape[0]
    return accuracy


def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels


def initialize(num_feature, num_class):
    W = np.random.standard_normal((num_feature, num_class))
    b = np.zeros((1, num_class))

    params = {}
    params['W'] = W
    params['b'] = b
    return params


def forward(data, labels, params):
    W = params['W']
    b = params['b']
    z = data @ W + b

    y = softmax(z)
    loss = -1 / data.shape[0] * sum(sum(np.log(y) * labels))
    return y, loss


def backward_prop(data, labels, params, forward_f, reg=0.001):
    y, loss = forward_f(data, labels, params)
    size = data.shape[0]

    dz = y - labels
    dW = data.T @ dz / size
    db = np.sum(dz, axis=0, keepdims=True) / size

    gradient = {}
    gradient['W'] = dW + 2 * reg * params['W']
    gradient['b'] = db
    return gradient


def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_f, backward_prop_f,
                           reg):
    n = train_data.shape[0]
    iteration = n // batch_size

    for i in range(iteration):
        data = train_data[i * batch_size: (i + 1) * batch_size]
        labels = train_labels[i * batch_size: (i + 1) * batch_size]
        gradient = backward_prop_f(data, labels, params, forward_f, reg)

        params['W'] -= learning_rate * gradient['W']
        params['b'] -= learning_rate * gradient['b']

    return


# set all the parameters
parser = argparse.ArgumentParser(description='Train a nn model.')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=150)
parser.add_argument('--reg', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.2)

# use inside training to pass the value to testing
parser.add_argument('--mean', type=float, default=0.0)
parser.add_argument('--std', type=float, default=0.0)
args = parser.parse_args()

class MyClassifier:
    def __init__(self, K, M):
        self.K = K  #Number of classes
        self.M = M  #Number of features
        self.W = []
        self.w = []

    def train(self, p, train_data, train_label):
        # pre-processing data (permutation, normalize)
        N_train = train_data.shape[0]
        train_labels = one_hot_labels(train_label)
        p = np.random.permutation(N_train)
        train_data = train_data[p, :]
        train_labels = train_labels[p, :]

        # divide the training set into training and evaluation set
        evl_size = 6000    # use 1 when train all the training data
        evl_data = train_data[0:evl_size, :]
        evl_labels = train_labels[0:evl_size, :]
        train_data = train_data[evl_size:, :]
        train_labels = train_labels[evl_size:, :]

        args.mean = np.mean(train_data)
        args.std = np.std(train_data)

        train_data = (train_data - args.mean) / args.std
        evl_data = (evl_data - args.mean) / args.std

        params = initialize(self.M, self.K)

        cost_train = []
        cost_evl = []
        accuracy_train = []
        accuracy_evl = []

        # training and update cost & accuracy
        for epoch in range(args.num_epochs):
            gradient_descent_epoch(train_data, train_labels, args.lr, args.batch_size, params, forward,
                                   backward_prop, args.reg)

            output, cost = forward(train_data, train_labels, params)
            cost_train.append(cost)
            accuracy_train.append(compute_accuracy(output, train_labels))

            output, cost = forward(evl_data, evl_labels, params)
            cost_evl.append(cost)
            accuracy_evl.append(compute_accuracy(output, evl_labels))

        self.W = params['W']
        self.w = params['b']

        # plots of loss and accuracy
        t = np.arange(args.num_epochs)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        name = 'regularized'

        ax1.plot(t, cost_train, 'r', label='train')
        ax1.plot(t, cost_evl, 'b', label='evl')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        ax1.legend()

        ax2.plot(t, accuracy_train, 'r', label='train')
        ax2.plot(t, accuracy_evl, 'b', label='evl')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.png')
        f = open('./' + name + '.txt', "w")
        f.write(str(params))
        f.close()

    def f(self, input):
        y = softmax(input)
        return np.argmax(y, axis=1)

    def classify(self, test_data):
        test_data = (test_data - args.mean) / args.std

        z = test_data @ self.W + self.w
        y = softmax(z)
        return self.f(y)

    def TestCorrupted(self, p, test_data):
        erase = np.random.choice([0, 1], test_data.shape, p=[p, 1-p])
        return self.classify(test_data * erase)


if __name__ == '__main__':
    # data pre-processing
    (X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()
    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)

    n_tarin = X_train.shape[0]
    n_test = X_test.shape[0]
    M = X_test.shape[1]

    mn_classifier = MyClassifier(args.num_classes, M)
    mn_classifier.train(1.0, X_train, y_train)

    y = mn_classifier.classify(X_test)
    yp = mn_classifier.TestCorrupted(1.0, X_test)
    accuracy_y = (y == y_test).sum() * 1. / n_test
    print(accuracy_y)
