# -*- coding: utf-8 -*-
"""
ECE 236A Project 1, MyClassifier.py template. Note that you should change the
name of this file to MyClassifier_{groupno}.py
"""
import pandas as pd
import numpy as np
import cvxpy as cp

class MyClassifier:
    def __init__(self,K,M):
        self.K = K  #Number of classes
        self.M = M  #Number of features
        self.W = []
        self.w = []
        
    def train(self, p, train_data, train_label):
        
        N_train = train_label.shape[0];
        erase = np.random.choice([0, 1], train_data.shape, p=[p, 1 - p])
        train_data_erase = train_data * erase
        train_label = np.expand_dims(train_label, axis=1)
        
        W = cp.Variable((self.M, 1))
        w = cp.Variable()
        loss = cp.sum(cp.pos(1-cp.multiply(train_label, train_data_erase @ W + w)))
        reg = cp.norm(W, 1)
        lambd = 0.001
        prob = cp.Problem(cp.Minimize(loss/N_train + lambd*reg))
        
        prob.solve()
        self.W = W.value
        self.w = w.value
        
        
    def f(self,input):
        return np.sign(input @ self.W + self.w)
        
    def classify(self,test_data):
        return self.f(test_data)
    
    def TestCorrupted(self,p,test_data):
        erase = np.random.choice([0, 1], test_data.shape, p=[p, 1 - p])
        return self.classify(erase * test_data)


mnist_train = pd.read_csv("D:\Python Project\Linear Programming Project 1\mnist_train.csv")
mnist_test = pd.read_csv("D:\Python Project\Linear Programming Project 1\mnist_test.csv")
mnist_train_selected = mnist_train[mnist_train["label"].isin([1, 7])]
mnist_test_selected = mnist_test[mnist_test["label"].isin([1, 7])]
y_train = np.array(mnist_train_selected['label'], dtype=float)
y_train = np.where(y_train==7,-1,y_train)
x_train = np.array(mnist_train_selected, dtype=float)
x_train = np.delete(x_train,0,1)
y_test = np.array(mnist_test_selected['label'],dtype=float)
y_test = np.where(y_test==7,-1,y_test)
x_test = np.array(mnist_test_selected, dtype=float)
x_test = np.delete(x_test,0,1)

p = 0.6
hyperplane = MyClassifier(2, 784)
hyperplane.train(p, x_train, y_train)
accuracy = np.sum(hyperplane.TestCorrupted(p, x_test).flatten() == y_test) / y_test.shape
print('Accuracy: ', accuracy)
