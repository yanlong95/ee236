import numpy as np
from keras.datasets import mnist

# Note: the labels for 7 are changed to -1 in order to train in a hinge loss {1: 1, 7: -1}
def bn_data():
    (X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()
    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)

    # binary train data
    idx1 = np.where(y_train == 1)
    idx7 = np.where(y_train == 7)
    X_train1 = X_train[idx1[0], :]
    X_train7 = X_train[idx7[0], :]
    y_train1 = y_train[idx1[0]]
    y_train7 = y_train[idx7[0]] // -7

    permutation = np.random.permutation(y_train1.shape[0]+y_train7.shape[0])
    train_data = np.concatenate((X_train1, X_train7), axis=0)[permutation, :]
    train_label = np.concatenate((y_train1, y_train7))[permutation]

    # binary test data
    idx_t1 = np.where(y_test == 1)
    idx_t7 = np.where(y_test == 7)
    X_test1 = X_test[idx_t1[0], :]
    X_test7 = X_test[idx_t7[0], :]
    y_test1 = y_test[idx_t1[0]]
    y_test7 = y_test[idx_t7[0]] // -7

    permutation_t = np.random.permutation(y_test1.shape[0] + y_test7.shape[0])
    test_data = np.concatenate((X_test1, X_test7), axis=0)[permutation_t, :]
    test_label = np.concatenate((y_test1, y_test7))[permutation_t]

    return train_data, train_label, test_data, test_label

if __name__ == '__main__':
    train_data, train_label, test_data, test_label = bn_data()
    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)