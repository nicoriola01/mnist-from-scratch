import numpy as np

#initializing weights with "He initialization" and biases with small random values
def init_parameters():
    input_1 = 784
    input_2 = 128
    output = 10

    bias_low = 0.001
    bias_high = 0.01

    W1 = np.random.randn(input_2, input_1) * np.sqrt(2 / input_1)
    b1 = np.random.rand(input_2, 1) * (bias_high - bias_low) + bias_low

    W2 = np.random.randn(output, input_2) * np.sqrt(2 / input_2)
    b2 = np.random.rand(output, 1) * (bias_high - bias_low) + bias_low

    return W1, b1, W2, b2

#standard forward propagation
def forward_propagation(data_train, W1, b1, W2, b2):
    Z1 = W1.dot(data_train) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softMax(Z2)

    return Z1, A1, Z2, A2

#standard backward propagation
def backward_propagation(data_train, labels_train, A1, A2, Z1, W2):
    one_hot_labels = one_hot(labels_train)
    dZ2 = A2 - one_hot_labels
    dW2 = 1 / data_train.shape[1] * dZ2.dot(A1.T)
    db2 = 1 / data_train.shape[1] * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * ReLu_derivative(Z1)
    dW1 = 1 / data_train.shape[1] * dZ1.dot(data_train.T)
    db1 = 1 / data_train.shape[1] * np.sum(dZ1, axis=1, keepdims=True)

    return db1, db2, dW1, dW2

#updating parameters with gradient descent after every epoch
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2):
    alpha = 0.01

    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

#turning labels into one hot vector
def one_hot(labels):
    one_hot_labels = np.zeros((labels.size, labels.max() + 1))
    one_hot_labels[np.arange(labels.size), labels] = 1

    return one_hot_labels.T

#activation function
def ReLu(Z):
    return np.maximum(Z, 0)

#derivative of activation function
def ReLu_derivative(Z):
    return Z > 0

#turning values into probabilities
def softMax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
