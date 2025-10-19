import numpy as np
from src.data_loader import labels_train, data_train
from src.neural_network import init_parameters, forward_propagation, backward_propagation, update_parameters


#returns the highest probability class
def get_prediction(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, labels):
    print(predictions, labels)
    return np.sum(predictions == labels) / labels.size


def gradient_descent(data_train, labels_train):
    W1, b1, W2, b2 = init_parameters()

    epochs = 1000
    for i in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(data_train, W1, b1, W2, b2)
        db1, db2, dW1, dW2 = backward_propagation(data_train, labels_train, A1, A2, Z1, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2)

        print("Epoch:", i, "Accuracy:", get_accuracy(get_prediction(A2), labels_train))

    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(data_train, labels_train)