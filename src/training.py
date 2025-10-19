import numpy as np
from src.data_loader import labels_train, data_train, data_test, labels_test, training_set
from src.neural_network import init_parameters, train, forward_propagation, calculate_loss, calculate_accuracy

batch_size = 16

#returns the highest probability class
def get_prediction(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size


def gradient_descent(data_train, labels_train):
    W1, b1, W2, b2 = init_parameters()

    epochs = 20
    for epoch in range(epochs):
        shuffle = np.random.permutation(data_train.shape[1])
        data_train_shuffle = data_train[:, shuffle]
        labels_train_shuffle = labels_train[shuffle]

        for i in range(0, data_train.shape[1], batch_size):
            x = data_train_shuffle[:, i: i + batch_size]
            y = labels_train_shuffle[i: i + batch_size]
            W1, b1, W2, b2 = train(W1, b1, W2, b2, x, y)

        _,_,_,current_prediction_train = forward_propagation(data_train, W1, b1, W2, b2)
        train_accuracy = calculate_accuracy(current_prediction_train, labels_train)

        _, _, _, current_prediction_test = forward_propagation(data_test, W1, b1, W2, b2)
        test_accuracy = calculate_accuracy(current_prediction_test, labels_test)

        training_loss = calculate_loss(current_prediction_train, labels_train)
        test_loss = calculate_loss(current_prediction_test, labels_test)

        print(f"Epoch: {epoch + 1}/{epochs} | Train accuracy: {train_accuracy:.4f} | "
              f"Test accuracy: {test_accuracy:.4f} | "
              f"Training loss: {training_loss:.4f} | Test loss: {test_loss:.4f}")

    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(data_train, labels_train)
print("Training completed successfully")

np.savez("../results/weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("Weights saved successfully")