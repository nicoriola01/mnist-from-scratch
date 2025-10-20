import numpy as np
from src.data_loader import data_test, labels_test
from src.neural_network import forward_propagation


def compute_confusion_matrix(labels, predictions):
     confusion_matrix = (np.zeros((10, 10))).astype(int)
     for true_pred, pred in zip(labels, predictions):
         confusion_matrix[true_pred, pred] += 1
     return confusion_matrix

def print_confusion_matrix(confusion_matrix):
    print("\nConfusion Matrix:\n")
    header = " " * 5 + " ".join([f"{i:<4}" for i in range(10)])
    print(header)
    print(" " * 4 + "-" * (len(header) - 4))
    for i, row in enumerate(confusion_matrix):
        row_str = f"{i:<2} | " + " ".join([f"{val:<4}" for val in row])
        print(row_str)

def print_report(confusion_matrix):
    report = ""
    report += f"{'':<10}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}\n"
    report += "-" * 52 + "\n"

    total_support = 0
    avg_precision = 0
    avg_recall = 0
    avg_f1 = 0

    for i in range(10):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        report += f"{i:<10}{precision:>10.2f}{recall:>10.2f}{f1_score:>10.2f}{support:>10}\n"

        total_support += support
        avg_precision += precision * support
        avg_recall += recall * support
        avg_f1 += f1_score * support

    report += "-" * 52 + "\n"
    avg_precision /= total_support
    avg_recall /= total_support
    avg_f1 /= total_support

    report += f"{'avg/total':<10}{avg_precision:>10.2f}{avg_recall:>10.2f}{avg_f1:>10.2f}{total_support:>10}\n"

    print("Classification Report:\n")
    print(report)


data = np.load("../results/weights.npz")
W1, b1, W2, b2 = data["W1"], data["b1"], data["W2"], data["b2"]

_,_,_,A2 = forward_propagation(data_test, W1, b1, W2, b2)
pred = np.argmax(A2, 0)
matrix = compute_confusion_matrix(labels_test, pred)
print_confusion_matrix(matrix)
print("\n")
print_report(matrix)