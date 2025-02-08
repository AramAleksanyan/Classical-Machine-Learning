import numpy as np
import pandas as pd
from collections import Counter


def distance(x, X):
    distances = []
    for i in range(X.shape[0]):
        d = 0
        for j in range(len(x)):
            d += (x[j] - X.iloc[i, j]) ** 2
        distances.append(np.sqrt(d))
    return distances


def k_nearest_neighbors(x, X, Y, k):
    distances = distance(x, X)
    sorted_indices = np.argsort(distances)
    nearest_labels = Y.iloc[sorted_indices[:k]]
    majority_vote = Counter(nearest_labels).most_common(1)[0][0]
    return majority_vote


def evaluate_k(X, Y, k_values_):
    best_k_ = None
    best_accuracy_ = 0

    for k in k_values_:
        predictions = []
        for i in range(X.shape[0]):
            sample_point = X.iloc[i].values
            predicted_class = k_nearest_neighbors(sample_point, X, Y, k=k)
            predictions.append(predicted_class)

        right_predictions = 0
        for i in range(len(predictions)):
            if Y.iloc[i] == predictions[i]:
                right_predictions += 1

        accuracy = round(right_predictions / len(Y), 3) * 100
        print(f'Accuracy for K={k}: {accuracy}%')

        if accuracy > best_accuracy_:
            best_accuracy_ = accuracy
            best_k_ = k

    return best_k_, best_accuracy_


address = r"C:\Users\aram_\OneDrive\Desktop\AUA\CSV\diabetes2.csv"
data = pd.read_csv(address)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

k_values = range(1, int(np.sqrt(len(Y))), 2)  # step = 2

best_k, best_accuracy = evaluate_k(X, Y, k_values)
print(f'Best k value: {best_k} with an accuracy of {best_accuracy}%')
