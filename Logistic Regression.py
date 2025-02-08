import pandas as pd
import numpy as np


address = r"C:\Users\aram_\OneDrive\Desktop\AUA\CSV\health care diabetes.csv"
data = pd.read_csv(address)

Y = data.iloc[:, -1].values
Y = np.where(Y > 0.5, 1, 0)

X = data.iloc[:, :-1].values
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term


def normalize(X):
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Prevent division by zero
    return (X - np.mean(X, axis=0)) / std

X = normalize(X)

def initialize_random_weights(n):
    return np.random.rand(n, 1) * 0.01

n_features = X.shape[1]
weights = initialize_random_weights(n_features)
bias = np.random.rand() * 0.1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(x, w, b):
    return sigmoid(np.dot(x, w) + b)

def compute_cost(x, y, w, b):
    m = x.shape[0]
    epsilon = 1e-15  # Avoid log(0)
    predictions = np.clip(h(x, w, b), epsilon, 1 - epsilon)
    cost = -(1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

def gradients(x, y, w, b):
    m = x.shape[0]
    predictions = h(x, w, b)
    error = predictions - y.reshape(-1, 1)
    dw = (1 / m) * np.dot(x.T, error)
    db = (1 / m) * np.sum(error)
    return dw, db

def gradient_descent(x, y, w, b, alpha, iters):
    for i in range(iters):
        dw, db = gradients(x, y, w, b)
        w -= alpha * dw
        b -= alpha * db
        if i % 50 == 0:  # Print cost every 50 iterations
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {i}, Cost: {cost}")
    return w, b

def predict(x, w, b, threshold=0.5):
    probabilities = h(x, w, b)
    return (probabilities >= threshold).astype(int)

alpha = 0.01
iterations = 2000

weights, bias = gradient_descent(X, Y, weights, bias, alpha, iterations)

print("Trained weights:", weights)
print("Trained bias:", bias)

predictions = predict(X, weights, bias)
accuracy = np.mean(predictions == Y.reshape(-1, 1)) * 100
print(f"Training Accuracy: {accuracy:.2f}%")
