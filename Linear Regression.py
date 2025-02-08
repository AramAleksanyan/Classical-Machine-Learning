import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

address = r"C:\Users\aram_\OneDrive\Desktop\AUA\CSV\possum.csv"
data_training = pd.read_csv(address)
Y = data_training.iloc[:, -1].values
Y = np.array(Y, dtype=float)
range_Y = np.max(Y) - np.min(Y)

X = data_training.iloc[:, :-1].values
X = np.array(X, dtype=float)

lambda_param = 10
identity_matrix = np.eye(X.shape[1])

XtX = np.dot(X.T, X)
XtY = np.dot(X.T, Y)
XtX_lambda = XtX + lambda_param * identity_matrix
XtX_lambda_inv = np.linalg.inv(XtX_lambda)
B = np.dot(XtX_lambda_inv, XtY)

Y_predicted = np.dot(X, B)

MSE = 0
for i in range(len(Y)):
    MSE += (Y_predicted[i] - Y[i]) ** 2

MSE = MSE / len(Y)
print(f'Processed points total: {len(Y)}')
print('Mean Squared Error:', round(MSE, 4))

ROOT_MSE = np.sqrt(MSE)
print(f'Root Mean Squared Error: {round(ROOT_MSE, 4)}')

MAE = 0
for i in range(len(Y)):
    MAE += abs(Y_predicted[i] - Y[i])

MAE = MAE / len(Y)
print(f'Mean Absolute Error: {round(MAE, 4)}')

ROOT_MSE_percentage = (ROOT_MSE / range_Y) * 100
print(f'RMSE Percentage of Range: {round(ROOT_MSE_percentage, 4)} %')

if X.shape[1] == 2:
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 1], Y, color='blue', label='Data Points')

    x_values = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    y_values = B[0] + B[1] * x_values
    plt.plot(x_values, y_values, color='red', label='Regression Line')

    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Data Points and Regression Line')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print(f"Data is {X.shape[1] + 1}D. Plotting is only supported for 2D data.")
    