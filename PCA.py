import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

address = r"C:\Users\aram_\OneDrive\Desktop\AUA\CSV\glass.csv"
address_2 = r"C:\Users\aram_\OneDrive\Desktop\AUA\CSV\Clustering_gmm.csv"
data = pd.read_csv(address_2)
print(data)

n_components = 1
X = data.iloc[:, :]
X_centered = X - np.mean(X, axis=0)

covariance_matrix = np.cov(X_centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

indexes = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[indexes]
eigenvectors = eigenvectors[:, indexes]

components = eigenvectors[:, :n_components].T
X_transformed = np.dot(X_centered, components.T)

print("\nOriginal Data Shape:", X.shape)
print("Transformed Data Shape:", X_transformed.shape)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c='blue', edgecolor='k', s=50)
plt.title('Original Data (First Two Features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X_transformed, np.zeros_like(X_transformed), c='red', edgecolor='k', s=50)
plt.title('Transformed Data (First Principal Component)')
plt.xlabel('Principal Component 1')
plt.yticks([])  # Hide y-axis

plt.tight_layout()
plt.show()
