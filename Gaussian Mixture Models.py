import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


def initialize_parameters_kmeans(X, k):
    n, d = X.shape
    kmeans = KMeans(n_clusters=k, init='k-means++').fit(X)
    means_ = kmeans.cluster_centers_
    covariances_ = np.array([np.eye(d)] * k)
    weights_ = np.ones(k) / k
    return means_, covariances_, weights_


def e_step(X, means, covariances, weights):
    n, d = X.shape
    k = len(weights)
    responsibilities_ = np.zeros((n, k))
    for i in range(k):
        rv = multivariate_normal(mean=means[i], cov=covariances[i])
        responsibilities_[:, i] = weights[i] * rv.pdf(X)
    responsibilities_ /= responsibilities_.sum(axis=1, keepdims=True)
    return responsibilities_


def m_step(X, responsibilities):
    n, d = X.shape
    k = responsibilities.shape[1]
    new_means = np.zeros((k, d))
    new_covariances = np.zeros((k, d, d))
    new_weights = np.zeros(k)
    for i in range(k):
        N_i = np.sum(responsibilities[:, i])
        new_means[i] = np.sum(responsibilities[:, i].reshape(-1, 1) * X, axis=0) / N_i
        diff = X - new_means[i]
        new_covariances[i] = np.dot((responsibilities[:, i].reshape(-1, 1) * diff).T, diff) / N_i
        new_covariances[i] += 1e-6 * np.eye(d)  # Regularization to avoid singular matrix
        new_weights[i] = N_i / n
    return new_means, new_covariances, new_weights


def gmm_em(X, k, max_iter=100, tol=1e-6):
    global responsibilities
    means, covariances, weights = initialize_parameters_kmeans(X, k)
    log_likelihoods = []
    for iteration in range(max_iter):
        responsibilities = e_step(X, means, covariances, weights)
        means, covariances, weights = m_step(X, responsibilities)
        log_likelihood = np.sum(np.log(np.sum([
            weights[i] * multivariate_normal(mean=means[i], cov=covariances[i]).pdf(X)
            for i in range(k)], axis=0)))
        log_likelihoods.append(log_likelihood)
        if iteration > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break
    return means, covariances, weights, responsibilities, log_likelihoods


address = r"C:\Users\aram_\OneDrive\Desktop\AUA\CSV\Clustering_gmm.csv"
data = pd.read_csv(address)

# Plotting data
x = data.iloc[:, 0]
y = data.iloc[:, 1]
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='green', label='Data points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Data Plot')
plt.legend()
plt.show()

X = data.values
k = abs(int(eval(input("Enter the number of Clusters: "))))

means, covariances, weights, responsibilities, log_likelihoods = gmm_em(X, k)
cluster_assignments = np.argmax(responsibilities, axis=1)

cmap = plt.colormaps['tab10']
colors = cmap(np.linspace(0, 1, k))

plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_data = X[cluster_assignments == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=colors[i], label=f'Cluster {i + 1}')
    plt.scatter(means[i, 0], means[i, 1], color='black', marker='x', s=100, label=f'Mean {i + 1}')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('GMM Clustering Results')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(log_likelihoods, label='Log-Likelihood')
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood Convergence')
plt.legend()
plt.show()
