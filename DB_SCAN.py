import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

address = r"C:\Users\aram_\OneDrive\Desktop\AUA\CSV\Clustering_gmm.csv"
data = pd.read_csv(address)

def db_scan(data_, radius_, min_pts):
    data_ = data_.values
    n_points = data_.shape[0]

    labels_ = np.full(n_points, -1)
    cluster_id = 0

    def region_query(point_idx_):
        neighbors_ = []
        for i in range(n_points):
            if np.linalg.norm(data_[point_idx_] - data_[i]) <= radius_:
                neighbors_.append(i)
        return neighbors_

    def expand_cluster(point_idx_, neighbors_):
        nonlocal cluster_id
        labels_[point_idx_] = cluster_id

        i = 0
        while i < len(neighbors_):
            neighbor_idx = neighbors_[i]
            if labels_[neighbor_idx] == -1:
                labels_[neighbor_idx] = cluster_id
            elif labels_[neighbor_idx] == 0:
                labels_[neighbor_idx] = cluster_id
                new_neighbors = region_query(neighbor_idx)
                if len(new_neighbors) >= min_pts:
                    neighbors_ += new_neighbors
            i += 1


    for point_idx in range(n_points):
        if labels_[point_idx] != -1:  # already visited
            continue
        neighbors = region_query(point_idx)
        if len(neighbors) < min_pts:
            labels_[point_idx] = -1  # mark as noise
        else:
            cluster_id += 1
            expand_cluster(point_idx, neighbors)
    return labels_


radius = 3
minPts = 3

labels = db_scan(data, radius, minPts)
data = data.values

plt.figure(figsize=(10, 8))
plt.scatter(data[labels == -1, 0], data[labels == -1, 1], c='k', marker='x', label='Noise')

unique_labels = set(labels) - {-1}
colors = plt.cm.get_cmap("tab10", len(unique_labels))
for label, color in zip(unique_labels, colors.colors):
    plt.scatter(data[labels == label, 0], data[labels == label, 1], color=color, label=f'Cluster {label}')

plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
