import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Generate some sample data
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Initialize the DBSCAN model
dbscan_model = DBSCAN(eps=0.5, min_samples=10)

# Fit the model to the data
dbscan_model.fit(X)

# Get the cluster labels
labels = dbscan_model.labels_

# Print the number of clusters found
print("Number of clusters:", len(set(labels)))

# Print the cluster labels
print("Cluster labels:", labels)

# Plot the data and the cluster centers
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(dbscan_model.components_[:, 0], dbscan_model.components_[:, 1], c='red')
plt.show()