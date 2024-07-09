import numpy as np
from collections import deque

class DBSCAN:
    def __init__(self, eps, min_pts):
        self.eps = eps
        self.min_pts = min_pts

    def fit_predict(self, X):
        self.X = X
        self.labels = np.zeros(len(X), dtype=int)  # Initialize labels
        cluster_id = 0

        for i in range(len(X)):
            if self.labels[i] == 0:  # If the point is not visited
                neighbors = self.region_query(i)
                if len(neighbors) < self.min_pts:
                    self.labels[i] = -1  # Mark as noise
                else:
                    cluster_id += 1
                    self.labels[i] = cluster_id
                    self.expand_cluster(neighbors, cluster_id)
                    
        return self.labels

    def expand_cluster(self, neighbors, cluster_id):
        i = 0
        while i < len(neighbors):
            point = neighbors[i]
            if self.labels[point] == 0:
                self.labels[point] = cluster_id
                new_neighbors = self.region_query(point)
                if len(new_neighbors) >= self.min_pts:
                    neighbors += new_neighbors
            elif self.labels[point] == -1:  # If previously marked as noise
                self.labels[point] = cluster_id
            i += 1

    def region_query(self, index):
        neighbors = []
        for i in range(len(self.X)):
            if np.linalg.norm(self.X[index] - self.X[i]) < self.eps:
                neighbors.append(i)
        return neighbors


# Example usage
if __name__ == "__main__":
    # Generating sample data
    np.random.seed(0)
    X = np.random.randn(100, 2)

    # Instantiating DBSCAN object
    dbscan = DBSCAN(eps=0.5, min_pts=5)

    # Fitting the data and obtaining cluster labels
    labels = dbscan.fit_predict(X)

    # Printing cluster labels
    print(labels)
