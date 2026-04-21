import numpy as np

class kmeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices]
        for i in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            self.labels_ = np.argmin(distances, axis=1)
            new_centers = np.array([X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.linalg.norm(new_centers - self.cluster_centers_) < self.tol:
                break
            self.cluster_centers_ = new_centers
    
    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)