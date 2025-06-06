import numpy as np


class KMeansHandMade:
    def __init__(self, k, max_iters=300, eps=1e-4, seed=None):
        self.k = k
        self.max_iters = max_iters
        self.eps = eps
        self.seed = seed

    def fit(self, X):
        np.random.seed(self.seed)
        n_samples, n_features = X.shape

        initial_indices = np.random.choice(n_samples, self.k, replace=False)
        centroids = X[initial_indices].astype(float)

        labels = np.zeros(n_samples, dtype=int)

        for iteration in range(self.max_iters):
            distances = np.zeros((n_samples, self.k), dtype=float)
            for i in range(n_samples):
                for j in range(self.k):
                    diff = X[i] - centroids[j]
                    distances[i, j] = np.sqrt(np.dot(diff, diff))
            for i in range(n_samples):
                labels[i] = np.argmin(distances[i])

            new_centroids = np.zeros((self.k, n_features), dtype=float)
            for cluster_idx in range(self.k):
                members = X[labels == cluster_idx]
                if len(members) == 0:
                    rand_idx = np.random.randint(0, n_samples)
                    new_centroids[cluster_idx] = X[rand_idx]
                else:
                    new_centroids[cluster_idx] = np.mean(members, axis=0)

            shift = np.abs(new_centroids - centroids)
            if np.all(shift < self.eps):
                centroids = new_centroids
                break

            centroids = new_centroids

        self.centroids = centroids
        self.labels_ = labels

        total_inertia = 0
        for i in range(n_samples):
            center = centroids[labels[i]]
            diff = X[i] - center
            total_inertia += np.dot(diff, diff)
        self.inertia_ = total_inertia

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Не выполнен fit")

        m_samples = X.shape[0]
        labels = np.zeros(m_samples, dtype=int)

        for i in range(m_samples):
            min_dist = None
            best_cluster = None
            for j in range(self.k):
                diff = X[i] - self.centroids[j]
                d = np.sqrt(np.dot(diff, diff))
                if (min_dist is None) or (d < min_dist):
                    min_dist = d
                    best_cluster = j
            labels[i] = best_cluster

        return labels
