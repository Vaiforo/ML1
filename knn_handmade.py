import numpy as np
from collections import Counter

class KNNHandMade:
    def __init__(self, n_neighbors=5, metric='manhattan', weights='distance'):
        self.n_neighbors = n_neighbors
        self.metric = metric.lower()
        self.weights = weights.lower()

        if n_neighbors <= 0:
            raise ValueError("Ты чё пёс, кол-во соседей должно быть положительным")
        if metric not in ['euclidean', 'manhattan']:
            raise ValueError("Есть только два стула: 'euclidean' or 'manhattan'")
        if weights not in ['uniform', 'distance']:
            raise ValueError("Есть только два стула: 'uniform' or 'distance'")

    def fit(self, X_train, y_train):
        if len(X_train) != len(y_train):
            raise ValueError("X_train и y_train не из одного огорода, длина батвы разная")
        
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        return self

    def predict(self, X_test):
        X_test = np.array(X_test)

        predictions = []
        for x in X_test:
            distances = self.get_distances(x)

            k_indexes = np.argsort(distances)[:self.n_neighbors]
            k_nearest_points = self.y_train[k_indexes]

            if self.weights == 'uniform':
                points_counts = {}
                for point in k_nearest_points:
                    points_counts[point] = points_counts.get(point, 0) + 1

                prediction = max(points_counts, key=points_counts.get)
                predictions.append(prediction)
            else:
                k_distances = distances[k_indexes]
                weights = 1 / k_distances
                
                class_weights = {}
                for point, weight in zip(k_nearest_points, weights):
                    class_weights[point] = class_weights.get(point, 0) + weight
                predictions.append(max(class_weights, key=class_weights.get))
        return np.array(predictions)

    def get_distances(self, x):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        else:
            return np.sum(np.abs(self.X_train - x), axis=1)