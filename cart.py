import numpy as np


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, max_depth=None, classification_task=True):
        self.max_depth = max_depth
        self.classification_task = classification_task
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        _, num_features = X.shape
        unique_values = np.unique(y)

        if len(unique_values) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return TreeNode(value=self._leaf_value(y))

        best_feature, best_threshold = self._best_split(X, y, num_features)
        if best_feature is None:
            return TreeNode(value=self._leaf_value(y))

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_node = self._grow_tree(
            X[left_indices], y[left_indices], depth + 1)
        right_node = self._grow_tree(
            X[right_indices], y[right_indices], depth + 1)

        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

    def _best_split(self, X, y, num_features):
        best_gain = -1 if self.classification_task else float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if self.classification_task:
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold
                else:
                    if gain < best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        if self.classification_task:
            return self._gini_gain(X, y, feature, threshold)
        else:
            return self._mse_gain(X, y, feature, threshold)

    def _gini_gain(self, X, y, feature, threshold):
        parent_loss = self._gini(y)

        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0

        num_samples = len(y)
        num_left = np.sum(left_indices)
        num_right = np.sum(right_indices)

        child_loss = (num_left / num_samples) * self._gini(y[left_indices]) + (
            num_right / num_samples) * self._gini(y[right_indices])

        return parent_loss - child_loss

    def _mse_gain(self, X, y, feature, threshold):
        self._mse(y)

        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return float('inf')

        num_samples = len(y)
        num_left = np.sum(left_indices)
        num_right = np.sum(right_indices)

        child_loss = (num_left / num_samples) * self._mse(y[left_indices]) + (
            num_right / num_samples) * self._mse(y[right_indices])

        return child_loss

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        return 1 - sum((count / total_samples) ** 2 for count in counts)

    def _mse(self, y):
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _leaf_value(self, y):
        if self.classification_task:
            return np.bincount(y.astype(int)).argmax()
        else:
            return np.mean(y)

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.root) for sample in X])

    def _predict_sample(self, sample, node):
        if node.value is not None:
            return node.value
        if sample[node.feature] < node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)
