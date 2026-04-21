import numpy as np
from collections import Counter

class Tree:
    def __init__(self, depth=0, max_depth=5):
        self.depth = depth
        self.max_depth = max_depth
        self.is_leaf = False



    def entropy(self, y, weights):
        total_weight = np.sum(weights)
        ent = 0.0
        for label in np.unique(y):
            w = np.sum(weights[y == label])
            p = w / total_weight
            ent -= p * np.log2(p + 1e-10)
        return ent

    def info_gain(self, left_y, right_y, left_w, right_w, parent_entropy):
        total_weight = np.sum(left_w) + np.sum(right_w)
        child_entropy = (np.sum(left_w) / total_weight) * self.entropy(left_y, left_w) + \
                        (np.sum(right_w) / total_weight) * self.entropy(right_y, right_w)
        return parent_entropy - child_entropy

    def fit(self, X_train, y_train, sample_weights=None):
        self.x = X_train
        self.y = y_train

        n_samples = len(self.y)
        if sample_weights is None:
            self.sample_weights = np.ones(n_samples) / n_samples
        else:
            self.sample_weights = sample_weights / np.sum(sample_weights)

        if len(set(self.y)) == 1 or self.depth == self.max_depth:
            self.is_leaf = True
            self.prediction = max(set(self.y), key=list(self.y).count)
            return


        best_gain = 0
        best_split = None
        parent_entropy = self.entropy(self.y, self.sample_weights)
        _, n_features = self.x.shape

        for feature_index in range(n_features):
            thresholds = np.unique(self.x[:, feature_index])
            for threshold in thresholds:
                left_indices = self.x[:, feature_index] <= threshold
                right_indices = self.x[:, feature_index] > threshold
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                left_w = self.sample_weights[left_indices]
                right_w = self.sample_weights[right_indices]

                gain = self.info_gain(self.y[left_indices], self.y[right_indices],
                                      left_w, right_w, parent_entropy)
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices
                    }

        if best_gain > 0:
            self.feature_index = best_split['feature_index']
            self.threshold = best_split['threshold']

            self.left = Tree(

                depth=self.depth + 1,
                max_depth=self.max_depth
            )
            self.right = Tree(

                depth=self.depth + 1,
                max_depth=self.max_depth
            )
            self.left.fit(
                self.x[best_split['left_indices']],
                self.y[best_split['left_indices']],
                sample_weights=self.sample_weights[best_split['left_indices']],
                )
            self.right.fit(
                self.x[best_split['right_indices']],
                self.y[best_split['right_indices']],
                sample_weights=self.sample_weights[best_split['right_indices']],
            )
        else:
            self.is_leaf = True
            self.prediction = Counter(self.y).most_common(1)[0][0]

    def _predict_one(self, x):
        if self.is_leaf:
            return self.prediction
        if x[self.feature_index] <= self.threshold:
            return self.left._predict_one(x)
        else:
            return self.right._predict_one(x)

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])
