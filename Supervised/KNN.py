import numpy as np


class KNN:
    def __init__(self, k=3):
        self.k = k
        if self.k <= 0 and self.k % 2 != 0:
            raise ValueError("k must be a positive integer")

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.y_train = y_train
        if self.k > len(self.X_train):
            raise ValueError("k must be less than or equal to the number of training samples")


    def predict(self, x_test):
        predicitons = []
        for x in x_test:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            near_k = np.argsort(distances)[:self.k]
            k_near_laber = self.y_train[near_k]
            values, counts = np.unique(k_near_laber, return_counts=True)
            predicitons.append(values[np.argmax(counts)])
        return np.array(predicitons)
