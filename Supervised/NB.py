import numpy as np

class NaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.priors = {}
        self.likelihoods = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        # prior вероятности классов
        for c in self.classes:
            self.priors[c] = np.sum(y == c) / len(y)

        # вероятности признаков по классам
        for c in self.classes:
            X_c = X[y == c]
            self.likelihoods[c] = []

            for i in range(n_features):
                values, counts = np.unique(X_c[:, i], return_counts=True)
                total = len(X_c)
                probs = {}

                for val in values:
                    probs[val] = (counts[values == val][0] + self.alpha) / (total + self.alpha * len(values))

                self.likelihoods[c].append(probs)

    def predict(self, X):
        preds = []

        for x in X:
            class_probs = {}

            for c in self.classes:
                prob = np.log(self.priors[c])

                for i, val in enumerate(x):
                    probs = self.likelihoods[c][i]

                    # если значение не встречалось — используем Laplace-гладкость
                    if val in probs:
                        prob += np.log(probs[val])
                    else:
                        # невидимое значение
                        unseen = self.alpha / (self.alpha * len(probs))
                        prob += np.log(unseen)

                class_probs[c] = prob

            preds.append(max(class_probs, key=class_probs.get))

        return np.array(preds)
