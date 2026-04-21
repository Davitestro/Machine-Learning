import numpy as np

class BLC:
    def __init__(self):
        pass

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.variat = np.unique(y)

        x_pos = self.x[self.y == 1]
        x_neg = self.x[self.y == 0]

        mean_pos = np.mean(x_pos, axis=0)
        mean_neg = np.mean(x_neg, axis=0)

        m = (mean_pos + mean_neg) / 2
        self.w = mean_pos - mean_neg

        self.b = np.dot(self.w, m)
    def predict(self, test_data):
        predictions = []
        for sample in test_data:
            result = np.dot(self.w, sample) - self.b
            if result >= 0:
                predictions.append(1)
            else:
                predictions.append(0)
        return np.array(predictions)
    
    def predict_1v1(self, test_data):
        predictions = []
        for sample in test_data:
            results = []
            for i in range(len(self.variat)):
                for j in range(i + 1, len(self.variat)):
                    class_i_data = self.x[self.y == self.variat[i]]
                    class_j_data = self.x[self.y == self.variat[j]]

                    mean_i = np.mean(class_i_data, axis=0)
                    mean_j = np.mean(class_j_data, axis=0)

                    m = (mean_i + mean_j) / 2
                    w = mean_i - mean_j
                    b = np.dot(w, m)

                    result = np.dot(w, sample) - b
                    if result >= 0:
                        results.append(self.variat[i])
                    else:
                        results.append(self.variat[j])
            # Majority vote
            unique, counts = np.unique(results, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            predictions.append(majority_class)
        return np.array(predictions)