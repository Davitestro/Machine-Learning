import numpy as np
from DT import Tree
from sklearn.model_selection import KFold
from copy import deepcopy

class bagging:
    def __init__(self, base_estimator=None, n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators_ = []
    
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.estimators_ = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            estimator = self.base_estimator()
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)
    
    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        # Голосование: если >= половины проголосовали за 1 → итог = 1
        final = (np.sum(predictions, axis=0) >= (len(self.estimators_) / 2)).astype(int)
        return final

        

class random_forest:
    def __init__(self, n_estimators=10, depth=0, max_depth=None):
        self.n_estimators = n_estimators
        self.depth = depth
        self.max_depth = max_depth
        self.trees = []
    
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        # Сохраняем оригинальные классы
        self.original_classes = np.unique(y)
        self.class_to_int = {c: i for i, c in enumerate(self.original_classes)}
        self.int_to_class = {i: c for c, i in self.class_to_int.items()}
        
        # Перекодируем y -> int
        y_encoded = np.array([self.class_to_int[val] for val in y])

        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap_sample(X, y_encoded)
            tree = Tree(X_sample, y_sample, depth=self.depth, max_depth=self.max_depth)
            tree.fit()
            self.trees.append(tree)


    def predict(self, X):
        # Все деревья возвращают int-классы
        predictions = np.array([tree.predict(X) for tree in self.trees])
        final_predictions = []

        for i in range(X.shape[0]):
            pred = predictions[:, i]

            # На случай если что-то поехало
            pred = pred.astype(int)
            if np.any(pred < 0):
                pred = pred - pred.min()

            counts = np.bincount(pred)
            final_predictions.append(np.argmax(counts))

        # Маппим обратно: int -> оригинальный класс
        return np.array([self.int_to_class[p] for p in final_predictions])





class boosting:
    def __init__(self,model=None, n_estimators=10):
        self.model = model
        self.n_estimators = n_estimators
        self.estimators_ = []
        self.estimator_weights_ = []
    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            print("Training estimator", i+1)
            estimator = self.model()
            estimator.fit(X, y, sample_weights=sample_weights)
            predictions = estimator.predict(X)
            incorrect = (predictions != y)
            error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)
            if error > 0.5:
                continue
            estimator_weight = 0.5 * np.log((1 - error) / (error + 1e-10))
            sample_weights *= np.exp(-estimator_weight * y * predictions)
            sample_weights /= np.sum(sample_weights)
            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)
    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            final_predictions += weight * estimator.predict(X)
        return np.sign(final_predictions)




class stacking:
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

        self.fitted_base_models = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = X.shape[0]
        n_models = len(self.base_models)

        # meta-features
        Z = np.zeros((n_samples, n_models))

        self.fitted_base_models = []

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for m_idx, model in enumerate(self.base_models):
            fold_models = []

            for train_idx, val_idx in kf.split(X):
                m = deepcopy(model)

                m.fit(X[train_idx], y[train_idx])
                Z[val_idx, m_idx] = m.predict(X[val_idx])

                fold_models.append(m)

            self.fitted_base_models.append(fold_models)

        # обучаем meta-model
        self.meta_model.fit(Z, y)

        return self

    def predict(self, X):
        X = np.asarray(X)

        n_samples = X.shape[0]
        n_models = len(self.fitted_base_models)

        Z = np.zeros((n_samples, n_models))

        for m_idx, fold_models in enumerate(self.fitted_base_models):
            preds = np.column_stack([
                model.predict(X) for model in fold_models
            ])
            Z[:, m_idx] = preds.mean(axis=1)

        return self.meta_model.predict(Z)