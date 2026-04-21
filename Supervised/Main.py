from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


def evaluate_model(model_name, model, X_test, y_test):
    # try:
    preds = model.predict(X_test)
    # except AttributeError:
    #     preds = model.predict_1v1(X_test)

    # ensure shape is correct
    preds = np.array(preds).ravel()
    y_test = np.array(y_test).ravel()

    # compute accuracy
    acc = accuracy_score(y_test, preds)

    # print detailed comparison
    print(f"\n--- {model_name} ---")
    print(f"Predictions: {preds}")
    print(f"Real labels: {y_test}")
    print(f"Accuracy: {acc:.4f}")

    # show mismatches (if any)
    mismatches = np.where(preds != y_test)[0]
    if len(mismatches) == 0:
        print("✅ All predictions correct!")
    else:
        print("❌ Mismatched indices:", mismatches)
        for i in mismatches:
            print(f"   sample {i}: predicted {preds[i]}, real {y_test[i]}")



'''regresion/classification moddels dataset'''

def plot_decision_surfaces(model, X, y):
  fig, ax = plt.subplots()

  # plot the data, the decision boundary and mark support vectors
  ax.plot(X[y==1, 0], X[y==1, 1], "ro", label="positives")
  ax.plot(X[y==-1, 0], X[y==-1, 1], "bo", label="negatives")

  ax.plot(model.support_vectors[:, 0], model.support_vectors[:, 1],
          "*", color="black", markersize=12, label="support vectors")

  xmin, ymin = np.min(X, axis=0)-2
  xmax, ymax = np.max(X, axis=0)+2

  xx, yy = np.meshgrid(np.linspace(xmin, xmax+1, 100),
                       np.linspace(ymin, ymax+1, 100))

  grid = pd.DataFrame({'x':xx.flatten(),'y':yy.flatten()})

  y_pred = model.predict(grid)
  grid[y_pred == 1].plot(x='x',y='y',kind='scatter',
                               s=20,color='pink', ax=ax)
  grid[y_pred == -1].plot(x='x',y='y',kind='scatter',
                               s=20,color='cyan', ax=ax)

  plt.xlim(xmin, xmax)
  plt.xticks(np.arange(xmin, xmax+1))
  plt.xlabel("X1")

  plt.ylim(ymin, ymax)
  plt.yticks(np.arange(ymin, ymax+1))
  plt.ylabel("X2")

  ax.legend(loc = 'upper left')

  plt.show()

def generate_multiclass_dataset(num_classes=3, points_per_class=50, spread=0.3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    X = []
    y = []
    for cls in range(num_classes):
        # случайный центр класса
        center = np.random.rand(2) * 5
        # генерируем точки вокруг центра
        X_cls = np.random.randn(points_per_class, 2) * spread + center
        X.append(X_cls)
        y.extend([cls]*points_per_class)
    
    X = np.vstack(X)
    y = np.array(y)
    return X, y

def generate_hard_classification(n_samples=5000, noise=0.2):
    """
    Boosting / Tree
    """
    X = np.random.randn(n_samples, 2)

    # XOR-логика
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

    # Добавляем noise
    flip_mask = np.random.rand(n_samples) < noise
    y[flip_mask] = 1 - y[flip_mask]

    return shuffle(X, y)


def generate_multi_interactions(n_samples=5000):
    """
    Random Forest / Boosting.
    """
    X = np.random.randn(n_samples, 5)

    # сложная функция с 3 взаимодействиями
    f = (
        2 * X[:, 0] * X[:, 1] +
        1.5 * np.sin(3 * X[:, 2]) +
        2.5 * (X[:, 3] > 0) * (X[:, 4] < 0)
    )

    # переводим в классы
    y = (f > np.median(f)).astype(int)

    return shuffle(X, y)


def generate_fractal_data(n_samples=6000):
    """
    for ansamble trees
    """
    X = np.random.rand(n_samples, 2)

    def cantor(x):
        # принадлежит ли x кантору
        while x > 0:
            if 1/3 < x < 2/3:
                return False
            x *= 3
        return True

    y = np.array([
        int(cantor(x1) ^ cantor(x2))
        for x1, x2 in X
    ])

    return shuffle(X, y)

# --- пример использования ---
X, y = generate_multiclass_dataset(num_classes=2, points_per_class=100, spread=0.4, seed=42)

for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


'''classification/classification moddels dataset'''

# def generate_weather_dataset(n_samples=100):
#     weather = ["Sunny", "Overcast", "Rainy"]
#     temperature = ["Hot", "Mild", "Cool"]
#     humidity = ["High", "Normal"]
#     windy = [True, False]
    
#     data = []
#     for _ in range(n_samples):
#         w = random.choice(weather)
#         t = random.choice(temperature)
#         h = random.choice(humidity)
#         wi = random.choice(windy)
        
#         # простая логика для целевой переменной
#         # Decision Tree сможет уловить эту зависимость
#         if w == "Overcast":
#             play = "Yes"
#         elif w == "Sunny" and h == "High":
#             play = "No"
#         elif w == "Rainy" and wi:
#             play = "No"
#         else:
#             play = "Yes"
        
#         data.append([w, t, h, wi, play])
    
#     df = pd.DataFrame(data, columns=["Weather", "Temperature", "Humidity", "Windy", "Play"])
#     return df

# # генерим
# df = generate_weather_dataset(200)


# X_train, X_test, y_train, y_test = train_test_split(
#     df[["Weather", "Temperature", "Humidity", "Windy"]],
#     df["Play"],
#     test_size=0.2,
#     random_state=42
# )


""" BLC: Bayesian Linear Classifier """
# from BLC import BLC


# blc = BLC(X_train, y_train)

# evaluate_model("BLC", blc, X_test, y_test)


"""KNN: K-Nearest Neighbors Classifier """
# from KNN import KNN

# knn = KNN(X_train, y_train, k=3)

# evaluate_model("KNN", knn, X_test, y_test)


"""Evaluation of Decision Tree Classifier"""

# from DT import Tree

# dt = Tree(X_train.values, y_train.values, max_depth=5)

# dt.fit()

# evaluate_model("Decision Tree", dt, X_test.values, y_test.values)


"""Naive Bayes Classifier"""
# from NB import NaiveBayes

# # aplha optimum 89
# NB = NaiveBayes(0.89)

# NB.fit(X_train.values, y_train.values)

# evaluate_model("Naive Bayes", NB, X_test.values, y_test.values)


"""SVM: Support Vector Machine Classifier"""

# from SVM import SupportVectorMachine


# svm = SupportVectorMachine(C=1.0, kernel_name = 'linear', gamma=0.2, power=3)
# svm.fit(X_train, y_train)
# evaluate_model("Support Vector Machine", svm, X_test, y_test)


"""Ansamble Methods Evaluation"""

from Ansamble import bagging, boosting, stacking, random_forest
from SVM import SupportVectorMachine
from DT import Tree
from KNN import KNN
from BLC import BLC



# X, y = generate_hard_classification(n_samples=2000, noise=0.15)
# y = np.where(y == 0, -1, 1)  # Convert 0 labels to -1 for SVM compatibility
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# svm_base = lambda: SupportVectorMachine(C=1.0, kernel_name = 'rbf', gamma=0.5)
# tree_base = lambda: Tree(X_train, y_train, max_depth=5)
# bagging_model = bagging(base_estimator=svm_base, n_estimators=5)

# bagging_model.fit(X_train, y_train)
# y_test = np.where(y_test == -1, 0, 1)  # Convert 0 labels to -1 for SVM compatibility
# evaluate_model("Bagging with SVM", bagging_model, X_test, y_test)


# X, y = generate_multi_interactions(n_samples=2000)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# forest_model = random_forest(n_estimators=10, depth=0, max_depth=5)
# forest_model.fit(X_train, y_train)
# tree = random_forest(n_estimators=10, depth=0, max_depth=5)
# tree.fit(X_train, y_train)

# evaluate_model("Random Forest", tree, X_test, y_test)

# X,y = generate_multi_interactions()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# tree_base = lambda: Tree(max_depth=5)
# boosting_model = boosting(model=tree_base, n_estimators=10)
# boosting_model.fit(X_train, y_train)

# evaluate_model("Boosting with Decision Trees", boosting_model, X_test, y_test)

# X, y = generate_multi_interactions(n_samples=2000)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# base_models = [
#     Tree(max_depth=5),
#     KNN(k=5),
#     SupportVectorMachine(C=1.0, kernel_name='rbf', gamma=0.5)
# ]

# meta_model = BLC()

# stack = stacking(
#     base_models=base_models,
#     meta_model=meta_model,
#     n_folds=5
# )

# stack.fit(X_train, y_train)

# evaluate_model("Stacking Ensemble", stack, X_test, y_test)


"""Dataset Visualization"""

'''XOR dataset visualization (for reference)'''
# plot_decision_surfaces(bagging, X, y)

'''Decision Tree dataset visualization'''

# print(df.head())
# print(df.info())