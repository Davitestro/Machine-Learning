# Model_obsorving

This repository is a compact collection of reimplemented AI / machine learning models intended for learning and quick familiarization. Each file contains a simple, readable implementation of a common algorithm.

Purpose

- Provide clear, minimal implementations for study and experimentation.
- Allow quick inspection of common ML techniques and basic usage examples.

Dependencies

- Python 3.x
- numpy
- cvxopt (required for SVM)

How to use

- Inspect the source file for a model to see its constructor and methods.
- Import the class, initialize with training data or parameters, call fit/predict according to the model's API.
- See the file headers for small implementation notes.

Models included (filename — short description)

1. SVM.py — SupportVectorMachine

   - Implements a kernel SVM (linear, polynomial, RBF) solved via cvxopt quadratic programming.
   - Supports hard-margin (C=0) and soft-margin (C>0). Predictions use learned support vectors and Lagrange multipliers.
2. BLC.py — BLC (Basic Linear Classifier / simple centroid classifier)

   - Computes class means and uses the vector difference as weight w and a midpoint threshold b.
   - Provides predict for binary decisions and predict_1v1 for pairwise (one-vs-one) multi-class voting.
3. NB.py — NaiveBayes

   - Discrete-feature naive Bayes with Laplace smoothing (alpha).
   - Estimates class priors and feature likelihoods from unique discrete values and predicts by log-probabilities.
4. KNN.py — K-Nearest Neighbors

   - Basic KNN classifier using Euclidean distance.
   - Constructor expects training data (X_train, y_train) and an odd k; predicts by majority vote among k nearest neighbors.
5. DT.py — Decision Tree (entropy / information gain)

   - Recursive Tree class using entropy and information gain to choose splits over feature thresholds.
   - Supports max_depth and returns class predictions at leaves.
6. Ansamble.py — Ensemble methods (bagging, random_forest)

   - bagging: bootstrap aggregation of a base estimator.
   - random_forest: builds multiple trees via bootstrap sampling and aggregates by majority vote.
   - boosting / stacking: placeholders (not implemented).

Notes and tips

- API details are minimal and differ per file: read the class docstrings / constructors to understand how to instantiate and call fit/predict.
- For SVM, install cvxopt (it must be available to solve QP). If cvxopt is not desired, replace solver call with another QP solver or approximate SVM.
- DT and Ansamble implementations are simple; they can be extended (feature subsampling for random forest, pruning, criteria options).
- These implementations are educational and may not be optimized for production or large datasets.

Contributing

- Pull requests and issues are welcome. When adding models or tests, include a short README in the model folder and small usage examples or tests.

This file is meant to give a quick overview to get familiar with the models in this folder.
