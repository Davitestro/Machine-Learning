import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt

def dataset_clusters(n_samples=1000, n_features=2, n_clusters=4, cluster_std=1.0, random_state=42):
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return X



def plot_before_after(
    X,
    model,
    title="Unsupervised model",
    use_labels=True,
    reduce_dim=True
):
    """
    X: np.array (n_samples, n_features)
    model: learning model from unsupervised methods
    use_labels: color points by predicted labels
    reduce_dim: if output dimension > 2, reduce to 2D using PCA
    """

    # ---------- FIT ----------
    if hasattr(model, "fit_predict"):
        y_pred = model.fit_predict(X)
    else:
        model.fit(X)
        y_pred = getattr(model, "labels_", None)

    # ---------- TRANSFORM ----------
    if hasattr(model, "transform"):
        X_trans = model.transform(X)
    elif hasattr(model, "embedding_"):
        X_trans = model.embedding_
    elif hasattr(model, "components_"):
        X_trans = X @ model.components_.T
    else:
        X_trans = X.copy()

    # ---------- REDUCE TO 2D ----------
    if reduce_dim and X_trans.shape[1] > 2:
        from sklearn.decomposition import PCA
        X_trans = PCA(n_components=2).fit_transform(X_trans)

    # ---------- PLOT ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X[:, 0], X[:, 1], s=10)
    axes[0].set_title("Before")

    if use_labels and y_pred is not None:
        axes[1].scatter(X_trans[:, 0], X_trans[:, 1], c=y_pred, s=10, cmap="tab10")
    else:
        axes[1].scatter(X_trans[:, 0], X_trans[:, 1], s=10)

    axes[1].set_title("After")

    plt.suptitle(title)
    plt.show()


x = dataset_clusters(n_samples=500, n_features=2, n_clusters=4, cluster_std=0.60, random_state=0)

""" Example with K-Means """

# from KM import kmeans

# model = kmeans(n_clusters=4, random_state=0)
# plot_before_after(x, model, title="K-Means Clustering", use_labels=True, reduce_dim=False)


""" Example with PCA """
# from PCA import PCA


# model = PCA(n_components=2)
# plot_before_after(x, model, title="PCA Dimensionality Reduction", use_labels=False, reduce_dim=False)

