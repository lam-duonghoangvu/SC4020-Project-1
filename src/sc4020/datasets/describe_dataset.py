"""Shape of a dataset, measured before clustering, to decide which validity metrics can be trusted on it.

Density spread and outlier share use each point's distance to its 10th nearest neighbour. With labels,
the classes are also scored as if they were a clustering (silhouette, Davies-Bouldin), and two
classifiers test their shape: 1-nearest-neighbour follows any shape, nearest-centroid only convex,
roughly spherical classes, so a large gap between them means the classes are not convex.
"""

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, NearestNeighbors

N_NEIGHBOURS = 10
OUTLIER_IQR_FACTOR = 3.0  # far outlier: k-NN distance above Q3 + 3 IQR (Tukey's outer fence)
MAX_FOLDS = 5
SILHOUETTE_SAMPLE = 5000
SEED = 0


def describe_shape(X, y=None) -> dict:
    """Density spread and far-outlier share of X; with labels y, also class balance, separation and convexity."""
    X = np.asarray(X, dtype=float)
    neighbour_distances = (
        NearestNeighbors(n_neighbors=N_NEIGHBOURS + 1).fit(X).kneighbors(X)[0][:, -1]
    )
    q1, q3 = np.percentile(neighbour_distances, [25, 75])
    p10, p90 = np.percentile(neighbour_distances, [10, 90])
    shape = {
        "n_points": len(X),
        "n_features": X.shape[1],
        "has_labels": y is not None,
        "density_spread": float(p90 / p10),
        "far_outlier_share": float(
            (neighbour_distances > q3 + OUTLIER_IQR_FACTOR * (q3 - q1)).mean()
        ),
    }
    if y is None:
        return shape
    y = np.asarray(y)
    class_sizes = np.unique(y, return_counts=True)[1]
    n_folds = int(min(MAX_FOLDS, class_sizes.min()))
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
    sample_size = min(SILHOUETTE_SAMPLE, len(X))
    return shape | {
        "n_classes": len(class_sizes),
        "smallest_class_share": float(class_sizes.min() / len(y)),
        "largest_class_share": float(class_sizes.max() / len(y)),
        "label_silhouette": float(
            silhouette_score(X, y, sample_size=sample_size, random_state=SEED)
        ),
        "label_davies_bouldin": float(davies_bouldin_score(X, y)),
        "nearest_neighbour_accuracy": float(
            cross_val_score(KNeighborsClassifier(1), X, y, cv=folds).mean()
        ),
        "nearest_centroid_accuracy": float(
            cross_val_score(NearestCentroid(), X, y, cv=folds).mean()
        ),
    }
