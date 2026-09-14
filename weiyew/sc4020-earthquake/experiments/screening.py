"""Stage 0 screening of earthquake locations (../AGENTS.md).

Usage: python -m experiments.screening
Writes results/screening.csv and figures/screening.png.

Silhouette uses K-means against nulls with latitude and longitude shuffled independently, on two
feature spaces: 3D unit vectors (the choice from notebooks/eda.ipynb, section 5) and flat degrees
(the first screening run, kept for comparison).
HDBSCAN uses haversine distance against points uniform in the bounding box.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from quake.features import to_unit_vectors
from quake.paths import figures_dir, results_dir
from quake.metrics import NOISE_LABEL
from quake.screening import hdbscan_null_test, silhouette_null_test, uniform_box_null
from quake.usgs import load_earthquakes

SEED = 0
# ../AGENTS.md: 25 and 50 for about 5,000 entities.
MIN_CLUSTER_SIZES = (25, 50)


def plot_clusters(ax, degrees, title):
    labels = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZES[0], metric="haversine", copy=True).fit_predict(
        np.radians(degrees)
    )
    noise = labels == NOISE_LABEL
    ax.scatter(degrees[noise, 1], degrees[noise, 0], s=2, c="lightgrey")
    ax.scatter(degrees[~noise, 1], degrees[~noise, 0], s=2, c=labels[~noise], cmap="tab20")
    ax.set(
        xlabel="Longitude",
        ylabel="Latitude",
        title=f"{title}: {labels.max() + 1} clusters, {noise.mean():.0%} noise",
    )


def main():
    events = load_earthquakes()
    degrees = events[["latitude", "longitude"]].to_numpy()
    print(f"{len(events)} earthquakes")

    silhouette = pd.concat(
        [
            silhouette_null_test(degrees, seed=SEED, to_features=lambda d: to_unit_vectors(d[:, 0], d[:, 1])).assign(
                features="unit vectors"
            ),
            silhouette_null_test(degrees, seed=SEED).assign(features="degrees"),
        ],
        ignore_index=True,
    )
    density = hdbscan_null_test(
        np.radians(degrees), MIN_CLUSTER_SIZES, null="uniform_box", metric="haversine", seed=SEED
    ).assign(features="radians (haversine)")
    results = pd.concat(
        [silhouette.assign(test="silhouette_vs_shuffled"), density.assign(test="hdbscan_vs_uniform_box")],
        ignore_index=True,
    )
    results.to_csv(results_dir() / "screening.csv", index=False)

    fig, (ax_real, ax_null) = plt.subplots(1, 2, figsize=(14, 4.5))
    plot_clusters(ax_real, degrees, "Earthquakes")
    plot_clusters(ax_null, uniform_box_null(degrees, np.random.default_rng(SEED)), "Uniform null")
    fig.suptitle(f"HDBSCAN (haversine, min_cluster_size={MIN_CLUSTER_SIZES[0]})")
    fig.tight_layout()
    fig.savefig(figures_dir() / "screening.png", dpi=150)

    print(silhouette.round(3).to_string(index=False))
    print(density.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
