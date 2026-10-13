"""Stage 0 screening of the 2023 earthquake locations: is there structure beyond a null model?

Silhouette uses K-Means against nulls with latitude and longitude shuffled independently, on two
feature spaces: 3D unit vectors (the representation used everywhere else) and flat degrees (the
first screening run, kept for comparison). HDBSCAN uses haversine distance against points uniform
in the bounding box.

Usage: python -m sc4020.experiments.earthquakes_screening
Writes results/earthquakes_2023/screening.csv and figures/screening.png.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from sc4020 import generate_figures as gf
from sc4020 import methods
from sc4020.datasets import load_earthquakes_2023
from sc4020.datasets.preprocess_dataset import to_unit_vectors
from sc4020.paths import results_dir
from sc4020.screening import hdbscan_null_test, silhouette_null_test, uniform_box_null

TOPIC = "earthquakes_2023"
SEED = 0
MIN_CLUSTER_SIZES = (25, 50)  # 25 and 50 for about 5,000 entities


def main():
    quakes = load_earthquakes_2023()
    degrees = quakes.coords_deg
    print(f"{len(degrees)} earthquakes")

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
    pd.concat(
        [silhouette.assign(test="silhouette_vs_shuffled"), density.assign(test="hdbscan_vs_uniform_box")],
        ignore_index=True,
    ).to_csv(results_dir(TOPIC) / "screening.csv", index=False)

    panels = []
    for title, data in [("Earthquakes", degrees), ("Uniform null", uniform_box_null(degrees, np.random.default_rng(SEED)))]:
        labels = methods.hdbscan(MIN_CLUSTER_SIZES[0], metric="haversine").fit_predict(np.radians(data))
        panels.append((data[:, ::-1], labels, title))
    fig = gf.plot_screening(
        panels, xlabel="Longitude", ylabel="Latitude",
        title=f"HDBSCAN (haversine, min_cluster_size={MIN_CLUSTER_SIZES[0]})",
    )
    gf.save(fig, TOPIC, "screening")

    print(silhouette.round(3).to_string(index=False))
    print(density.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
