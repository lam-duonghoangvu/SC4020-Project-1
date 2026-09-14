"""Stage 0 screening of HDB resale transactions (../AGENTS.md, ../PLAN_HDB.md).

Usage: python -m experiments.screening

On a random 5,000-row sample across all years (seed 0), with the four features in
hdb.features.SCREENING_FEATURES standardized:
  results/screening.csv           silhouette vs shuffled null (K 2 to 8); HDBSCAN vs shuffled null (25, 50)
  results/screening_labels.csv    K-means ARI vs flat type and town, and a floor-area-only baseline
  results/screening_discrete.csv  do HDBSCAN clusters follow single discrete values? real vs jittered data
  results/screening_roles.csv     raw vs standardized scale, nominal vs adjusted price, HDBSCAN without storey
  results/screening_kmeans_profiles.csv  median features of the K=4 K-means clusters
  figures/screening.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from hdb.features import SCREENING_FEATURES, jitter_ties, load_price_index, load_resale, screening_features, standardize
from hdb.paths import figures_dir, results_dir
from hdb.screening import NOISE_LABEL, cluster_counts, hdbscan_null_test, shuffled_null, silhouette_null_test, single_value_share

SEED = 0
SAMPLE_SIZE = 5000
# ../AGENTS.md: 25 and 50 for about 5,000 entities.
MIN_CLUSTER_SIZES = (25, 50)


def label_agreement(sample, X):
    rows = []
    for label in ["flat_type", "town"]:
        k = sample[label].nunique()
        for features, data in [("all four", X), ("floor area only", X[:, [SCREENING_FEATURES.index("floor_area_sqm")]])]:
            labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(data)
            rows.append({"label": label, "k": k, "features": features, "ari": adjusted_rand_score(sample[label], labels)})
    return pd.DataFrame(rows)


def discrete_value_check(sample, features, rng):
    """HDBSCAN on real and jittered values; for real clusters, how often one discrete value fills a cluster."""
    jittered = jitter_ties(pd.DataFrame(features, columns=SCREENING_FEATURES), rng).to_numpy()
    rows = []
    labels_by_data = {}
    for name, data in [("real", features), ("jittered", jittered)]:
        for size in MIN_CLUSTER_SIZES:
            labels = HDBSCAN(min_cluster_size=size, copy=True).fit_predict(standardize(data))
            labels_by_data[(name, size)] = labels
            row = {"data": name, "min_cluster_size": size, **cluster_counts(labels)}
            for column, values in [
                ("storey_range", sample["storey_range"]),
                ("floor_area_sqm", sample["floor_area_sqm"]),
                ("remaining_lease", sample["remaining_lease"]),
                ("flat_type", sample["flat_type"]),
            ]:
                shares = single_value_share(labels, values)
                row[f"clusters_single_{column}"] = int((shares == 1.0).sum())
                row[f"median_share_{column}"] = float(shares.median()) if len(shares) else np.nan
            rows.append(row)
    return pd.DataFrame(rows), labels_by_data


def role_checks(sample, features):
    """Checks for the report role found after the first screening run (../PLAN_HDB.md, Stage 0 results).

    Returns one row per check, and the median features of the K=4 standardized K-means clusters.
    """
    X = standardize(features)
    year = sample["month"].str[:4]
    nominal = features.assign(real_price=sample["resale_price"].astype(float))
    rows = []

    def km(data, k):
        return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(data)

    raw_labels = km(features.to_numpy(), 6)
    price_ranges = features.groupby(raw_labels)["real_price"].agg(["min", "max"]).sort_values("min")
    overlaps = int((price_ranges["min"].to_numpy()[1:] <= price_ranges["max"].to_numpy()[:-1]).sum())
    rows += [
        {"check": "scale", "setting": "raw features, K=6", "metric": "ari_flat_type", "value": adjusted_rand_score(sample["flat_type"], raw_labels)},
        {"check": "scale", "setting": "standardized, K=6", "metric": "ari_flat_type", "value": adjusted_rand_score(sample["flat_type"], km(X, 6))},
        {"check": "scale", "setting": "raw features, K=6", "metric": "overlapping_price_ranges", "value": overlaps},
        {"check": "price_adjustment", "setting": "nominal price, K=10", "metric": "ari_sale_year", "value": adjusted_rand_score(year, km(standardize(nominal), 10))},
        {"check": "price_adjustment", "setting": "adjusted price, K=10", "metric": "ari_sale_year", "value": adjusted_rand_score(year, km(X, 10))},
    ]

    without_storey = standardize(features.drop(columns="storey_mid"))
    for name, data in [("real", without_storey), ("shuffled null", shuffled_null(without_storey, np.random.default_rng(SEED)))]:
        counts = cluster_counts(HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZES[0], copy=True).fit_predict(data))
        rows += [{"check": "hdbscan_without_storey", "setting": name, "metric": metric, "value": value} for metric, value in counts.items()]

    labels = km(X, 4)
    profiles = (
        features.assign(cluster=labels, flat_type=sample["flat_type"].to_numpy())
        .groupby("cluster")
        .agg(
            n=("floor_area_sqm", "size"),
            floor_area_sqm=("floor_area_sqm", "median"),
            remaining_lease_years=("remaining_lease_years", "median"),
            storey_mid=("storey_mid", "median"),
            real_price=("real_price", "median"),
            top_flat_type=("flat_type", lambda v: v.value_counts().index[0]),
        )
    )
    return pd.DataFrame(rows), profiles


def main():
    resale = load_resale()
    price_index = load_price_index()
    features = screening_features(resale, price_index)
    print(f"{len(resale)} transactions, {len(features)} with a price index value")

    sample_index = features.sample(n=SAMPLE_SIZE, random_state=SEED).index
    sample = resale.loc[sample_index]
    raw = features.loc[sample_index, SCREENING_FEATURES].to_numpy()
    X = standardize(raw)

    silhouette = silhouette_null_test(X, seed=SEED)
    density = hdbscan_null_test(X, MIN_CLUSTER_SIZES, seed=SEED)
    pd.concat(
        [silhouette.assign(test="silhouette_vs_shuffled"), density.assign(test="hdbscan_vs_shuffled")],
        ignore_index=True,
    ).assign(sample_size=SAMPLE_SIZE, seed=SEED).to_csv(results_dir() / "screening.csv", index=False)

    labels = label_agreement(sample, X)
    labels.to_csv(results_dir() / "screening_labels.csv", index=False)

    discrete, labels_by_data = discrete_value_check(sample, raw, np.random.default_rng(SEED))
    discrete.to_csv(results_dir() / "screening_discrete.csv", index=False)

    roles, profiles = role_checks(sample, features.loc[sample_index, SCREENING_FEATURES])
    roles.to_csv(results_dir() / "screening_roles.csv", index=False)
    profiles.to_csv(results_dir() / "screening_kmeans_profiles.csv")

    null = shuffled_null(X, np.random.default_rng(SEED))
    null_labels = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZES[0], copy=True).fit_predict(null)
    pca = PCA(n_components=2).fit(X)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(silhouette["k"], silhouette["real_silhouette"], "o-", label="real")
    axes[0].plot(silhouette["k"], silhouette["null_silhouette"], "s--", label="shuffled null (mean of 3)")
    axes[0].set(xlabel="K", ylabel="Silhouette", title="K-means silhouette", ylim=(0, None))
    axes[0].legend()
    for ax, data, point_labels, title in [
        (axes[1], X, labels_by_data[("real", MIN_CLUSTER_SIZES[0])], "Real"),
        (axes[2], null, null_labels, "Shuffled null"),
    ]:
        points = pca.transform(data)
        noise = point_labels == NOISE_LABEL
        ax.scatter(points[noise, 0], points[noise, 1], s=2, c="lightgrey")
        ax.scatter(points[~noise, 0], points[~noise, 1], s=2, c=point_labels[~noise], cmap="tab20")
        counts = cluster_counts(point_labels)
        ax.set(
            xlabel="PC1",
            ylabel="PC2",
            title=f"{title}: {counts['n_clusters']} clusters, {counts['noise_fraction']:.0%} noise",
        )
    fig.suptitle(f"HDB resale screening, {SAMPLE_SIZE:,} sales (HDBSCAN min_cluster_size={MIN_CLUSTER_SIZES[0]}, PCA view)")
    fig.tight_layout()
    fig.savefig(figures_dir() / "screening.png", dpi=150)

    with pd.option_context("display.width", 200):
        print(silhouette.round(3).to_string(index=False))
        print(density.round(3).to_string(index=False))
        print(labels.round(3).to_string(index=False))
        print(discrete.round(3).T.to_string())
        print(roles.round(3).to_string(index=False))
        print(profiles.round(1).to_string())


if __name__ == "__main__":
    main()
