"""Stage 0 screening of HDB resale sales: is there structure beyond a null model, and what drives it?

On a random 5,000-sale sample across all years (seed 0), with the four features standardized:
  screening.csv                 silhouette vs shuffled null (K 2 to 8); HDBSCAN vs shuffled null (25, 50)
  screening_labels.csv          K-Means ARI vs flat type and town, and a floor-area-only baseline
  screening_discrete.csv        do HDBSCAN clusters follow single discrete values? real vs jittered data
  screening_roles.csv           raw vs standardized scale, nominal vs adjusted price, HDBSCAN without storey
  screening_kmeans_profiles.csv median features of the K=4 K-Means clusters
  figures/screening.png

Usage: python -m sc4020.experiments.hdb_screening
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from sc4020 import generate_figures as gf
from sc4020 import methods
from sc4020.datasets import load_hdb_resale
from sc4020.datasets.preprocess_dataset import HDB_FEATURES, jitter_ties, standardize
from sc4020.paths import results_dir
from sc4020.screening import cluster_counts, hdbscan_null_test, shuffled_null, silhouette_null_test, single_value_share

TOPIC = "hdb_resale"
SEED = 0
SAMPLE_SIZE = 5000
MIN_CLUSTER_SIZES = (25, 50)  # 25 and 50 for about 5,000 entities


def kmeans_labels(data, k):
    return methods.kmeans(k, random_state=SEED).fit_predict(data)


def label_agreement(sales, X):
    rows = []
    for label in ["flat_type", "town"]:
        k = sales[label].nunique()
        for features, data in [("all four", X), ("floor area only", X[:, [HDB_FEATURES.index("floor_area_sqm")]])]:
            rows.append({"label": label, "k": k, "features": features, "ari": adjusted_rand_score(sales[label], kmeans_labels(data, k))})
    return pd.DataFrame(rows)


def discrete_value_check(sales, features, rng):
    """HDBSCAN on real and jittered values; for real clusters, how often one discrete value fills a cluster."""
    jittered = jitter_ties(pd.DataFrame(features, columns=HDB_FEATURES), rng).to_numpy()
    rows, labels_by_data = [], {}
    for name, data in [("real", features), ("jittered", jittered)]:
        for size in MIN_CLUSTER_SIZES:
            labels = methods.hdbscan(size).fit_predict(standardize(data))
            labels_by_data[(name, size)] = labels
            row = {"data": name, "min_cluster_size": size, **cluster_counts(labels)}
            for column in ["storey_range", "floor_area_sqm", "remaining_lease", "flat_type"]:
                shares = single_value_share(labels, sales[column])
                row[f"clusters_single_{column}"] = int((shares == 1.0).sum())
                row[f"median_share_{column}"] = float(shares.median()) if len(shares) else np.nan
            rows.append(row)
    return pd.DataFrame(rows), labels_by_data


def role_checks(sales, features):
    """Scale, price adjustment and storey checks; plus the median features of the K=4 clusters."""
    X = standardize(features)
    year = sales["month"].str[:4]
    nominal = features.assign(real_price=sales["resale_price"].astype(float).to_numpy())
    raw_labels = kmeans_labels(features.to_numpy(), 6)
    price_ranges = features.groupby(raw_labels)["real_price"].agg(["min", "max"]).sort_values("min")
    overlaps = int((price_ranges["min"].to_numpy()[1:] <= price_ranges["max"].to_numpy()[:-1]).sum())
    rows = [
        {"check": "scale", "setting": "raw features, K=6", "metric": "ari_flat_type", "value": adjusted_rand_score(sales["flat_type"], raw_labels)},
        {"check": "scale", "setting": "standardized, K=6", "metric": "ari_flat_type", "value": adjusted_rand_score(sales["flat_type"], kmeans_labels(X, 6))},
        {"check": "scale", "setting": "raw features, K=6", "metric": "overlapping_price_ranges", "value": overlaps},
        {"check": "price_adjustment", "setting": "nominal price, K=10", "metric": "ari_sale_year", "value": adjusted_rand_score(year, kmeans_labels(standardize(nominal), 10))},
        {"check": "price_adjustment", "setting": "adjusted price, K=10", "metric": "ari_sale_year", "value": adjusted_rand_score(year, kmeans_labels(X, 10))},
    ]
    without_storey = standardize(features.drop(columns="storey_mid"))
    for name, data in [("real", without_storey), ("shuffled null", shuffled_null(without_storey, np.random.default_rng(SEED)))]:
        counts = cluster_counts(methods.hdbscan(MIN_CLUSTER_SIZES[0]).fit_predict(data))
        rows += [{"check": "hdbscan_without_storey", "setting": name, "metric": metric, "value": value} for metric, value in counts.items()]

    profiles = (
        features.assign(cluster=kmeans_labels(X, 4), flat_type=sales["flat_type"].to_numpy())
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
    hdb = load_hdb_resale(n_samples=SAMPLE_SIZE, random_state=SEED, jitter=False)
    sales, features, X = hdb.sales, hdb.features_raw, hdb.data
    print(f"{len(hdb.all_sales)} transactions; screening sample of {len(X)}")
    out = results_dir(TOPIC)

    silhouette = silhouette_null_test(X, seed=SEED)
    density = hdbscan_null_test(X, MIN_CLUSTER_SIZES, seed=SEED)
    pd.concat(
        [silhouette.assign(test="silhouette_vs_shuffled"), density.assign(test="hdbscan_vs_shuffled")], ignore_index=True
    ).assign(sample_size=SAMPLE_SIZE, seed=SEED).to_csv(out / "screening.csv", index=False)

    labels = label_agreement(sales, X)
    labels.to_csv(out / "screening_labels.csv", index=False)
    discrete, labels_by_data = discrete_value_check(sales, features.to_numpy(), np.random.default_rng(SEED))
    discrete.to_csv(out / "screening_discrete.csv", index=False)
    roles, profiles = role_checks(sales, features)
    roles.to_csv(out / "screening_roles.csv", index=False)
    profiles.to_csv(out / "screening_kmeans_profiles.csv")

    null = shuffled_null(X, np.random.default_rng(SEED))
    pca = PCA(n_components=2).fit(X)
    panels = [
        (pca.transform(X), labels_by_data[("real", MIN_CLUSTER_SIZES[0])], "Real"),
        (pca.transform(null), methods.hdbscan(MIN_CLUSTER_SIZES[0]).fit_predict(null), "Shuffled null"),
    ]
    fig = gf.plot_screening(
        panels, xlabel="PC1", ylabel="PC2", silhouette=silhouette,
        title=f"HDB resale screening, {SAMPLE_SIZE:,} sales (HDBSCAN min_cluster_size={MIN_CLUSTER_SIZES[0]}, PCA view)",
    )
    gf.save(fig, TOPIC, "screening")

    with pd.option_context("display.width", 200):
        for table in [silhouette, density, labels, roles]:
            print(table.round(3).to_string(index=False))
        print(discrete.round(3).T.to_string())
        print(profiles.round(1).to_string())


if __name__ == "__main__":
    main()
