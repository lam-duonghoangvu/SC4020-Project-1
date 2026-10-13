# Cross-dataset comparison: K-Means, DBSCAN, HDBSCAN, GMM on every dataset (archived)

These notes record the protocol and findings of the earlier all-datasets run (`team_comparison`), which
ran the four methods on all 10 report datasets with the density methods capped at `max(k_range)`
clusters. That run has been removed: each dataset is now run by its own notebook in `experiments/`
(see the table in `README.md`), with numbers in `results/datasets/`, `results/earthquakes_2023/`,
`results/hdb_resale/` and `results/hotspot/`. The Wine and Breast Cancer rows of the report's shared
table come from notebooks 01 and 02 (`results/comparison/report_d6_d7.csv`).

## Datasets

Each is loaded by `sc4020.datasets.load_<name>()` with the defaults below.

| Dataset | Points x features | Labels | Preprocessing |
|---|---|---|---|
| wine | 178 x 13 | 3 cultivars | standardized |
| breast_cancer | 569 x 30 | benign / malignant | standardized |
| uber_pickups | 15,000 x 2 | none | Apr 2014 NYC pickups sample, projected to km |
| make_moons | 1,000 x 2 | 2 moons | noise 0.07, standardized |
| mall_customers | 200 x 3 | none | age, income, spending score, standardized |
| fashion_mnist | 3,000 x 50 | 10 classes | stratified sample, pixels/255, PCA to 50 |
| earthquakes_1965_2016 | 23,412 x 3 | none | M >= 5.5; 3D unit vectors |
| ecoli | 336 x 7 | 8 sites | standardized |
| earthquakes_2023 | 7,643 x 3 | none | USGS M >= 4.5 in 2023; 3D unit vectors |
| hdb_resale | 20,000 x 4 | none | area, lease, storey, index-adjusted price; ties jittered; standardized |

Two representation choices matter:
- **Earthquakes (both catalogues):** clustered as 3D unit vectors, so all four
  methods share one valid distance. A km projection breaks at +/-180 longitude.
- **HDB resale:** the jittered features are used for all four methods, not only
  the density ones. The HDB EDA notebook shows the untouched ties give one density
  cluster per storey band; K-Means is barely affected.

## Protocol (`src/sc4020/tuning.py`)

Settings are chosen without labels, by the shared rule of the report (Section 3.2, "Selection rule").
Labels are used only to compute ARI afterwards.

| Method | Parameter search | Criterion |
|---|---|---|
| K-Means | k in 2-10 | silhouette |
| GMM | k in 2-10 x covariance type (full / tied / diag / spherical) | BIC |
| DBSCAN | eps at 0.5-4x the k-distance knee x 4 values of `min_samples` | highest silhouette with >= 2 clusters and <= 50% noise |
| HDBSCAN | `min_cluster_size` in 5-400 x `min_samples` in {default, 5, 10} | highest silhouette with >= 2 clusters and <= 50% noise |

- **Known k:** on labelled datasets, K-Means and GMM are given the true k. What
  they would pick on their own is in `label_free_choice`; the best-ARI setting is in
  `best_ari_choice`.
- **DBCV** (Moulavi et al. 2014, `src/sc4020/metrics.py`) is reported for every result as a
  density-aware check on the silhouette, but no longer chooses settings: one rule chooses on every
  dataset. The implementation matches the `hdbscan` package's `validity_index` on 5 test cases.
- **Same cluster budget:** every method may use at most 10 clusters (50 for the
  earthquake catalogues). `tuning_valid = False` means no setting passed, and the least-noisy
  setting is shown.
- **Metrics** (`sc4020.metrics.evaluate`):
  - **ARI, NMI, AMI, Fowlkes-Mallows:** only on labelled datasets; noise counts as its own group.
  - **Silhouette, Davies-Bouldin, Calinski-Harabasz:** on non-noise points only.
  - **DBCV:** on all points, with noise counting against the score.
  - **Sampling:** silhouette is estimated on 5,000 points and DBCV on 3,000 when n is larger.

## Results

> These tables come from the run before the code was merged, when density settings were ranked
> by DBCV. Current per-dataset numbers are in the `results/` files named above.

### Labelled datasets: ARI (1 = perfect match with the true classes)

| Dataset | K-Means | DBSCAN | HDBSCAN | GMM | Best |
|---|---|---|---|---|---|
| wine | 0.897 | 0.296 | 0.342 | **0.915** | GMM |
| breast_cancer | 0.654 | 0.222* | 0.156* | **0.774** | GMM |
| make_moons | 0.481 | 0.916 | **0.947** | 0.501 | HDBSCAN |
| fashion_mnist | 0.343 | 0.017 | 0.023 | **0.389** | GMM |
| ecoli | 0.508 | 0.395 | 0.038 | **0.618** | GMM |

\* No DBSCAN or HDBSCAN setting found 2+ clusters with at most 50% noise; the
least-noisy setting is shown (55-59% noise).

### Unlabelled datasets: silhouette / DBCV (clusters found, noise %)

There is no ground truth here, so two label-free scores are shown. Silhouette
favours round, compact clusters; DBCV favours dense regions separated by sparse
gaps. Where both agree, the result is more trustworthy.

| Dataset | K-Means | DBSCAN | HDBSCAN | GMM |
|---|---|---|---|---|
| uber_nyc | **0.741** / -0.769 (2, 0%) | 0.636 / 0.264 (5, 1%) | 0.640 / **0.404** (3, 5%) | 0.362 / -0.648 (9, 0%) |
| mall_customers | 0.428 / 0.003 (6, 0%) | 0.277 / 0.201 (3, 23%) | **0.557** / **0.284** (6, 38%) | 0.343 / -0.091 (5, 0%) |
| earthquakes_1965_2016 | 0.528 / -0.500 (40, 0%) | -0.017 / 0.046 (2, 0%) | **0.615** / **0.507** (16, 28%) | 0.478 / -0.277 (50, 0%) |
| earthquakes_2023 | 0.580 / -0.377 (40, 0%) | 0.280 / 0.369 (16, 12%) | **0.653** / **0.437** (38, 22%) | 0.550 / -0.219 (50, 0%) |
| hdb_resale | **0.319** / -0.757 (4, 0%) | 0.123 / -0.334 (3, 7%) | 0.157 / **-0.101** (2, 30%) | 0.171 / -0.574 (10, 0%) |

### Fit time (seconds, final model only)

| Dataset | n | K-Means | DBSCAN | HDBSCAN | GMM |
|---|---|---|---|---|---|
| earthquakes_1965_2016 | 23,412 | 0.35 | 0.80 | 4.39 | 9.59 |
| hdb_resale | 20,000 | 0.09 | 0.79 | 3.27 | 1.74 |
| uber_nyc | 15,000 | 0.04 | 0.63 | 1.49 | 0.92 |
| earthquakes_2023 | 7,643 | 0.18 | 0.08 | 0.34 | 3.01 |
| fashion_mnist | 3,000 | 0.10 | 0.02 | 0.42 | 2.48 |

The small datasets all fit in under 0.3 s.

## What the results say

- **GMM wins most labelled real-world datasets:** wine, breast_cancer,
  fashion_mnist and ecoli. Real classes are rarely perfect spheres, and GMM's
  covariance lets clusters stretch and differ in size. K-Means is a close second
  on all four.
- **K-Means never wins a labelled dataset, but is never far behind GMM** (within
  0.02-0.12 ARI) and is the fastest method on every dataset.
- **Density methods win when cluster shape matters** (make_moons). K-Means and
  GMM cut each moon in half even with the right k. HDBSCAN edges DBSCAN because
  it needs no single global radius.
- **Density methods fail in higher dimensions.** On wine, breast_cancer,
  fashion_mnist and ecoli (7-50 features), DBSCAN and HDBSCAN find only 2
  clusters or discard a third or more of the points. Distances in many
  dimensions carry little density contrast.
- **HDBSCAN wins the spatial and customer data:** both earthquake catalogues and
  mall_customers are best on silhouette and DBCV together, and uber_nyc is best
  on DBCV. It follows the plate boundaries and flags the scattered events as
  noise.
- **DBSCAN's single radius fails on global earthquakes.** Every setting within
  the budget scores a negative DBCV, so the chosen one merges almost everything
  into 2 clusters.
- **HDB resale has no density structure.** Every method has a negative DBCV,
  which matches the HDB notebooks' finding of weak, partition-like structure. K-Means
  (k=4, the same k the HDB clustering notebook chose) has the best silhouette.
- **Ecoli has no density-separated classes.** HDBSCAN finds only 2 clusters at
  every setting it tried, and the best DBCV splits off a small isolated group
  (ARI 0.04).

## Caveats

- **K-Means and GMM get the true k on labelled data; the density methods do
  not.** On their own, K-Means would pick k=10 on make_moons and k=3 on
  fashion_mnist (see `label_free_choice`), so their labelled results are an
  upper bound.
- **Silhouette and DBCV each favour one family.** For unlabelled data, read both
  scores, the noise %, and the maps.
- **GMM hits the 50-component limit on both earthquake catalogues.** BIC keeps
  improving, so GMM has no natural stopping point there.
