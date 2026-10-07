# Team comparison: K-Means, DBSCAN, HDBSCAN, GMM on every team dataset

`run_team_comparison.py` runs the four methods on all 10 datasets the team uses,
with one shared tuning and evaluation protocol, so every number below can be
compared across datasets and members.

```bash
# from the hoanganh folder; about 12 minutes, mostly the two earthquake catalogues and HDB
PYTHONUTF8=1 python comparison/run_team_comparison.py
PYTHONUTF8=1 python comparison/run_team_comparison.py wine ecoli     # a subset
```

`PYTHONUTF8=1` is needed on Windows: weiyew's USGS downloader writes the
catalogue with the default encoding.

The data is downloaded on first use into `hoanganh/data/team/` (gitignored).
Kaggle datasets come from the kagglehub cache.

Outputs:
- `results/team_results.csv`: one row per dataset and method (params, metrics, runtimes)
- `results/team_tuning.csv`: every setting tried during tuning
- `figures/team/<dataset>.png`: cluster plots (PCA 2D, or maps for spatial data)

## Datasets

| Dataset | Owner | Points x features | Labels | Preprocessing (owner's unless noted) |
|---|---|---|---|---|
| wine | hoanganh, lam | 178 x 13 | 3 cultivars | standardized |
| breast_cancer | hoanganh | 569 x 30 | benign / malignant | standardized |
| uber_nyc | hoanganh, Ky | 15,000 x 2 | none | Apr 2014 NYC pickups sample, projected to km (Ky) |
| make_moons | lam | 1,000 x 2 | 2 moons | noise 0.07, standardized |
| mall_customers | lam | 200 x 3 | none | age, income, spending score, standardized |
| fashion_mnist | lam | 3,000 x 50 | 10 classes | stratified sample, pixels/255, PCA to 50 |
| earthquakes_1965_2016 | Ky | 23,412 x 3 | none | M >= 5.5; 3D unit vectors (weiyew's method) |
| ecoli | Ky | 336 x 7 | 8 sites | standardized |
| earthquakes_2023 | weiyew | 7,643 x 3 | none | USGS M >= 4.5 in 2023; 3D unit vectors |
| hdb_resale | weiyew | 20,000 x 4 | none | area, lease, storey, index-adjusted price; ties jittered; standardized |

Two deliberate changes from the owners' code:
- **Earthquakes (both catalogues):** clustered as 3D unit vectors, so all four
  methods share one valid distance. Ky's km projection breaks at +/-180 longitude.
- **HDB resale:** the jittered features are used for all four methods, not only
  the density ones. weiyew's EDA shows the untouched ties give one density
  cluster per storey band; K-Means is barely affected.

## Protocol (`src/sc4020/tuning.py`)

Settings are chosen without labels. Labels are used only to compute ARI afterwards.

| Method | Parameter search | Criterion |
|---|---|---|
| K-Means | k in 2-10 | silhouette |
| GMM | k in 2-10 x covariance type (full / tied / diag / spherical) | BIC |
| DBSCAN | eps at 0.5-4x the k-distance knee x 4 values of `min_samples` | DBCV |
| HDBSCAN | `min_cluster_size` in 5-400 x `min_samples` in {default, 5, 10} | DBCV |

- **Known k:** on labelled datasets, K-Means and GMM are given the true k. What
  they would pick on their own is in `label_free_choice`.
- **Why DBCV** (Moulavi et al. 2014, `src/sc4020/dbcv.py`): density methods are
  tuned with DBCV rather than silhouette, because silhouette assumes round
  clusters. The implementation matches the `hdbscan` package's `validity_index`
  on 5 test cases.
- **Same cluster budget:** every method may use at most 10 clusters (50 for the
  earthquake catalogues). A density setting with more than 50% noise is
  rejected. `tuning_valid = False` means nothing passed, and the least-noisy
  setting is shown.
- **Metrics:**
  - **ARI:** only on labelled datasets; noise counts as its own group.
  - **Silhouette and Davies-Bouldin:** on non-noise points only.
  - **DBCV:** on all points, with noise counting against the score.
  - **Sampling:** silhouette is estimated on 5,000 points and DBCV on 3,000 when n is larger.

## Results

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
  which matches weiyew's finding of weak, partition-like structure. K-Means
  (k=4, the same k weiyew chose) has the best silhouette.
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

## `run_four_models.py` (earlier showcase attempt, not a team comparison)

Runs one "showcase" dataset per method: digits for K-Means, make_moons for
DBSCAN, varied_density for HDBSCAN, breast_cancer for GMM. digits and
varied_density were added for that experiment and are not team datasets. With
the DBCV tuning above, only the K-Means and GMM showcases hold. Its outputs are
`results/four_models_*.csv`, `results/showcase_summary.csv` and `figures/*.png`.
