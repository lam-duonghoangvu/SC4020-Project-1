# Per-dataset notebooks (`experiments/01` to `06`)

One notebook per dataset, with the same sections in each: dataset loading, one section per method
(K-Means with random starts, K-Means++, DBSCAN, HDBSCAN, GMM, then Spectral, Agglomerative and
Affinity Propagation), metrics, visualization. Each method section calls the step helpers in
`sc4020.run`: `select_method` picks the setting, `fit_method` fits it; the metrics section scores
every fit with `score_methods` and saves it with `save_dataset_run`.

| Notebook | Dataset |
| --- | --- |
| `01_scikit_learn_wine_clustering.ipynb` | `wine`, plus the report's shared table rows (`results/comparison/report_d6_d7.csv`) |
| `02_scikit_learn_breast_cancer_clustering.ipynb` | `breast_cancer`, plus the shared table rows and `report/figures/bc_cases.png` |
| `03_scikit_learn_make_moons_clustering.ipynb` | `make_moons` |
| `04_kaggle_mall_customers_clustering.ipynb` | `mall_customers` |
| `05_kaggle_fashion_mnist_clustering.ipynb` | `fashion_mnist` |
| `06_kaggle_uber_pickups_clustering.ipynb` | `uber_pickups` |

```bash
uv run jupyter nbconvert --to notebook --execute --inplace experiments/01_scikit_learn_wine_clustering.ipynb
```

`MAX_CLUSTERS` in each notebook's first cell optionally caps the number of DBSCAN and HDBSCAN
clusters. There is no cap by default. The cap matters: on make_moons, DBSCAN picks 72 clusters
(ARI 0.009) without it and 2 clusters (ARI 0.916) with a cap of 10.

Outputs, per dataset:

- `results/datasets/<dataset>_results.csv`: one row per method (parameters, every metric, DBCV, fit
  and tuning time, whether the shared rule found a valid setting, and the label-free and best-ARI
  choices).
- `results/datasets/<dataset>_tuning.csv`: every setting tried while tuning.
- `results/datasets/figures/<dataset>_{data,kmeans_k_selection,dbscan_k_distance,metrics,clusters}.png`:
  every figure the notebook shows; the cluster plot has the ground truth (if any) and one panel per
  method, as maps for coordinate datasets.

## How each method is set

| Method               | Setting                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------- |
| K-Means, K-Means (random init) | the true k on labelled data, else the k with the largest silhouette gap over fake data among seed-stable k (`tuning`) |
| GMM                  | k and covariance type by BIC (k fixed to the true k on labelled data)                         |
| DBSCAN, HDBSCAN      | the shared rule: best silhouette among settings with >= 2 clusters and <= 50% noise           |
| Spectral             | the K-Means k; RBF affinity on Wine and Mall, a nearest-neighbour graph elsewhere             |
| Agglomerative        | the K-Means k, Ward linkage                                                                   |
| Affinity Propagation | damping 0.9 and preference -150 on Breast Cancer, damping 0.8 on Wine, damping 0.9 elsewhere |

Spectral, Agglomerative and Affinity Propagation need memory quadratic in the number of points, so
they are skipped above 10,000 points (e.g. `uber_pickups`, 15,000 pickups). Silhouette and DBCV are
estimated on samples of 5,000 and 3,000 points for larger data.

The two earlier Wine notebooks became one. Spectral uses the RBF affinity there, as the report states
(ARI 0.445); the old model-zoo notebook used a nearest-neighbour graph (ARI 0.880).

## Notes carried over from the notebooks

### Breast Cancer Wisconsin (Diagnostic)

569 tumours, 30 standardized features, target 0 benign (357) and 1 malignant (212).

- **GMM (ARI 0.77) and Spectral (0.76) are best:** benign and malignant cells form two distributions
  with different spreads, which GMM's per-cluster covariance and the nearest-neighbour graph capture
  better than K-Means (0.65).
- **Agglomerative (Ward, 0.58):** minimizing within-cluster variance mirrors K-Means but is less
  accurate here.
- **DBSCAN in 30-D:** pairwise distances become sparse, so a single global eps either collapses
  everything into one cluster or discards the dispersed malignant points as noise.
- **Affinity Propagation:** without a fixed cluster count it finds many local sub-densities and
  over-fragments the two classes (16 clusters).

### Fashion-MNIST

A stratified sample of 3,000 images (balanced across the 10 classes), reduced to 50 PCA components;
results describe this sample, not the full 70,000 images.

- Visually similar classes (shirt/pullover/coat, sneaker/sandal/ankle boot) overlap heavily in
  pixel/PCA space, so ARI and NMI are much lower than on Wine or make_moons.
- DBSCAN and HDBSCAN struggle (distances concentrate even at 50 dimensions, and local density is
  uneven across similar classes); watch `noise_fraction` and `n_clusters` against the true 10.
- Spectral uses a nearest-neighbour graph: image embeddings are not well captured by one global RBF
  bandwidth.

### make_moons

1,000 points, two interleaving half-moons, `noise=0.07`: non-convex by design.

- **Spectral with a nearest-neighbour graph** follows the curved shape (ARI 1.000); HDBSCAN recovers
  most of it (0.709).
- **DBSCAN without a cluster cap** picks a small eps that splits the moons into 72 clusters (ARI 0.009);
  with `MAX_CLUSTERS = 10` it finds the two moons (0.916).
- **K-Means and GMM** assume roughly spherical clusters and cut across both moons with a straight
  boundary (ARI about 0.5).
- Internal metrics can favour K-Means and GMM here although their ARI is worse: the silhouette also
  assumes convex, well-separated clusters, so it rewards the convex but wrong answer.

### Mall Customers

200 shoppers; age, annual income and spending score, standardized. No labels, so only internal
metrics apply, and k comes from the silhouette (6).

- The well-known five or six segments (e.g. high income and low spending) are clearest in the raw
  income-versus-spending plane.
- Density methods may merge or split segments differently from the globular assumption of K-Means,
  GMM and Spectral.

### Uber pickups (NYC, April 2014)

15,000 pickups inside NYC, projected to km, so DBSCAN's eps is a distance in km. No labels.

- **DBSCAN** traces the elongated density corridors of Manhattan and isolates major hubs (JFK,
  LaGuardia) as separate components; dispersed pickups across Staten Island, Brooklyn and Queens fall
  into noise instead of pulling centroids.
- **K-Means and GMM** must assign every point, so isolated suburban trips get arbitrary radial
  boundaries and the Manhattan corridor is cut into rigid convex cells.

### Wine

178 wines, 13 standardized chemical measurements, 3 cultivars.

- **K-Means and GMM** match the cultivars closely (ARI 0.897 and 0.915): the classes are close to
  convex, similarly sized Gaussian blobs, which is what both methods assume.
- **Agglomerative (Ward)** reaches 0.790; **Spectral** with the RBF affinity 0.445, sensitive to the
  bandwidth.
- **DBSCAN and HDBSCAN** collapse to 2 clusters with much noise: there is no density gap between
  cultivars in 13 dimensions, so a single global eps merges neighbours and drops peripheral wines.
- **Affinity Propagation** splits the 3 cultivars into 15 micro-clusters.
- Knowing the true k gives K-Means, GMM and Spectral an advantage; the density methods had to find
  the cluster count themselves.
