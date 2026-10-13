# Hotspot study: K-Means vs the DBSCAN family on Earthquakes 1965-2016, Uber NYC and Ecoli

## Reproduce

From the repo root (datasets download into `data/` on first use; see `data/README.md`):

```bash
# sweeps, main comparison, ablations, fake-data check, shared-table rows, case statistics, figures
uv run jupyter nbconvert --to notebook --execute --inplace experiments/13_multi_source_hotspot_clustering.ipynb
cd docs/hotspot_draft_report && pdflatex report.tex && pdflatex report.tex
```

## Files

| Path | What |
|---|---|
| `experiments/13_multi_source_hotspot_clustering.ipynb` | every experiment, in order -> `results/hotspot/*.csv` (incl. `shared_rule.csv`, `case_stats.csv`), `chosen_params.json`, `kdist_curves.json`, and the figures |
| `src/sc4020/generate_figures.py` | `fig_*` functions, drawn by `hotspot_figures` -> `results/hotspot/figures/f1..f10*.png` |
| `results/hotspot/main.csv` | main comparison, 3 datasets x 4 methods |
| `results/hotspot/sweep.csv` | every parameter-grid run |
| `results/hotspot/ablation.csv` | 4 ablations (initialisation, distance metric, scaling, dimensionality) |
| `experiments/14_multi_source_hotspot_study.ipynb` | earlier exploratory version on Ridgecrest 2019, Chicago thefts, Ecoli and make_moons |
| `docs/hotspot_draft_report/report.tex` | individual 13-page draft report, 11pt |

## Datasets

- **D1 Earthquakes** 23,412 events, global M>=5.5, 1965-2016, as 3D unit vectors. No labels.
- **D2 UberNYC** 15,000 pickups in NYC, April 2014, projected to km. No labels.
- **D3 Ecoli** 336 x 7, 8 classes, UCI (Horton & Nakai 1996), standardized.

## Selection

K-Means k is the number of classes on Ecoli and the silhouette's k on the coordinates (the elbow
is recorded too). DBSCAN and HDBSCAN use the shared rule (`tuning.select_dbscan_by_shared_rule` /
`select_hdbscan_by_shared_rule`): the highest silhouette among settings with at least 2 clusters and at most 50%
noise, on the grids in the report (minPts {4, 8, 16, 32} or {3, 5, 8, 12}, eps at 0.1 to 3 times the
k-distance knee; min_cluster_size 10 to 250 or 5 to 30). On Ecoli the best-ARI setting is reported
next to the rule's choice.

## Headline results of the individual draft report

These come from the draft report's run, before the code was merged; `results/hotspot/` holds the
current numbers.
- Silhouette ranks the 4 methods in reverse of ARI on D3 (rank corr **-0.80**).
- DBSCAN at the k-distance knee traces plate boundaries with 2% noise but scores
  silhouette **-0.03**; the silhouette optimum discards 29% of events.
- Dimensionality crossover: DBSCAN wins at 2-3D, K-Means++ wins at 7D.
- k-means++ seeding is worth **0.24 ARI** on D3 at identical k.
