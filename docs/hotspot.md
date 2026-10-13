# Hotspot study: K-Means vs the DBSCAN family on Earthquakes 1965-2016, Uber NYC and Ecoli

## Reproduce

From the repo root (datasets download into `data/` on first use; see `data/README.md`):

```bash
uv run python -m sc4020.experiments.hotspot        # sweeps, main comparison, ablations, fake-data check
uv run python -m sc4020.experiments.shared_rule    # shared-table rows for Uber and Ecoli
uv run python -m sc4020.experiments.case_stats     # numbers behind the Uber and Ecoli paragraphs
uv run python -m sc4020.generate_figures hotspot   # the 11 figures
cd docs/hotspot_draft_report && pdflatex report.tex && pdflatex report.tex
```

## Files

| Path | What |
|---|---|
| `src/sc4020/experiments/hotspot.py` | all experiments -> `results/hotspot/*.csv`, `chosen_params.json`, `kdist_curves.json` |
| `src/sc4020/experiments/shared_rule.py` | `results/hotspot/shared_rule.csv` |
| `src/sc4020/experiments/case_stats.py` | `results/hotspot/case_stats.csv` |
| `src/sc4020/generate_figures.py` | `fig_*` functions -> `results/hotspot/figures/f1..f10*.png` |
| `results/hotspot/main.csv` | main comparison, 3 datasets x 4 methods |
| `results/hotspot/sweep.csv` | every parameter-grid run |
| `results/hotspot/ablation.csv` | 4 ablations (initialisation, distance metric, scaling, dimensionality) |
| `notebooks/hotspot_study.ipynb` | earlier exploratory version on Ridgecrest 2019, Chicago thefts, Ecoli and make_moons |
| `docs/hotspot_draft_report/report.tex` | individual 13-page draft report, 11pt |

## Datasets

- **D1 Earthquakes** 23,412 events, global M>=5.5, 1965-2016, as 3D unit vectors. No labels.
- **D2 UberNYC** 15,000 pickups in NYC, April 2014, projected to km. No labels.
- **D3 Ecoli** 336 x 7, 8 classes, UCI (Horton & Nakai 1996), standardized.

## Selection

K-Means k is the number of classes on Ecoli and the silhouette's k on the coordinates (the elbow
is recorded too). DBSCAN and HDBSCAN use the shared rule (`tuning.select_dbscan` /
`select_hdbscan`): the highest silhouette among settings with at least 2 clusters and at most 50%
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
