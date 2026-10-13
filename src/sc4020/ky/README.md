# Clustering for Spatial Hotspot Detection — K-Means vs DBSCAN family

## Reproduce
```bash
pip install scikit-learn pandas scipy matplotlib
# datasets: data/earthquakes.csv, data/uber-raw-data-apr14.csv, data/ecoli.arff
# (download commands in data/README.md; run everything from the repo root)

python -m sc4020.ky.run && python -m sc4020.ky.figures
cd docs/ky && pdflatex report.tex && pdflatex report.tex
```

## Files

Paths are from the repo root; `src/` is `src/sc4020/ky/`.

| path | what |
|---|---|
| `src/data.py`    | loaders, equirectangular projection to km |
| `src/core.py`    | algorithm wrappers, metrics, k-distance knee detector |
| `src/run.py`     | all experiments -> results/ky/*.csv |
| `src/figures.py` | all 10 figures -> results/ky/figures/ |
| `results/ky/main.csv` | main comparison, 3 datasets x 4 methods |
| `results/ky/sweep.csv` | every parameter-grid run |
| `results/ky/ablation.csv` | 4 ablations |
| `docs/ky/report.tex` | 13-page report, 11pt |

## Datasets
- **D1 Earthquakes** 23,412 x 2 — global M>=5.5, 1965-2016. No labels.
- **D2 UberNYC** 6,000 x 2 — NYC pickups April 2014. No labels.
- **D3 Ecoli** 336 x 7, 8 classes — UCI, Horton & Nakai 1996.

## Headline results
- Silhouette ranks the 4 methods in reverse of ARI on D3 (rank corr **-0.80**).
- DBSCAN at the k-distance knee traces plate boundaries with 2% noise but scores
  silhouette **-0.03**; the silhouette optimum discards 29% of events.
- Dimensionality crossover: DBSCAN wins at 2-3D, K-Means++ wins at 7D.
- k-means++ seeding is worth **0.24 ARI** on D3 at identical k.
