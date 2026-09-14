# Clustering for Spatial Hotspot Detection — K-Means vs DBSCAN family

## Reproduce
```bash
pip install scikit-learn pandas scipy matplotlib
# datasets
mkdir -p data
curl -o data/earthquakes.csv https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv
curl -o data/uber_apr14.csv  https://raw.githubusercontent.com/plotly/datasets/master/uber-rides-data1.csv
git clone --depth 1 https://github.com/deric/clustering-benchmark.git
cp clustering-benchmark/src/main/resources/datasets/real-world/ecoli.arff data/

cd src && python run.py && python figures.py
cd ../report && pdflatex report.tex && pdflatex report.tex
```

## Files
| path | what |
|---|---|
| `src/data.py`    | loaders, equirectangular projection to km |
| `src/core.py`    | algorithm wrappers, metrics, k-distance knee detector |
| `src/run.py`     | all experiments -> results/*.csv |
| `src/figures.py` | all 10 figures |
| `results/main.csv` | main comparison, 3 datasets x 4 methods |
| `results/sweep.csv` | every parameter-grid run |
| `results/ablation.csv` | 4 ablations |
| `report/report.tex` | 13-page report, 11pt |

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
