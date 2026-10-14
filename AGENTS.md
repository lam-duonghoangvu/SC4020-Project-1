# AGENTS.md

Rules for any coding agent (and human) editing this repository. They are mandatory, not suggestions.

## Project layout

- Library code lives in `src/sc4020/` (one package, no per-member folders).
- Dataset loading and preprocessing: `src/sc4020/datasets/`. Per-dataset helpers: `src/sc4020/run.py`.
- Every experiment is a notebook in `experiments/` (flat, no subfolders), named
  `NN_<source>_<dataset>_<step>.ipynb`: `NN` is the run order, `<source>` where the data comes from
  (`scikit_learn`, `kaggle`, `usgs`, `data_gov`, `uci`). Notebooks that depend on each other get
  consecutive numbers; steps that run in sequence or share a purpose go in one notebook, not separate
  scripts. Notebooks import from `sc4020`; nothing in `sc4020` imports them.
- Every figure in a notebook is shown, then saved, through `show_and_save(fig, topic, name)` (to
  `results/<topic>/figures/`); never `fig.savefig` or `plt.show()`.
- In notebooks, import from `sc4020.generate_figures` by name, at the top of the cell that uses it, and only
  what that cell uses (`from sc4020.generate_figures import plot_k_distance, show_and_save`); no `gf` alias.
- Generated outputs in `results/` and `report/figures/`, tests in `tests/`.
- Python `>=3.13`, managed with `uv`. Run tests with `uv run pytest`.

## Imports

- **Never write `from __future__ import annotations`** (or any other `from __future__` import). Python 3.13
  already supports every annotation syntax we use (`list[int]`, `X | None`, ...), so the import adds nothing.
  Remove it if you find it.
- Absolute imports only, rooted at the package: `from sc4020.metrics import NOISE_LABEL`, not `from .metrics ...`.
- Order: standard library, third party, then `sc4020`, separated by blank lines. No wildcard imports.
- Standard aliases only: `import numpy as np`, `import pandas as pd`, `import matplotlib.pyplot as plt`,
  `import seaborn as sns`. No other aliases.

## Naming convention

Names must say what the thing is. A reader should understand a name without reading its definition.

| Kind                                       | Style                                                         | Examples                                                                  |
| ------------------------------------------ | ------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Modules and packages                       | `snake_case`, a noun for what they hold                       | `clusters.py`, `preprocess_dataset.py`, `hdb_scalability.py`                |
| Functions                                  | `snake_case`, a verb phrase or the name of the value returned | `relabel_by_size`, `load_dataset`, `spherical_centre`, `cross_line_pairs` |
| Variables and parameters                   | `snake_case`, full words                                      | `labels`, `neighbours`, `max_km`, `min_cluster_size`                      |
| Constants (module level, never reassigned) | `UPPER_SNAKE_CASE`                                            | `NOISE_LABEL`, `EARTH_RADIUS_KM`, `K_RANGE`, `SEED`                       |
| Classes and dataclasses                    | `PascalCase` nouns                                            | `ClusterResult`, `Selection`                                              |
| Private helpers                            | leading underscore                                            | `_core_distances`                                                           |
| Booleans                                   | read as a yes/no question                                     | `is_noise`, `has_converged`, `both_clustered`                             |
| Collections                                | plural nouns                                                  | `labels`, `seeds`, `datasets`                                             |
| Mappings                                   | `<value>_by_<key>`                                            | `scores_by_method`, `labels_by_seed`                                      |

Further rules:

- **No unexplained abbreviations or single letters.** Write `distances`, not `d`; `n_clusters`, not `nc`;
  `dataset_name`, not `ds`. Allowed exceptions, because they are the standard notation in the course and in
  scikit-learn:
  - `X` for a feature matrix and `y` for target labels;
  - `k` (K-Means / k-NN), `eps`, `min_samples`, `min_cluster_size`, `n_` prefixes (`n_points`, `n_clusters`);
  - `i`, `j` as loop indices over short loops; `lat`, `lon`, `km`; `df` for a single obvious DataFrame;
  - metric names `ari`, `nmi`, `dbcv`, `bic`.
- Put the unit in the name when a quantity has one: `max_km`, `radius_rad`, `runtime_s`.
- Match scikit-learn parameter names exactly when wrapping or forwarding to it (`n_init`, `covariance_type`).
- Use British spelling, as the rest of the code does: `centre`, `neighbours`, `normalise`, `colour`.
- One name per concept across the whole package. Do not call the same thing `labels` in one module and
  `assignments` in another.
- Do not shadow built-ins (`id`, `type`, `list`, `max`, `input`, ...).
- DataFrame column names and dictionary keys follow the same `snake_case` rule (`"pairs_same_cluster"`).
- File names for outputs: `snake_case`, prefixed by what produced them (`wine_clusters.png`, `report_d6.csv`).

## Style

- Every public function gets a one-line docstring saying what it returns; add detail below it only if needed.
- Type hints are optional; when used, use built-in generics (`list[int]`, `dict[str, float]`, `X | None`),
  never `typing.List` / `Optional` (`typing.Any` is fine).
- Keep the comment density of the surrounding code: explain _why_, not _what_.
- Fix random seeds with the module-level `SEED` constant; never hard-code seeds inline.
