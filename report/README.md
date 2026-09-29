# Group report (SC4020 Project 1)

Skeleton for `group_XX_report.pdf`. Each section is its own file, so members can edit in parallel.

## Build
    cd report
    pdflatex main.tex && pdflatex main.tex          # report
    pdflatex contribution.tex                       # group_XX_contribution.pdf

Before submitting, set `\draftfalse` in `main.tex`: owners, TODOs and DECIDE notes disappear.

## Who owns what
| File | Section | Owner |
|---|---|---|
| 00-abstract.tex | Abstract | Ky (write last) |
| 01-introduction.tex | 1 Introduction, related work | Ky |
| 02-methods.tex | 2 Methods | Ky, Lam, Hoang Anh |
| 03a-datasets.tex | 3.1 Datasets, preprocessing | each dataset owner; Wei Yew |
| 03b-protocol.tex | 3.2 Metrics, parameter selection | Ky, Wei Yew |
| 03c-comparison.tex | 3.3 Comparison by data property | All |
| 03d-ablations.tex | 3.4 Ablations | Ky, Wei Yew |
| 03e-improvements.tex | 3.5 Improvements | All |
| 03f-runtime.tex | 3.6 Runtime | Wei Yew, Lam |
| 03g-cases.tex | 3.7 Success and failure cases | each owner |
| 03h-factors.tex | 3.8 Key factors, guidance | Ky |
| 04-conclusion.tex | 4 Conclusion, limitations | Ky |
| references.tex | References | shared |
| contribution.tex | Contribution PDF | All |

## Rubric map
- Methods explained clearly: 2
- Parameter settings: 3.2
- Ablation of key components: 3.4
- Key factors: 3.8
- Comparisons, strengths and weaknesses: 3.3
- Success and failure cases, visualisations: 3.7
- Challenging datasets: 3.1 (D2, D5, D8)
- Solutions to improve the methods: 3.5

## Rules
- Write in your own file. To comment on someone else's, add `% [Name]: note`.
- Every number comes from a results CSV in the repo; put its path in a `%` comment.
- Label prefixes: `sec:`, `tab:`, `fig:`. New figures go in `report/figures/`.
- Macros: `\todo{Name}{task}`, `\decide{question}`, `\owner{Name}`, `\figtodo{Name}{what}`.

## Decide at the meeting
1. Final dataset list (Table 1); Ky's or Wei Yew's earthquake pipeline.
2. One parameter-selection rule for all methods (3.2).
3. Shared metric code (3.2).
4. Who assembles `group_XX_code.zip` (no data, no external links).
