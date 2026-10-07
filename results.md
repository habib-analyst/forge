# FORGE benchmark results

Tasks: 12 | Success rate: 83% | Mean iterations: 1.5

First-attempt success — early half: 100%, late half: 67% (Δ -33.3 pp — confounded by task difficulty; see PAPER.md §5)

FORGE ≥ human baseline on 5/12 tasks

| task | forge iters | 1st-try | forge metric | baseline | target |
|---|---|---|---|---|---|
| digits_8x8 | 1 | yes | 0.9639 | 0.9806 | 0.93 |
| shapes_cnn | 1 | yes | 1.0000 | 1.0000 | 0.95 |
| blobs_noisy | 1 | yes | 0.9786 | 1.0000 | 0.9 |
| iris_tabular | 1 | yes | 0.9333 | 0.9667 | 0.9 |
| wine_tabular | 1 | yes | 0.9722 | 0.9722 | 0.93 |
| breast_cancer | 1 | yes | 0.9825 | 0.9912 | 0.93 |
| diabetes_regression | 2 | no | — | 0.5353 | 0.55 |
| two_moons | 1 | yes | 0.8188 | 0.8500 | 0.78 |
| circles | 1 | yes | 0.9937 | 0.9937 | 0.9 |
| text_sentiment | 1 | yes | 1.0000 | 1.0000 | 0.93 |
| parity_sequence | 6 | no | — | 0.4800 | 0.7 |
| digits_imbalanced | 1 | yes | 0.9275 | 0.8986 | 0.72 |

## Failure taxonomy fix rates
{
  "nan_loss_divergence": 0.0,
  "below_target": 0.0
}