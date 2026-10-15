# Approximate seen- vs unseen-shape comparison from train_fit results (EXPLORATORY)

Sources: `docs/results/train-fit/train_fit_runs.csv`, `docs/results/train-fit-positions-all/train_fit_runs.csv`. Final models in eval mode; train = training part of the grouped split (5998 samples: seen shapes and seen samples), val = grouped validation (402 samples, unseen shapes). diff = mean - mean_B over seeds (percentage points); SE = Welch. B means: rel_all train 1.4359 / val 2.4416 %, rel_vy train 4.2349 / val 6.0674 %. **Limit:** the training set is not a true random validation (its samples were fitted); the night-6 grouped-holdout experiment is the clean version. 3-seed configurations: rough.

| config | seeds | rel_all train (seen shapes, seen samples): diff % (diff/SE) | rel_all val (unseen shapes): diff % (diff/SE) | rel_all z_train - z_val | rel_vy train (seen shapes, seen samples): diff % (diff/SE) | rel_vy val (unseen shapes): diff % (diff/SE) | rel_vy z_train - z_val |
|---|---|---|---|---|---|---|---|
| MD8 | 8/8 | -0.1442 (-4.00) | -0.0053 (-0.25) | -3.75 | -0.3999 (-4.87) | -0.2396 (-2.71) | -2.17 |
| SA8 | 8/8 | -0.0999 (-3.70) | +0.0624 (+3.80) | -7.50 | -0.2219 (-2.47) | +0.0988 (+1.05) | -3.52 |
| CBAM8 | 3/8 | -0.0844 (-2.86) | +0.0547 (+2.24) | -5.11 | -0.2186 (-1.55) | -0.0643 (-0.45) | -1.09 |
| B-large | 8/8 | -0.0124 (-0.55) | +0.0369 (+1.86) | -2.41 | -0.0336 (-0.36) | -0.0013 (-0.02) | -0.35 |
| MD32 | 8/8 | -0.0428 (-1.61) | +0.0252 (+1.76) | -3.37 | -0.1958 (-2.38) | -0.2195 (-2.56) | +0.17 |
| SA32 | 8/8 | -0.0798 (-2.86) | +0.0297 (+1.71) | -4.56 | -0.2301 (-2.24) | -0.1130 (-1.15) | -1.09 |
| MDv8 | 8/8 | -0.2216 (-10.49) | +0.0528 (+3.51) | -14.00 | -0.5127 (-4.83) | -0.0882 (-0.82) | -4.01 |
| MDv32 | 3/8 | -0.0944 (-3.72) | -0.0050 (-0.29) | -3.44 | -0.3328 (-2.74) | -0.3369 (-2.87) | +0.13 |
| MDv64 | 3/8 | -0.1500 (-4.89) | -0.0272 (-1.71) | -3.17 | -0.7571 (-10.86) | -0.7344 (-9.69) | -1.17 |
| MDv8-16-32 | 3/8 | -0.2954 (-14.56) | -0.0314 (-0.63) | -13.93 | -0.6921 (-7.53) | -0.4112 (-2.95) | -4.58 |

