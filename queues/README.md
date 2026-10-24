# Run queues

Each file lists training runs for `scripts/run_queue.py`, one per line:

```
<run name>  est=<minutes>m  <scripts/train.py arguments>
```

`run_queue.py` runs them one at a time, evaluates each on the validation split right after training, skips runs
that are already finished, and stops before a run that would end after `--cutoff`:

```bash
python scripts/run_queue.py queues/<file>.txt --out-root ~/runs/<name> --cutoff YYYY-MM-DDTHH:MM
```

All runs use 133 epochs, batch 10, lr 6e-4 with the original decay and the fixed profile-disjoint split
(`splits/val_grouped_seed0.txt`) unless the line says otherwise. Run names in the files use the code names
(`MDv64` = variant `MDv` at level 64); the poster names are in the last column.

## Start here

| File | Runs | Purpose |
|---|---|---|
| `reproduce-final.txt` | B, MDv64-local, seeds 0-7 | Reproduce the final comparison (B vs the final architecture U-Net + M64-loc). About 4.2 h on an RTX 4060 (measured: B 12.5 min, MDv64-local 19 min per run; est= adds a margin). |

## Queues behind the results

These are the queues that produced the runs in the poster and report, kept as an exact record of the arguments.
Their header comments are historical: they refer to the launcher scripts and other queue files of the time, which
are no longer in the repository but can be found at the git tag `experiments-final`.

| File | Configurations [seeds] | Poster name | Results (`docs/results/`) |
|---|---|---|---|
| `night1.txt` | B at 133 epochs [0-2] (`ep133`), budget check at 67 epochs | B | `night1.csv`, `test`, `val-final` |
| `night2.txt` | MD8, SA8, CBAM8, MD32, SA32, B-large [0-2] | first block comparison; B-large | `test`, `val-final` |
| `night3.txt` | B, MD8, SA8, MD32, SA32, B-large [3-7] | B, B-large | `test`, `val-final` |
| `night4.txt` | MDv8 [0-7]; data-quantity runs (B, MD8, SA8 on 25 % / 50 % of the training set) [0-2] | M8 | `test-mdv8`, `val-mdv8`, `data-quantity` |
| `night5.txt` | MDv32, MDv64, MDv8-16-32 [0-2] | M32, M64 (Finding 2) | `val-positions-all` |
| `night6b.txt` | DRv11, TR, B (`GH_B`) [0-4]; MD8, MDv8-16-32 [0-2]; split `grouped-holdout` | DRv11, Truncated (Finding 1) | `test-confirm-dr`, `val-confirm-dr` |
| `night7.txt` | MDv64 [3-7] | M64 (Finding 2) | `test-confirm-mdv64`, `val-confirm-mdv64` |
| `night8a.txt` | MDv64-local, MDv64-dec, MDv16 [0-4] | M64-loc, M64-dec, M16 (Findings 2-3) | `val-ablation64-8seed` |
| `night8b.txt` | Conv64 [0-4] | Conv64 (Finding 3) | `val-ablation64-8seed` |
| `night9.txt` | one-ingredient ablations of MDv64-local (`L-*`), SA64 [0-4] | SA64 (Finding 3) | `val-ablation64-8seed` |
| `night10.txt` | MDv64-local [5-7] | M64-loc (final test) | `test-final`, `val-final-mdv64local` |
| `bigdata.txt` | B, MDv64-local, MDv64 [0-2] on the full 53.8k set (split `grouped-full`, cloud VM) | Finding 3, full set | `val-bigdata` |

The variants and their code names are described in `mdf/__init__.py`. The protocol (seeds, pre-registration,
single test pass) is described in the README and in `docs/lab-notebook.md`.
