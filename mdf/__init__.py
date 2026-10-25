"""mdf: Mamba-style blocks in the DfpNet U-Net for RANS airfoil flow prediction.

The task (Thuerey et al. 2020, train/DfpNet.py): a U-Net maps a 128x128 input (free-stream vx, free-stream vy,
airfoil mask) to the 128x128 RANS solution (pressure p, velocity vx, vy) in one forward pass. This package adds
one extra block, a "global mixer", at a chosen resolution of that U-Net, and provides everything needed to train
and compare the variants under a controlled protocol (fixed split, explicit seeds, paired comparisons).

Modules, one role each:
  models        the model factory: the baseline (the unmodified TurbNetG) and every variant, built from a small
                config dict; where in the U-Net each block is inserted
  blocks        the blocks themselves: Mamba / SS2D (two versions), self-attention, CBAM
  ablation      ablations of the 64x64 Mamba block: no scan (the final architecture), decoder placement, a plain
                conv block, one-ingredient removals
  deep_replace  the deep path of the U-Net (below 8x8) removed, or replaced by a stack of Mamba blocks
  mamba_ref     a pure-PyTorch selective scan (CPU, no compiled kernels): the same computation as mamba_ssm's
                CUDA kernels, used for tests and CPU inference
  data          loading, normalisation and the fixed train / validation splits
  explore       per-sample analysis: flow regions (wall band, wake), cached inference, paired bootstrap
  runinfo       run metadata: seeding, git commit, versions, timers, peak memory

Names. The poster and report use short names; the code (and `scripts/train.py --variant ... --level ...`) uses:

  report        poster      code (variant, level)   what it is
  B             B           B                       the original DfpNet (TurbNetG, 32 base ch., 2.33 M parameters)
  B-large       B-large     B-large                 DfpNet with 33 base channels: +parameters, no block (control)
  M^sh_8        -           MD, level 8             first Mamba version (one 1-D Mamba layer shared by 4 scans)
  M_8..M_64     M8..M64     MDv, level 8..64        one SS2D Mamba block (VMamba version) in the encoder
  M^dec_64      M64-dec     MDv-dec                 the same block on the decoder side at 64x64
  M^loc_64      M64-loc     MDv-local, level 64     M64 without the scan (only local mixing): FINAL ARCHITECTURE
  SA_64         SA64        SA, level 64            one self-attention block at 64x64
  Conv_64       Conv64      Conv, level 64          one plain residual conv block at 64x64 (control)
  TR            Truncated   TR                      DfpNet with the deep path (below 8x8) removed
  DR-M          DRv11       DRv11                   TR with 11 Mamba blocks at 8x8 in place of the deep path
  "5 variants"  -           L-nogate, L-nonorm, ... one-ingredient ablations of M64-loc
  -             -           CBAM, SA (8, 32), MD 32 earlier comparisons at 8x8 / 32x32 (lab notebook)

The blocks (all except CBAM) are residual, y = x + f(x), with the last projection of f zero-initialised: they
start as the identity, so at initialisation a one-block variant computes exactly the same function as B (and
DRv11 the same as TR). Training then decides how much the block contributes; see mdf/models.py.
"""
