# Methods (auto-generated)
- Stimuli: templated sentences with a single `{w}` slot; discovery vs eval families are disjoint. Optional label permutation and unordered concepts for null controls.
- Delta pairs: adjacent ordinal steps within the same template; strategies: cartesian (all synonym pairs) or random (one pair per edge).
- Activations: last-token pooled residual stream captured per transformer block via nnsight.
- PCA: per-layer PCA on Δh; record PC1 explained variance and k90 (PCs to reach 90% variance). PC1 sign anchored so projection correlates positively with ordinal index.
- Rotation: principal angles between top-k PCA subspaces of adjacent layers (k=min(10,k90) or fixed).
- Random/null baselines: random directions/subspaces; label permutation; unordered concept mode.
- Ablation: neurons selected on discovery only; selectors: lookahead, local_corr, local_ridge; controls: random and anti-selected neurons. Effect measured on eval as change in |projection| along concept direction.
