# TODO — Portfolio Readiness Fixes

## Blockers (fix before showing anyone)

- [ ] **Ablation plots are flat zero** — `artifacts/ablation/sentiment__eval/dose_probe.png` and the behavior ablation bar chart both render at zero. Diagnose why: the hook in `src/adapters/gpt2.py` zeros `c_fc` output but layer 0 may be too early for the effect to propagate through the residual stream. Try ablating at layers 3–5 instead of layer 0. Confirm the paired effect (`drop_target - drop_random`) is non-zero before plotting. (`src/ablation.py`, `src/behavior.py`, `src/adapters/gpt2.py`)

- [x] **`transfer_matrix` uses sample index as label** — `src/specificity.py:186` sets `labels = np.arange(proj_flat.shape[0])` instead of actual ordinal concept levels from the samples. The clean -0.75 off-diagonal in the cross-concept heatmap is an artifact of this. Fix: extract real levels from the sample list. (`src/specificity.py:186`)

- [x] **Behavior probe scores on training data** — `src/behavior.py:73-74` fits Ridge on `cache_disc.residual` and calls `.predict()` on the same split. The Spearman scores are in-sample. Fix: fit on discovery, score on `cache_eval.residual` / `labels_eval`. (`src/behavior.py:73-74`)

## Unfinished / Visible Loose Ends

- [ ] **`nnsight` dependency is declared but never used** — `pyproject.toml` lists `nnsight>=0.5.0` and a commit message says "About to change everything to nnsight," but all adapters use raw PyTorch hooks. Either complete the nnsight migration or remove the dependency and the dangling commit context. (`pyproject.toml`, `src/adapters/`)

- [ ] **`geometry.py` and `stimuli.py` are partially orphaned** — These form an older pipeline path superseded by `metrics.py` + `data.py`. `geometry_runner.py` calls `metrics.py`, not `geometry.py`, but `tests/test_geometry.py` still tests `geometry.py` directly. Decide: delete the old path and update tests, or explicitly document it as a standalone math module. (`src/geometry.py`, `src/stimuli.py`, `tests/test_geometry.py`)

- [x] **`cross_generalization` function is unused and has a shape bug** — Defined in `src/specificity.py:128-138` but never called from `run_specificity`. Also has a broken indexing guard (`proj.ndim==3` check using `dir_a.shape[0]` which is the hidden dim, not layers). Remove or fix and wire up. (`src/specificity.py:128-138`)

- [x] **Duplicate `from __future__ import annotations`** — Lines 1 and 3 of `src/capture.py` both import it. Delete one. (`src/capture.py:1-3`)

## Lower Priority / Polish

- [ ] **README shows no results** — `README.md` only lists commands. Add a short results section with the key finding (k90 expansion, permutation null plot) so someone landing on the repo understands what it found, not just how to run it.

- [ ] **`activations.py` role is unclear** — An older capture path that duplicates some logic from `capture.py`. Either remove it or document why it exists alongside `capture.py`. (`src/activations.py`)

- [ ] **PC1 variance story needs a clearer framing** — ~20% EVR with wide CIs is not a clean result on its own. Either run on a larger model where the signal is stronger, or reframe the PC1 plot as motivation for the k90 expansion finding rather than a standalone claim.
