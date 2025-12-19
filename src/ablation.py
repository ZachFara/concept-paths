from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.decomposition import PCA

from .adapters import ModelAdapter
from .data import Sample
from .plots import plot_curves_with_ci
from .selection import select_neurons
from .utils import ensure_dir, save_json


def concept_direction_pca(deltas: np.ndarray, labels: np.ndarray, layer: int) -> np.ndarray:
    x = deltas[:, layer, :]
    pca = PCA(n_components=1, svd_solver="full")
    pca.fit(x)
    comp = pca.components_[0]
    if labels.shape[0] == x.shape[0]:
        proj = x @ comp
        corr = np.corrcoef(proj.flatten(), labels.flatten())[0, 1]
        if corr < 0:
            comp = -comp
    return comp / (np.linalg.norm(comp) + 1e-8)


def projection_per_pair(deltas: np.ndarray, direction: np.ndarray) -> np.ndarray:
    proj = deltas @ direction
    return np.abs(proj)


def bootstrap_ci(arr: np.ndarray, n_boot: int = 200) -> dict:
    rng = np.random.default_rng(0)
    samples = []
    n = arr.shape[0]
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        samples.append(arr[idx].mean())
    samp = np.array(samples)
    return {"mean": arr.mean(), "ci_low": np.percentile(samp, 2.5), "ci_high": np.percentile(samp, 97.5)}


def run_ablation_once(
    *,
    adapter: ModelAdapter,
    samples: List[Sample],
    mlp: np.ndarray,  # [N, L, D]
    residual: np.ndarray,  # [N, L, H]
    layer: int,
    m: int,
    selection_method: str,
    seed: int,
) -> dict:
    pairs = []
    by_template: Dict[int, Dict[int, List[int]]] = {}
    for idx, s in enumerate(samples):
        by_template.setdefault(s.template_id, {}).setdefault(s.level, []).append(idx)
    for template_id, levels in by_template.items():
        max_level = max(levels.keys())
        for lvl in range(max_level):
            if lvl not in levels or (lvl + 1) not in levels:
                continue
            for i in levels[lvl]:
                for j in levels[lvl + 1]:
                    pairs.append((i, j, lvl + 1))
    if not pairs:
        raise ValueError("No delta pairs constructed for ablation")
    neg_idx = np.array([i for i, _, _ in pairs], dtype=np.int64)
    pos_idx = np.array([j for _, j, _ in pairs], dtype=np.int64)
    pair_labels = np.array([lbl for _, _, lbl in pairs], dtype=np.float32)

    deltas = (residual[pos_idx] - residual[neg_idx]).astype(np.float32)
    direction = concept_direction_pca(deltas, pair_labels, layer)

    idx = select_neurons(
        method=selection_method,
        mlp_acts=mlp,
        labels=np.array([s.level for s in samples], dtype=np.float32),
        layer=layer,
        m=m,
        seed=seed,
    )

    prompts = [s.prompt_text for s in samples]
    baseline_resid = adapter.capture(prompts, batch_size=4).residual
    ablated_resid = adapter.capture(
        prompts, batch_size=4, ablate_layer=layer, neuron_idx=idx
    ).residual
    rng = np.random.default_rng(seed)
    d_mlp = mlp.shape[2]
    rand_idx = rng.choice(d_mlp, size=len(idx), replace=False)
    rand_resid = adapter.capture(
        prompts, batch_size=4, ablate_layer=layer, neuron_idx=rand_idx
    ).residual

    baseline_deltas = (baseline_resid[pos_idx] - baseline_resid[neg_idx]).astype(np.float32)
    ablated_deltas = (ablated_resid[pos_idx] - ablated_resid[neg_idx]).astype(np.float32)
    rand_deltas = (rand_resid[pos_idx] - rand_resid[neg_idx]).astype(np.float32)

    base_proj = projection_per_pair(baseline_deltas[:, layer, :], direction)
    abl_proj = projection_per_pair(ablated_deltas[:, layer, :], direction)
    rand_proj = projection_per_pair(rand_deltas[:, layer, :], direction)

    drop_target = base_proj - abl_proj
    drop_random = base_proj - rand_proj
    paired = drop_target - drop_random
    stats = bootstrap_ci(paired, n_boot=200)
    sign_flip_p = (np.sum(paired <= 0) + 1) / (paired.size + 1)
    return {
        "indices": idx.tolist(),
        "drop_target": drop_target,
        "drop_random": drop_random,
        "paired": paired,
        "stats": stats,
        "sign_flip_p": float(sign_flip_p),
    }


def dose_response(
    *,
    adapter: ModelAdapter,
    samples: List[Sample],
    mlp: np.ndarray,
    residual: np.ndarray,
    layer: int,
    m_list: Sequence[int],
    selection_method: str,
    seed: int,
) -> dict:
    effects = []
    ci_low = []
    ci_high = []
    for m in m_list:
        res = run_ablation_once(
            adapter=adapter,
            samples=samples,
            mlp=mlp,
            residual=residual,
            layer=layer,
            m=m,
            selection_method=selection_method,
            seed=seed,
        )
        paired = res["paired"]
        effects.append(paired.mean())
        ci = bootstrap_ci(paired, n_boot=200)
        ci_low.append(ci["ci_low"])
        ci_high.append(ci["ci_high"])
    auc = float(np.trapz(effects, x=list(m_list)))
    return {
        "effects": effects,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "auc": auc,
    }


def run_ablation_pipeline(
    *,
    adapter: ModelAdapter,
    samples: List[Sample],
    mlp: np.ndarray,
    residual: np.ndarray,
    layer: int,
    m_list: Sequence[int],
    methods: Sequence[str],
    seed: int,
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    stats_all: Dict[str, dict] = {}
    for method in methods:
        dose = dose_response(
            adapter=adapter,
            samples=samples,
            mlp=mlp,
            residual=residual,
            layer=layer,
            m_list=m_list,
            selection_method=method,
            seed=seed,
        )
        stats_all[method] = dose
        plot_curves_with_ci(
            x=np.array(m_list),
            mean=np.array(dose["effects"]),
            ci_low=np.array(dose["ci_low"]),
            ci_high=np.array(dose["ci_high"]),
            label=method,
            title=f"Dose response {method}",
            ylabel="paired effect",
            outpath=out_dir / f"dose_{method}.png",
            raw_out=out_dir / f"dose_{method}.npz",
        )
    def _to_jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_jsonable(v) for v in obj]
        return obj

    save_json(out_dir / "ablation_stats.json", _to_jsonable(stats_all))
