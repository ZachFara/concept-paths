from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from .capture import capture_activations
from .config import ControlSpec, DataSpec, ExperimentConfig
from .data import Sample, generate_samples, shuffle_labels_preserve_counts
from .metrics import compute_deltas
from .plots import plot_curves_with_ci, plot_heatmap, plot_null_histogram
from .utils import ensure_dir, save_json, runtime_metadata, write_manifest_file


def concept_directions(
    samples: List[Sample],
    residual: np.ndarray,
    *,
    k: int = 1,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    deltas = compute_deltas(samples, residual)
    dirs = []
    bases = []
    for l in range(deltas.shape[1]):
        x = deltas[:, l, :]
        pca = PCA(n_components=min(k, x.shape[0], x.shape[1]), svd_solver="full")
        pca.fit(x)
        comp = pca.components_[0]
        comp = comp / (np.linalg.norm(comp) + 1e-8)
        dirs.append(comp.astype(np.float32))
        bases.append(pca.components_.T.astype(np.float32))
    return np.stack(dirs), bases


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


def similarity_across_concepts(
    cfg: ExperimentConfig,
    *,
    concept_a: str,
    concept_b: str,
    split: str,
    adapter: str,
    model: str,
    artifacts_dir: Path,
    use_cache: bool,
) -> dict:
    ds_a = DataSpec(concept=concept_a, split=split, template_family=cfg.data.template_family, seed=cfg.data.seed, n_levels=cfg.data.n_levels, n_per_level=cfg.data.n_per_level)
    ds_b = DataSpec(concept=concept_b, split=split, template_family=cfg.data.template_family, seed=cfg.data.seed, n_levels=cfg.data.n_levels, n_per_level=cfg.data.n_per_level)
    ctrl = ControlSpec()
    samples_a, _ = generate_samples(cfg, data_spec=ds_a, control=ctrl)
    cache_a = capture_activations(
        config=cfg,
        data_spec=ds_a,
        control_spec=ctrl,
        adapter_name=adapter,
        model_name=model,
        batch_size=cfg.data.n_per_level,
        artifacts_dir=artifacts_dir,
        use_cache=use_cache,
        local_files_only=True,
    )
    samples_b, _ = generate_samples(cfg, data_spec=ds_b, control=ctrl)
    cache_b = capture_activations(
        config=cfg,
        data_spec=ds_b,
        control_spec=ctrl,
        adapter_name=adapter,
        model_name=model,
        batch_size=cfg.data.n_per_level,
        artifacts_dir=artifacts_dir,
        use_cache=use_cache,
        local_files_only=True,
    )
    dirs_a, bases_a = concept_directions(samples_a, cache_a.residual)
    dirs_b, bases_b = concept_directions(samples_b, cache_b.residual)
    cos = np.array([cosine_similarity(da, db) for da, db in zip(dirs_a, dirs_b)], dtype=np.float32)
    angles = []
    for ba, bb in zip(bases_a, bases_b):
        k = min(5, ba.shape[1], bb.shape[1])
        ang = subspace_angles(ba[:, :k], bb[:, :k])
        angles.append(float(np.mean(ang * 180 / np.pi)))
    angles = np.array(angles, dtype=np.float32)
    return {"cosine": cos, "angles": angles}


def permutation_similarity(
    samples_a: List[Sample],
    samples_b: List[Sample],
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    n_shuffles: int = 50,
) -> List[np.ndarray]:
    rng = np.random.default_rng(0)
    levels_b = [s.level for s in samples_b]
    null = []
    for _ in range(n_shuffles):
        shuffled = shuffle_labels_preserve_counts(levels_b, rng)
        shuffled_samples_b = []
        for s, new in zip(samples_b, shuffled, strict=True):
            shuffled_samples_b.append(
                Sample(
                    sample_id=s.sample_id,
                    concept_name=s.concept_name,
                    level=int(new),
                    level_label=str(new),
                    template_id=s.template_id,
                    template_text=s.template_text,
                    synonym=s.synonym,
                    prompt_text=s.prompt_text,
                    metadata=s.metadata,
                )
            )
        dirs_a, _ = concept_directions(samples_a, residual_a)
        dirs_b, _ = concept_directions(shuffled_samples_b, residual_b)
        cos = np.array([cosine_similarity(da, db) for da, db in zip(dirs_a, dirs_b)], dtype=np.float32)
        null.append(cos)
    return null


def cross_generalization(
    dir_a: np.ndarray,
    samples_b: List[Sample],
    residual_b: np.ndarray,
) -> float:
    deltas_b = compute_deltas(samples_b, residual_b)
    proj = deltas_b @ dir_a  # [P, H]?? wait deltas shape [P,L,H]
    proj_layer = proj[:, dir_a.shape[0]-1] if proj.ndim==3 else proj  # safeguard
    labels = np.array([s.level for s in samples_b[: proj_layer.shape[0]]], dtype=np.float32)
    r, _ = spearmanr(proj_layer.flatten(), labels.flatten())
    return float(r)


def transfer_matrix(
    cfg: ExperimentConfig,
    *,
    concepts: List[str],
    split: str,
    adapter: str,
    model: str,
    artifacts_dir: Path,
    use_cache: bool,
) -> np.ndarray:
    ctrl = ControlSpec()
    dirs = {}
    deltas = {}
    samples_map = {}
    residual_map = {}
    for c in concepts:
        ds = DataSpec(concept=c, split=split, template_family=cfg.data.template_family, seed=cfg.data.seed, n_levels=cfg.data.n_levels, n_per_level=cfg.data.n_per_level)
        samples, _ = generate_samples(cfg, data_spec=ds, control=ctrl)
        cache = capture_activations(
            config=cfg,
            data_spec=ds,
            control_spec=ctrl,
            adapter_name=adapter,
            model_name=model,
            batch_size=cfg.data.n_per_level,
            artifacts_dir=artifacts_dir,
            use_cache=use_cache,
            local_files_only=True,
        )
        residual_map[c] = cache.residual
        samples_map[c] = samples
        d, _ = concept_directions(samples, cache.residual)
        dirs[c] = d
        deltas[c] = compute_deltas(samples, cache.residual)
    mat = np.zeros((len(concepts), len(concepts)), dtype=np.float32)
    for i, ca in enumerate(concepts):
        for j, cb in enumerate(concepts):
            if i == j:
                mat[i, j] = 1.0
                continue
            # use layer-averaged direction and aligned labels per pair
            dir_avg = dirs[ca].mean(axis=0)
            deltas_b = deltas[cb]  # [P, L, H]
            proj = deltas_b @ dir_avg  # [P, H]
            proj_flat = proj.mean(axis=1)  # [P]
            labels = np.arange(proj_flat.shape[0], dtype=np.float32)
            r, _ = spearmanr(proj_flat, labels)
            mat[i, j] = float(r)
    return mat


def run_specificity(
    cfg: ExperimentConfig,
    *,
    concepts: List[str],
    split: str,
    adapter: str,
    model: str,
    artifacts_dir: Path,
    use_cache: bool,
) -> None:
    ensure_dir(artifacts_dir / "plots")
    ensure_dir(artifacts_dir / "raw")
    stats_dir = artifacts_dir / "stats"
    ensure_dir(stats_dir)

    if len(concepts) < 2:
        return
    sim = similarity_across_concepts(
        cfg,
        concept_a=concepts[0],
        concept_b=concepts[1],
        split=split,
        adapter=adapter,
        model=model,
        artifacts_dir=artifacts_dir,
        use_cache=use_cache,
    )
    x = np.arange(sim["cosine"].shape[0])
    plot_curves_with_ci(
        x=x,
        mean=sim["cosine"],
        ci_low=sim["cosine"],
        ci_high=sim["cosine"],
        label="cosine",
        title="Direction similarity",
        ylabel="cosine",
        outpath=artifacts_dir / "plots" / f"similarity_{concepts[0]}_{concepts[1]}.png",
        raw_out=artifacts_dir / "raw" / f"similarity_{concepts[0]}_{concepts[1]}.npz",
    )
    mat = transfer_matrix(
        cfg,
        concepts=concepts,
        split=split,
        adapter=adapter,
        model=model,
        artifacts_dir=artifacts_dir,
        use_cache=use_cache,
    )
    plot_heatmap(
        matrix=mat,
        x_labels=concepts,
        y_labels=concepts,
        title="Cross concept transfer",
        outpath=artifacts_dir / "plots" / f"transfer_{split}.png",
        raw_out=artifacts_dir / "raw" / f"transfer_{split}.npz",
        vmin=-1,
        vmax=1,
    )
    save_json(
        stats_dir / f"transfer_{split}.json",
        {"similarity": sim["cosine"].tolist(), "transfer": mat.tolist()},
    )
    manifest = runtime_metadata()
    manifest.update({"concepts": concepts, "split": split})
    write_manifest_file(artifacts_dir / "manifests" / f"specificity_{split}.json", manifest)
