from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from nnsight import LanguageModel
from sklearn.decomposition import PCA
from tqdm import tqdm

from .capture import _get_blocks, ensure_scanned, resolve_device
from .data import DeltaPair
from .metrics import deltas_from_pairs
from .utils import atomic_save_npz, ensure_dir, get_device, maybe_load_npz


@dataclass(frozen=True)
class NeuronSelection:
    layer: int
    neuron_idx: np.ndarray  # [m]
    score: np.ndarray  # [m] (correlation)


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


def concept_direction_pc1(deltas: np.ndarray) -> np.ndarray:
    """
    Compute a concept direction as PC1 of Δh at a single layer.
    deltas: [n_pairs, hidden]
    returns: [hidden] unit vector
    """
    pca = PCA(n_components=1, svd_solver="full", whiten=False)
    pca.fit(deltas)
    return _unit(pca.components_[0].astype(np.float32))


@torch.no_grad()
def capture_last_token_mlp_act(
    lm: LanguageModel,
    prompts: list[str],
    *,
    layer: int,
    batch_size: int,
    device: torch.device | None = None,
    desc: str = "mlp_act",
) -> np.ndarray:
    """
    Capture GPT-2 style MLP activation (post nonlinearity) at a given block.
    Returns: [n_prompts, d_mlp] float32 numpy on CPU.

    Notes:
      - This is model-architecture dependent; it is expected to work for GPT-2 blocks.
    """
    if device is None:
        device = get_device()
    device = resolve_device(lm, device)

    tokenizer = lm.tokenizer
    if prompts:
        ensure_scanned(lm, prompts[0])
    blocks = _get_blocks(lm)
    block = blocks[layer]
    if not hasattr(block, "mlp") or not hasattr(block.mlp, "act"):
        raise ValueError("This ablation helper currently supports GPT-2 style blocks with .mlp.act")

    out_chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        attention_mask = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))
        last_idx = attention_mask.sum(dim=1) - 1
        bsz = enc["input_ids"].shape[0]
        batch_arange = torch.arange(bsz, device=device)

        with lm.trace(enc):
            # Activation module returns a Tensor (not a tuple), so use `.output` not `.output[0]`.
            a = block.mlp.act.output.save()  # [B, S, d_mlp]

        a_t = a.value if hasattr(a, "value") else a
        # Some nnsight backends return [S, B, D] instead of [B, S, D].
        if a_t.ndim == 3 and a_t.shape[0] != bsz and a_t.shape[1] == bsz:
            a_t = a_t.transpose(0, 1)
        pooled = a_t[batch_arange, last_idx].detach().to("cpu", dtype=torch.float32).numpy()
        out_chunks.append(pooled)

    return np.concatenate(out_chunks, axis=0)


def build_or_load_mlp_act_cache(
    lm: LanguageModel,
    *,
    keys: list[str],
    prompts: list[str],
    artifacts_dir: Path,
    split: str,
    model_name: str,
    seed: int,
    layer: int,
    batch_size: int,
    device: torch.device | None = None,
    force_recompute: bool = False,
) -> np.ndarray:
    safe_model = model_name.replace("/", "__")
    path = artifacts_dir / "mlp_act" / f"{split}__{safe_model}__seed{seed}__layer{layer}.npz"
    if not force_recompute:
        loaded = maybe_load_npz(path)
        if loaded is not None:
            loaded_keys = loaded["keys"].astype(str).tolist()
            if loaded_keys == keys:
                return loaded["act"].astype(np.float32)

    act = capture_last_token_mlp_act(
        lm,
        prompts,
        layer=layer,
        batch_size=batch_size,
        device=device,
        desc=f"mlp_act:{split}:L{layer}",
    )
    atomic_save_npz(path, keys=np.array(keys, dtype=str), act=act.astype(np.float32))
    return act.astype(np.float32)


def corr_with_scalar(x: np.ndarray, s: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute Pearson correlation between each column of x and scalar s.
    x: [n, d]
    s: [n]
    returns: [d]
    """
    x_c = x - x.mean(axis=0, keepdims=True)
    s_c = s - s.mean()
    num = x_c.T @ s_c
    denom = np.sqrt((x_c**2).sum(axis=0) * (s_c**2).sum() + eps)
    return (num / denom).astype(np.float32)


def select_top_neurons(
    *,
    deltas_resid: np.ndarray,
    deltas_resid_next: np.ndarray,
    deltas_mlp: np.ndarray,
    layer: int,
    top_m: int,
) -> NeuronSelection:
    """
    Select top neurons by |corr(Δmlp_neuron, proj(Δh_{l+1}, concept_dir_l))|.

    deltas_resid: [n_pairs, n_layers, hidden]
    deltas_resid_next: [n_pairs, hidden] is deltas_resid[:, layer+1, :]
    deltas_mlp: [n_pairs, d_mlp] is Δ at layer's MLP act.
    """
    concept_dir = concept_direction_pc1(deltas_resid[:, layer, :])
    proj = deltas_resid_next @ concept_dir  # [n_pairs]
    corr = corr_with_scalar(deltas_mlp, proj)
    order = np.argsort(-np.abs(corr))
    sel = order[:top_m]
    return NeuronSelection(layer=layer, neuron_idx=sel.astype(np.int64), score=corr[sel])


@torch.no_grad()
def capture_last_token_resid_layer_with_ablation(
    lm: LanguageModel,
    prompts: list[str],
    *,
    ablate_layer: int,
    neuron_idx: np.ndarray,
    capture_layer: int,
    batch_size: int,
    device: torch.device | None = None,
    desc: str = "ablate",
) -> np.ndarray:
    """
    Run the model while clamping selected MLP activation neurons to 0 at ablate_layer,
    and capture the pooled residual output at capture_layer.

    Returns: [n_prompts, hidden] float32 numpy on CPU.
    """
    if device is None:
        device = get_device()
    device = resolve_device(lm, device)

    tokenizer = lm.tokenizer
    if prompts:
        ensure_scanned(lm, prompts[0])
    blocks = _get_blocks(lm)
    if capture_layer < 0 or capture_layer >= len(blocks):
        raise ValueError(f"capture_layer out of range: {capture_layer}")
    if ablate_layer < 0 or ablate_layer >= len(blocks):
        raise ValueError(f"ablate_layer out of range: {ablate_layer}")

    ablate_block = blocks[ablate_layer]
    capture_block = blocks[capture_layer]

    if not hasattr(ablate_block, "mlp") or not hasattr(ablate_block.mlp, "act"):
        raise ValueError("This ablation helper currently supports GPT-2 style blocks with .mlp.act")

    neuron_idx_t = torch.as_tensor(neuron_idx, device=device, dtype=torch.long)

    out_chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        attention_mask = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))
        last_idx = attention_mask.sum(dim=1) - 1
        bsz = enc["input_ids"].shape[0]
        batch_arange = torch.arange(bsz, device=device)

        with lm.trace(enc):
            a = ablate_block.mlp.act.output
            a[:, :, neuron_idx_t] = 0
            h = capture_block.output[0].save()  # [B, S, H]

        h_t = h.value if hasattr(h, "value") else h
        if h_t.ndim == 3 and h_t.shape[0] != bsz and h_t.shape[1] == bsz:
            h_t = h_t.transpose(0, 1)
        pooled = h_t[batch_arange, last_idx].detach().to("cpu", dtype=torch.float32).numpy()
        out_chunks.append(pooled)

    return np.concatenate(out_chunks, axis=0)


def projection_magnitude(deltas: np.ndarray, concept_dir: np.ndarray) -> float:
    """
    Mean absolute projection magnitude of Δ onto concept_dir.
    deltas: [n_pairs, hidden]
    """
    return float(np.mean(np.abs(deltas @ concept_dir)))


def eval_ablation_effect(
    *,
    keys: list[str],
    acts: np.ndarray,
    pairs: list[DeltaPair],
    lm: LanguageModel,
    prompts: list[str],
    selection: NeuronSelection,
    concept_dir: np.ndarray,
    d_mlp: int,
    batch_size: int,
    device: torch.device | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    Evaluate change in projection magnitude at layer+1 before/after neuron ablation,
    with a matched random-neuron control.
    """
    if device is None:
        device = get_device()
    if rng is None:
        rng = np.random.default_rng(0)

    layer = selection.layer
    concept_dir = _unit(concept_dir.astype(np.float32))

    # Baseline: use cached residual activations at capture_layer=layer+1
    deltas_all = deltas_from_pairs(keys=keys, acts=acts, pairs=pairs)
    baseline_pairs = deltas_all[:, layer + 1, :]
    baseline = projection_magnitude(baseline_pairs, concept_dir)

    # Target ablation capture
    ablated_h = capture_last_token_resid_layer_with_ablation(
        lm,
        prompts,
        ablate_layer=layer,
        neuron_idx=selection.neuron_idx,
        capture_layer=layer + 1,
        batch_size=batch_size,
        device=device,
        desc=f"ablate:eval:L{layer}",
    )

    # Random control ablation (same # neurons)
    d_mlp = int(d_mlp)
    if d_mlp <= 0:
        raise ValueError(f"d_mlp must be positive, got {d_mlp}")
    rand_idx = rng.choice(d_mlp, size=len(selection.neuron_idx), replace=False)
    rand_h = capture_last_token_resid_layer_with_ablation(
        lm,
        prompts,
        ablate_layer=layer,
        neuron_idx=rand_idx,
        capture_layer=layer + 1,
        batch_size=batch_size,
        device=device,
        desc=f"ablate:eval:rand:L{layer}",
    )

    # Compute Δ from captured layer+1 residuals
    key_to_row = {k: i for i, k in enumerate(keys)}
    neg_rows = np.array([key_to_row[p.neg_key] for p in pairs], dtype=np.int64)
    pos_rows = np.array([key_to_row[p.pos_key] for p in pairs], dtype=np.int64)
    delta_ablated = (ablated_h[pos_rows] - ablated_h[neg_rows]).astype(np.float32)
    delta_rand = (rand_h[pos_rows] - rand_h[neg_rows]).astype(np.float32)

    ablated = projection_magnitude(delta_ablated, concept_dir)
    rand = projection_magnitude(delta_rand, concept_dir)

    return {
        "baseline": baseline,
        "ablated": ablated,
        "random_ablated": rand,
        "effect": baseline - ablated,
        "random_effect": baseline - rand,
    }


def save_neuron_selections_csv(selections: list[NeuronSelection], outpath: Path) -> None:
    ensure_dir(outpath.parent)
    rows = []
    for s in selections:
        for idx, score in zip(s.neuron_idx.tolist(), s.score.tolist(), strict=True):
            rows.append({"layer": s.layer, "neuron_idx": int(idx), "score": float(score)})
    pd.DataFrame(rows).to_csv(outpath, index=False)
