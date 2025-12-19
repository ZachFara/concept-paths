from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def _save_raw_curve(outpath: Path, x: np.ndarray, curves: Mapping[str, np.ndarray]) -> None:
    ensure_dir(outpath.parent)
    data = {"x": x}
    data.update(curves)
    np.savez_compressed(outpath, **data)


def plot_curves_with_ci(
    *,
    x: np.ndarray,
    mean: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    label: str,
    title: str,
    ylabel: str,
    outpath: Path,
    raw_out: Optional[Path] = None,
) -> None:
    _setup_matplotlib()
    ensure_dir(outpath.parent)
    plt.figure(figsize=(7.0, 4.0))
    plt.plot(x, mean, label=label, linewidth=2)
    plt.fill_between(x, ci_low, ci_high, alpha=0.25)
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    if raw_out:
        _save_raw_curve(raw_out, x, {"mean": mean, "ci_low": ci_low, "ci_high": ci_high})


def plot_main_vs_control_overlay(
    *,
    x: np.ndarray,
    main: np.ndarray,
    control: np.ndarray,
    label_main: str,
    label_control: str,
    title: str,
    ylabel: str,
    outpath: Path,
    raw_out: Optional[Path] = None,
) -> None:
    _setup_matplotlib()
    ensure_dir(outpath.parent)
    plt.figure(figsize=(7.0, 4.0))
    plt.plot(x, main, label=label_main, linewidth=2)
    plt.plot(x, control, label=label_control, linewidth=2, linestyle="--")
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    if raw_out:
        _save_raw_curve(raw_out, x, {"main": main, "control": control})


def plot_null_histogram(
    *,
    null_values: np.ndarray,
    real_value: float,
    title: str,
    xlabel: str,
    outpath: Path,
    raw_out: Optional[Path] = None,
) -> None:
    _setup_matplotlib()
    ensure_dir(outpath.parent)
    plt.figure(figsize=(6.0, 4.0))
    plt.hist(null_values, bins=30, alpha=0.7, label="null")
    plt.axvline(real_value, color="red", linestyle="--", label="real")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    if raw_out:
        ensure_dir(raw_out.parent)
        np.savez_compressed(raw_out, null=null_values, real=np.array([real_value]))


def plot_heatmap(
    *,
    matrix: np.ndarray,
    x_labels: Optional[List[str]],
    y_labels: Optional[List[str]],
    title: str,
    outpath: Path,
    raw_out: Optional[Path] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    _setup_matplotlib()
    ensure_dir(outpath.parent)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    if x_labels:
        plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    if y_labels:
        plt.yticks(range(len(y_labels)), y_labels)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    if raw_out:
        np.savez_compressed(raw_out, matrix=matrix)


def plot_bar(
    *,
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    outpath: Path,
    raw_out: Optional[Path] = None,
) -> None:
    _setup_matplotlib()
    ensure_dir(outpath.parent)
    plt.figure(figsize=(6, 4))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    if raw_out:
        np.savez_compressed(raw_out, labels=np.array(labels), values=np.array(values))
