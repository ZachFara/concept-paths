from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from .utils import ensure_dir


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.0, 4.0),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def plot_curve_with_ci(
    *,
    x: np.ndarray,
    mean: np.ndarray,
    low: Optional[np.ndarray],
    high: Optional[np.ndarray],
    label: str,
    title: str,
    ylabel: str,
    outpath: Path,
) -> None:
    _setup_matplotlib()
    ensure_dir(outpath.parent)
    plt.figure()
    plt.plot(x, mean, label=label, linewidth=2)
    if low is not None and high is not None and low.size and high.size:
        plt.fill_between(x, low, high, alpha=0.2)
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_overlay_curves(
    *,
    x: np.ndarray,
    curves: Dict[str, np.ndarray],
    title: str,
    ylabel: str,
    outpath: Path,
) -> None:
    _setup_matplotlib()
    ensure_dir(outpath.parent)
    plt.figure()
    for label, values in curves.items():
        plt.plot(x, values, label=label, linewidth=2)
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_null_hist(
    *,
    values: np.ndarray,
    observed: float,
    title: str,
    xlabel: str,
    outpath: Path,
) -> None:
    _setup_matplotlib()
    ensure_dir(outpath.parent)
    plt.figure()
    plt.hist(values, bins=30, alpha=0.7)
    plt.axvline(observed, color="red", linestyle="--", linewidth=2, label="observed")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_heatmap(
    *,
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    outpath: Path,
    cmap: str = "viridis",
) -> None:
    _setup_matplotlib()
    ensure_dir(outpath.parent)
    plt.figure()
    plt.imshow(matrix, cmap=cmap, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    plt.yticks(range(len(ylabels)), ylabels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_bar(
    *,
    labels: list[str],
    values: np.ndarray,
    title: str,
    ylabel: str,
    outpath: Path,
) -> None:
    _setup_matplotlib()
    ensure_dir(outpath.parent)
    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
