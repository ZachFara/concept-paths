from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

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


def plot_metric_by_layer(
    *,
    values_by_split: Mapping[str, np.ndarray],
    title: str,
    ylabel: str,
    outpath: Path,
) -> None:
    """
    Plot one or more curves (e.g. discovery vs eval) against layer index.
    """
    _setup_matplotlib()
    ensure_dir(outpath.parent)

    plt.figure(figsize=(7.0, 4.0))
    for split, values in values_by_split.items():
        x = np.arange(len(values))
        plt.plot(x, values, marker="o", linewidth=2, label=split)

    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_with_band(
    *,
    x: np.ndarray,
    mean: np.ndarray,
    std: Optional[np.ndarray],
    label: str,
    title: str,
    ylabel: str,
    outpath: Path,
) -> None:
    _setup_matplotlib()
    ensure_dir(outpath.parent)
    plt.figure(figsize=(7.0, 4.0))
    plt.plot(x, mean, label=label, linewidth=2)
    if std is not None:
        plt.fill_between(x, mean - std, mean + std, alpha=0.25)
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
