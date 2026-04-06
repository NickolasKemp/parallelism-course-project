#!/usr/bin/env python3

import os
import sys
import time
from multiprocessing import cpu_count
from typing import List, Optional, Tuple

import numpy as np

if sys.platform == "darwin":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gradient_descent import (
    DataParallelGradientDescent,
    SequentialFiniteSumGradientDescent,
)

SIZES: List[int] = [1000, 5000, 10000, 20000, 50000]

N_WORKERS: Optional[int] = None

SEED = 42


def bench_one(
    n: int,
    rng: np.random.Generator,
    common: dict,
    n_workers: int,
) -> Tuple[float, float]:
    c = rng.uniform(0.0, 10.0, size=n)
    x0 = np.array([5.0])

    t0 = time.perf_counter()
    # SequentialFiniteSumGradientDescent(**common).fit(c, x0)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    DataParallelGradientDescent(**common, n_workers=n_workers).fit(c, x0)
    t_par = time.perf_counter() - t0

    return t_seq, t_par


if __name__ == "__main__":
    # n_workers = N_WORKERS if N_WORKERS is not None else max(2, cpu_count() or 4)
    n_workers = 16
    common = dict(learning_rate=0.1, max_iterations=150, tolerance=1e-5)
    rng = np.random.default_rng(SEED)

    print(
        f"{'N':>12}  {'Послідовно, мс':>14}  {'Паралельно, мс':>14}  {'Прискор.':>8}"
    )
    print("-" * 56)

    ns: List[int] = []
    seq_ms: List[float] = []
    par_ms: List[float] = []

    for n in SIZES:
        t_seq, t_par = bench_one(n, rng, common, n_workers)
        ms_seq = t_seq * 1000.0
        ms_par = t_par * 1000.0
        sp = t_seq / t_par if t_par > 0 else 0.0
        print(f"{n:>12}  {ms_seq:>14.2f}  {ms_par:>14.2f}  {sp:>7.2f}x")
        ns.append(n)
        seq_ms.append(ms_seq)
        par_ms.append(ms_par)

    _dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(_dir, "benchmark_finite_sum.png")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(ns, seq_ms, "o-", color="#1f77b4", linewidth=2, markersize=6, label="Послідовна")
    ax.plot(ns, par_ms, "s-", color="#ff7f0e", linewidth=2, markersize=6, label="Паралельна")
    ax.set_xlabel("N (кількість елементів c)")
    ax.set_ylabel("Час виконання, мс")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print()
    print(f"Графік збережено: {png_path}")
