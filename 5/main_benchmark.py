#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if sys.platform == "darwin":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from gradient_descent import DataParallelGradientDescent, SequentialFiniteSumGradientDescent

SIZES: Sequence[int] = (1000, 5000, 10000, 20000)

WORKERS: Sequence[int] = (2, 4, 5, 6, 7, 12)

COMMON = dict(learning_rate=0.1, max_iterations=200, tolerance=1e-5)

X0 = np.array([5.0])

DEC = 3


def bench_row(n: int) -> Tuple[float, Dict[int, float]]:
    """
    Для фіксованого n генеруємо c один раз, далі вимірюємо послідовний час
    та час паралельної версії для кожного значення з WORKERS.
    """
    rng = np.random.default_rng(42 + n)
    c = rng.uniform(0.0, 10.0, size=n)

    t0 = time.perf_counter()
    SequentialFiniteSumGradientDescent(**COMMON).fit(c, X0)
    t_seq = time.perf_counter() - t0

    t_par: Dict[int, float] = {}
    for w in WORKERS:
        t0 = time.perf_counter()
        DataParallelGradientDescent(**COMMON, n_workers=w).fit(c, X0)
        t_par[w] = time.perf_counter() - t0

    return t_seq, t_par


def print_table(rows: List[Tuple[int, float, Dict[int, float]]]) -> None:
    title = (
        "Таблиця 5.1 — Час виконання (с) та прискорення паралельної реалізації "
        "(data-parallel, скінчена сума)"
    )
    print()
    print(title)
    print()

    # Заголовок: перший рядок
    h1 = f"{'N':>8}  {'Послідовно':^12}"
    for w in WORKERS:
        h1 += f"  {w} обчислювачів".center(22)
    print(h1)

    # Другий рядок: підписи Час / Пр.
    h2 = f"{'':>8}  {'час, с':>12}"
    for _ in WORKERS:
        h2 += f"  {'час, с':>10}  {'Пр.':>8}"
    print(h2)
    print("-" * (22 + len(WORKERS) * 24))

    for n, t_seq, t_par in rows:
        line = f"{n:>8}  {t_seq:>12.{DEC}f}"
        for w in WORKERS:
            tp = t_par[w]
            sp = t_seq / tp if tp > 0 else float("nan")
            line += f"  {tp:>10.{DEC}f}  {sp:>8.{DEC}f}"
        print(line)
    print()


def write_csv(path: str, rows: List[Tuple[int, float, Dict[int, float]]]) -> None:
    fieldnames = ["N", "T_seq_s"]
    for w in WORKERS:
        fieldnames.append(f"T_parallel_{w}_s")
        fieldnames.append(f"speedup_{w}")

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for n, t_seq, t_par in rows:
            row: Dict[str, object] = {"N": n, "T_seq_s": f"{t_seq:.{DEC}f}"}
            for nw in WORKERS:
                tp = t_par[nw]
                sp = t_seq / tp if tp > 0 else ""
                row[f"T_parallel_{nw}_s"] = f"{tp:.{DEC}f}"
                row[f"speedup_{nw}"] = f"{sp:.{DEC}f}" if sp != "" else ""
            w.writerow(row)


def print_table_5_2(rows: List[Tuple[int, float, Dict[int, float]]]) -> None:
    """Таблиця 5.2: лише прискорення (Пр.) та ефективність (Еф. = Пр. / p)."""
    title = (
        "Таблиця 5.2 — Прискорення та ефективність паралельної реалізації "
        "(data-parallel, скінчена сума)"
    )
    print()
    print(title)
    print()

    h1 = f"{'N':>8}"
    for w in WORKERS:
        h1 += f"  {w} обчислювачів".center(22)
    print(h1)

    h2 = f"{'':>8}"
    for _ in WORKERS:
        h2 += f"  {'Пр.':>10}  {'Еф.':>8}"
    print(h2)
    print("-" * (8 + len(WORKERS) * 24))

    for n, t_seq, t_par in rows:
        line = f"{n:>8}"
        for w in WORKERS:
            tp = t_par[w]
            sp = t_seq / tp if tp > 0 else float("nan")
            eff = sp / w if tp > 0 and w > 0 else float("nan")
            line += f"  {sp:>10.{DEC}f}  {eff:>8.{DEC}f}"
        print(line)
    print()


def load_rows_from_csv(path: str) -> Tuple[List[Tuple[int, float, Dict[int, float]]], Tuple[int, ...]]:
    """Читає table_5_1.csv; повертає rows і кортеж воркерів з заголовка."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        workers_list: List[int] = []
        for fn in fieldnames:
            m = re.match(r"T_parallel_(\d+)_s", fn or "")
            if m:
                workers_list.append(int(m.group(1)))
        workers = tuple(workers_list)
        rows: List[Tuple[int, float, Dict[int, float]]] = []
        for row in reader:
            n = int(row["N"])
            t_seq = float(row["T_seq_s"])
            t_par = {w: float(row[f"T_parallel_{w}_s"]) for w in workers}
            rows.append((n, t_seq, t_par))
    return rows, workers


def plot_execution_times(
    rows: List[Tuple[int, float, Dict[int, float]]],
    out_path: str,
    workers: Optional[Sequence[int]] = None,
) -> None:
    """
    Графік: вісь X — N (розмір задачі), вісь Y — час виконання, мс.
    Криві: послідовна + паралельна для кожного p (як на рис. залежності часу від n).
    """
    wk = tuple(workers) if workers is not None else WORKERS
    if not rows:
        raise ValueError("Немає даних для графіка")

    ns = [r[0] for r in rows]
    t_seq_ms = [r[1] * 1000.0 for r in rows]

    color_seq = "#1f77b4"
    colors = ["#d62728", "#bcbd22", "#2ca02c", "#ff7f0e", "#17becf", "#9467bd", "#e377c2"]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    ax.plot(ns, t_seq_ms, color=color_seq, linewidth=2.2, label="sequential")

    for i, p in enumerate(wk):
        ys = [r[2][p] * 1000.0 for r in rows]
        c = colors[i % len(colors)]
        ax.plot(ns, ys, color=c, linewidth=2.0, label=f"threads_{p}")

    ax.set_xlabel("N")
    ax.set_ylabel("Час виконання, мс")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_csv_5_2(path: str, rows: List[Tuple[int, float, Dict[int, float]]]) -> None:
    fieldnames = ["N"]
    for w in WORKERS:
        fieldnames.append(f"speedup_{w}")
        fieldnames.append(f"efficiency_{w}")

    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for n, t_seq, t_par in rows:
            row: Dict[str, object] = {"N": n}
            for nw in WORKERS:
                tp = t_par[nw]
                sp = t_seq / tp if tp > 0 else float("nan")
                eff = sp / nw if tp > 0 and nw > 0 else float("nan")
                row[f"speedup_{nw}"] = f"{sp:.{DEC}f}" if tp > 0 else ""
                row[f"efficiency_{nw}"] = f"{eff:.{DEC}f}" if tp > 0 else ""
            wr.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Таблиці 5.1–5.2 та графік часу від N для finite-sum GD."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Лише зберегти benchmark_5.png з наявного table_5_1.csv (без замірів)",
    )
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(base, "benchmark_5.png")

    if args.plot_only:
        csv_path = os.path.join(base, "table_5_1.csv")
        if not os.path.isfile(csv_path):
            alt = os.path.join(_PROJECT_ROOT, "table_5_1.csv")
            if os.path.isfile(alt):
                csv_path = alt
            else:
                print(f"Файл не знайдено: {csv_path} (ані в корені проєкту)", file=sys.stderr)
                sys.exit(1)
        rows, workers_csv = load_rows_from_csv(csv_path)
        plot_execution_times(rows, png_path, workers=workers_csv)
        print(f"Графік збережено: {png_path}")
        return

    rows = []
    for n in SIZES:
        print(f"Вимірювання N = {n} ...", flush=True)
        t_seq, t_par = bench_row(n)
        rows.append((n, t_seq, t_par))

    print_table(rows)

    out1 = os.path.join(base, "table_5_1.csv")
    write_csv(out1, rows)
    print(f"CSV збережено: {out1}")

    print_table_5_2(rows)
    out2 = os.path.join(base, "table_5_2.csv")
    write_csv_5_2(out2, rows)
    print(f"CSV збережено: {out2}")

    plot_execution_times(rows, png_path, workers=WORKERS)
    print(f"Графік збережено: {png_path}")


if __name__ == "__main__":
    main()
