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

from gradient_descent import (
    DataParallelGradientDescent,
    SequentialGradientDescent,
    finite_sum_mean_squared_grad,
    finite_sum_mean_squared_partial_grad,
)

SIZES: Sequence[int] = (1000, 5000, 10000, 20000)

WORKERS: Sequence[int] = (2, 4, 6, 8, 10, 12)

COMMON = dict(learning_rate=0.1, max_iterations=200, tolerance=1e-5)

X0 = np.array([5.0])

DEC = 2

PARALLEL_IMPLS: Sequence[str] = ("map", "recursive")


def bench_row(n: int, parallel_impl: str = "map") -> Tuple[float, Dict[int, float]]:
    """
    Для фіксованого n генеруємо c один раз, далі вимірюємо послідовний час
    та час паралельної версії (обраний parallel_impl) для кожного w з WORKERS.
    """
    rng = np.random.default_rng(42 + n)
    c = rng.uniform(0.0, 10.0, size=n)

    t0 = time.perf_counter()
    SequentialGradientDescent(**COMMON).fit(
        X0, finite_sum_mean_squared_grad, c=c
    )
    t_seq = time.perf_counter() - t0

    t_par: Dict[int, float] = {}
    for w in WORKERS:
        t0 = time.perf_counter()
        DataParallelGradientDescent(
            **COMMON, n_workers=w, parallel_impl=parallel_impl
        ).fit(
            X0,
            partial_grad_fn=finite_sum_mean_squared_partial_grad,
            c=c,
        )
        t_par[w] = time.perf_counter() - t0

    return t_seq, t_par


def bench_row_compare_impls(n: int) -> Tuple[float, Dict[int, Dict[str, float]]]:
    """
    Для фіксованого n вимірює:
    - послідовний час;
    - паралельний час для кожного w з WORKERS і кожної impl з PARALLEL_IMPLS.
    """
    rng = np.random.default_rng(42 + n)
    c = rng.uniform(0.0, 10.0, size=n)

    t0 = time.perf_counter()
    SequentialGradientDescent(**COMMON).fit(
        X0, finite_sum_mean_squared_grad, c=c
    )
    t_seq = time.perf_counter() - t0

    t_par: Dict[int, Dict[str, float]] = {}
    for w in WORKERS:
        t_par[w] = {}
        for impl in PARALLEL_IMPLS:
            t0 = time.perf_counter()
            DataParallelGradientDescent(
                **COMMON, n_workers=w, parallel_impl=impl
            ).fit(
                X0,
                partial_grad_fn=finite_sum_mean_squared_partial_grad,
                c=c,
            )
            t_par[w][impl] = time.perf_counter() - t0

    return t_seq, t_par


def print_table(
    rows: List[Tuple[int, float, Dict[int, float]]],
    *,
    table_label: str = "Таблиця 5.1",
    impl_label: str = "map",
) -> None:
    title = f"{table_label} — Час виконання (с) та прискорення ({impl_label})"
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


def print_table_speedup_eff(
    rows: List[Tuple[int, float, Dict[int, float]]],
    *,
    title: str = "Таблиця — Прискорення та ефективність",
) -> None:
    """Лише прискорення (Пр.) та ефективність (Еф. = Пр. / p)."""
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


def write_csv_impl_compare(
    path: str, rows: List[Tuple[int, float, Dict[int, Dict[str, float]]]]
) -> None:
    fieldnames = ["N", "T_seq_s"]
    for w in WORKERS:
        for impl in PARALLEL_IMPLS:
            fieldnames.append(f"T_parallel_{impl}_{w}_s")
            fieldnames.append(f"speedup_{impl}_{w}")

    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for n, t_seq, t_par in rows:
            row: Dict[str, object] = {"N": n, "T_seq_s": f"{t_seq:.{DEC}f}"}
            for w in WORKERS:
                for impl in PARALLEL_IMPLS:
                    tp = t_par[w][impl]
                    sp = t_seq / tp if tp > 0 else float("nan")
                    row[f"T_parallel_{impl}_{w}_s"] = f"{tp:.{DEC}f}"
                    row[f"speedup_{impl}_{w}"] = f"{sp:.{DEC}f}" if tp > 0 else ""
            wr.writerow(row)


def plot_impl_comparison(
    rows: List[Tuple[int, float, Dict[int, Dict[str, float]]]],
    out_path: str,
    workers: Optional[Sequence[int]] = None,
) -> None:
    """
    Графік порівняння двох паралельних реалізацій:
    - map: суцільні лінії
    - recursive: пунктирні лінії
    """
    wk = tuple(workers) if workers is not None else WORKERS
    if not rows:
        raise ValueError("Немає даних для графіка")

    ns = [r[0] for r in rows]
    t_seq_ms = [r[1] * 1000.0 for r in rows]

    colors = ["#d62728", "#bcbd22", "#2ca02c", "#ff7f0e", "#17becf", "#9467bd", "#e377c2"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.plot(ns, t_seq_ms, color="#1f77b4", linewidth=2.2, label="sequential")

    for i, p in enumerate(wk):
        c = colors[i % len(colors)]
        ys_map = [r[2][p]["map"] * 1000.0 for r in rows]
        ys_rec = [r[2][p]["recursive"] * 1000.0 for r in rows]
        ax.plot(ns, ys_map, color=c, linewidth=2.0, linestyle="-", label=f"map_p{p}")
        ax.plot(ns, ys_rec, color=c, linewidth=2.0, linestyle="--", label=f"recursive_p{p}")

    ax.set_xlabel("N")
    ax.set_ylabel("Час виконання, мс")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", ncol=2, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_csv_speedup_eff(path: str, rows: List[Tuple[int, float, Dict[int, float]]]) -> None:
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


def _measure_rows(parallel_impl: str) -> List[Tuple[int, float, Dict[int, float]]]:
    rows = []
    for n in SIZES:
        print(f"Вимірювання N = {n} (impl={parallel_impl}) ...", flush=True)
        t_seq, t_par = bench_row(n, parallel_impl=parallel_impl)
        rows.append((n, t_seq, t_par))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Таблиці 5.1 (map, час+прискорення), 5.2 (recursive, час+прискорення), "
            "5.3 (map, прискорення+ефективність), 5.4 (recursive, прискорення+ефективність)."
        )
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Лише зберегти benchmark_5.png з наявного table_5_1.csv (без замірів)",
    )
    parser.add_argument(
        "--compare-impls",
        action="store_true",
        help="Додатково заміряти map vs recursive та зберегти окремий CSV/графік",
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

    rows_map = _measure_rows("map")
    print_table(rows_map, table_label="Таблиця 5.1", impl_label="map")
    out1 = os.path.join(base, "table_5_1.csv")
    write_csv(out1, rows_map)
    print(f"CSV збережено: {out1}")

    rows_recursive = _measure_rows("recursive")
    print_table(rows_recursive, table_label="Таблиця 5.2", impl_label="recursive")
    out2 = os.path.join(base, "table_5_2.csv")
    write_csv(out2, rows_recursive)
    print(f"CSV збережено: {out2}")

    print_table_speedup_eff(
        rows_map,
        title="Таблиця 5.3 — Прискорення та ефективність (map)",
    )
    out3 = os.path.join(base, "table_5_3.csv")
    write_csv_speedup_eff(out3, rows_map)
    print(f"CSV збережено: {out3}")

    print_table_speedup_eff(
        rows_recursive,
        title="Таблиця 5.4 — Прискорення та ефективність (recursive)",
    )
    out4 = os.path.join(base, "table_5_4.csv")
    write_csv_speedup_eff(out4, rows_recursive)
    print(f"CSV збережено: {out4}")

    if args.compare_impls:
        rows_impl = []
        for n in SIZES:
            print(f"Порівняння impl для N = {n} ...", flush=True)
            t_seq, t_par_impl = bench_row_compare_impls(n)
            rows_impl.append((n, t_seq, t_par_impl))

        out_impl_csv = os.path.join(base, "table_5_impl_compare.csv")
        write_csv_impl_compare(out_impl_csv, rows_impl)
        print(f"CSV (impl compare) збережено: {out_impl_csv}")

        out_impl_png = os.path.join(base, "benchmark_5_impl_compare.png")
        plot_impl_comparison(rows_impl, out_impl_png, workers=WORKERS)
        print(f"Графік (impl compare) збережено: {out_impl_png}")
    else:
        plot_execution_times(rows_map, png_path, workers=WORKERS)
        print(f"Графік збережено: {png_path}")


if __name__ == "__main__":
    main()
