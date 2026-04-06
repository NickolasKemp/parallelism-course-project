"""
Перевірка коректності: задача з відомим аналітичним мінімумом.

f(x) = (1/N) * sum_i (x - c_i)^2  ->  мінімум у x* = mean(c).

Приклад: c = [1, 3, 5, 7]  =>  x* = 4.

Запуск:
  python3 test_small.py           # усе: послідовна + паралельна + порівняння
  python3 test_small.py seq       # лише послідовна
  python3 test_small.py par       # лише паралельна
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Tuple

import numpy as np

from gradient_descent import (
    DataParallelGradientDescent,
    SequentialFiniteSumGradientDescent,
)

N_WORKERS = 2


def _ensure_spawn() -> None:
    if sys.platform == "darwin":
        import multiprocessing

        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass


def small_problem() -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    c = np.array([1.0, 3.0, 5.0, 7.0])
    x0 = np.array([0.0])
    expected = float(np.mean(c))
    common: Dict[str, Any] = dict(
        learning_rate=0.1,
        max_iterations=500,
        tolerance=1e-8,
    )
    return c, x0, expected, common


def _print_header(c: np.ndarray, expected: float) -> None:
    print("f(x) = (1/N) * sum_i (x - c_i)^2")
    print(f"  c = {c.tolist()},  N = {len(c)}")
    print(f"  Очікуваний мінімум: x* = mean(c) = {expected}")
    print()


def run_sequential() -> Tuple[np.ndarray, Dict[str, Any]]:
    c, x0, expected, common = small_problem()
    _print_header(c, expected)

    x, info = SequentialFiniteSumGradientDescent(**common).fit(c, x0)
    print(
        f"Послідовна:  x = {x},  converged={info['converged']}, "
        f"iters={info['iterations']}"
    )
    print()
    print(f"|x - x*| = {np.abs(x - expected).max():.2e}")
    ok = np.allclose(x, expected, atol=1e-5)
    print(f"np.allclose(x, expected, atol=1e-5): {ok}")
    assert ok
    return x, info


def run_parallel() -> Tuple[np.ndarray, Dict[str, Any]]:
    _ensure_spawn()
    c, x0, expected, common = small_problem()
    _print_header(c, expected)

    x, info = DataParallelGradientDescent(
        **common, n_workers=N_WORKERS
    ).fit(c, x0)
    print(
        f"Паралельна ({N_WORKERS} workers):  x = {x},  "
        f"converged={info['converged']}, iters={info['iterations']}"
    )
    print()
    print(f"|x - x*| = {np.abs(x - expected).max():.2e}")
    ok = np.allclose(x, expected, atol=1e-5)
    print(f"np.allclose(x, expected, atol=1e-5): {ok}")
    assert ok
    return x, info


def run_all() -> None:
    _ensure_spawn()
    c, x0, expected, common = small_problem()
    _print_header(c, expected)

    x_seq, info_seq = SequentialFiniteSumGradientDescent(**common).fit(c, x0)
    x_par, info_par = DataParallelGradientDescent(
        **common, n_workers=N_WORKERS
    ).fit(c, x0)

    print(
        f"Послідовна:  x = {x_seq},  converged={info_seq['converged']}, "
        f"iters={info_seq['iterations']}"
    )
    print(
        f"Паралельна ({N_WORKERS} workers):  x = {x_par},  "
        f"converged={info_par['converged']}, iters={info_par['iterations']}"
    )
    print()
    print(f"|x_seq - x*| = {np.abs(x_seq - expected).max():.2e}")
    print(f"|x_par - x*| = {np.abs(x_par - expected).max():.2e}")
    print(f"max |x_seq - x_par| = {np.max(np.abs(x_seq - x_par)):.2e}")
    print()
    ok_seq = np.allclose(x_seq, expected, atol=1e-5)
    ok_par = np.allclose(x_par, expected, atol=1e-5)
    ok_agree = np.allclose(x_seq, x_par, rtol=1e-9, atol=1e-10)
    print(f"np.allclose(x_seq, expected, atol=1e-5): {ok_seq}")
    print(f"np.allclose(x_par, expected, atol=1e-5): {ok_par}")
    print(f"np.allclose(x_seq, x_par): {ok_agree}")

    assert ok_seq
    assert ok_par
    assert ok_agree


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Мала перевірка finite-sum GD (послідовна / паралельна / обидві)."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=("seq", "par", "all"),
        help="seq — лише послідовна; par — лише паралельна; all — обидві та порівняння",
    )
    args = parser.parse_args()

    if args.mode == "seq":
        run_sequential()
    elif args.mode == "par":
        run_parallel()
    else:
        run_all()
