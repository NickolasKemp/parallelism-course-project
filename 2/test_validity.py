"""
Перевірка коректності GD на кількох типах задач.

f(x) = (1/N) * sum_i ||x - c_i||^2  ->  мінімум у x* = mean(c, axis=0).

Сценарії (від простого до складного):
1) finite-sum MSE: 1D, 2D;
2) аналітичні функції:
   - зсунутий квадратичний потенціал,
   - анізотропний квадратичний потенціал.
3) для паралельної частини використовується stress-тест на великому N
   (перевіряються обидві реалізації: map і recursive).

Запуск (з каталогу ``2/`` або з кореня проєкту):
  python3 test_validity.py           # усе: послідовна + паралельна + порівняння
  python3 test_validity.py seq       # лише послідовна
  python3 test_validity.py par       # лише паралельна
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from gradient_descent import (
    DataParallelGradientDescent,
    SequentialGradientDescent,
    finite_sum_mean_squared_grad,
    finite_sum_mean_squared_partial_grad,
)

N_WORKERS = 2


@dataclass(frozen=True)
class ValidityCase:
    name: str
    c: np.ndarray
    x0: np.ndarray
    learning_rate: float
    max_iterations: int
    tolerance: float


@dataclass(frozen=True)
class AnalyticFunctionCase:
    name: str
    x0: np.ndarray
    expected: np.ndarray
    learning_rate: float
    max_iterations: int
    tolerance: float
    grad_fn: Any


def _ensure_spawn() -> None:
    if sys.platform == "darwin":
        import multiprocessing

        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass


def build_validity_cases() -> List[ValidityCase]:
    case_1d = ValidityCase(
        name="1D / базовий",
        c=np.array([1.0, 3.0, 5.0, 7.0]),
        x0=np.array([0.0]),
        learning_rate=0.1,
        max_iterations=500,
        tolerance=1e-8,
    )
    case_2d = ValidityCase(
        name="2D / анізотропний",
        c=np.array(
            [
                [1.0, -5.0],
                [3.0, 0.0],
                [5.0, 4.0],
                [9.0, 7.0],
            ]
        ),
        x0=np.array([-10.0, 10.0]),
        learning_rate=0.08,
        max_iterations=1000,
        tolerance=1e-8,
    )
    return [case_1d, case_2d]


def shifted_quadratic_grad(
    x: np.ndarray,
    c: Optional[np.ndarray] = None,
) -> np.ndarray:
    del c
    target = np.array([2.0, -1.0], dtype=float)
    return 2.0 * (x - target)


def anisotropic_quadratic_grad(
    x: np.ndarray,
    c: Optional[np.ndarray] = None,
) -> np.ndarray:
    del c
    target = np.array([-3.0, 0.5, 4.0], dtype=float)
    diag = np.array([1.0, 5.0, 10.0], dtype=float)
    return 2.0 * diag * (x - target)


def build_analytic_function_cases() -> List[AnalyticFunctionCase]:
    return [
        AnalyticFunctionCase(
            name="Зсунутий квадратичний потенціал",
            x0=np.array([10.0, -10.0]),
            expected=np.array([2.0, -1.0]),
            learning_rate=0.1,
            max_iterations=400,
            tolerance=1e-8,
            grad_fn=shifted_quadratic_grad,
        ),
        AnalyticFunctionCase(
            name="Анізотропний квадратичний потенціал",
            x0=np.array([5.0, 5.0, -5.0]),
            expected=np.array([-3.0, 0.5, 4.0]),
            learning_rate=0.03,
            max_iterations=900,
            tolerance=1e-8,
            grad_fn=anisotropic_quadratic_grad,
        ),
    ]


def build_parallel_stress_case() -> ValidityCase:
    rng = np.random.default_rng(7)
    c = rng.normal(loc=1.5, scale=2.0, size=(20000, 3))
    return ValidityCase(
        name="Stress / великий N 20000, d=3",
        c=c,
        x0=np.array([15.0, -12.0, 8.0]),
        learning_rate=0.1,
        max_iterations=120,
        tolerance=1e-6,
    )


def _case_expected(case: ValidityCase) -> np.ndarray:
    c = np.asarray(case.c, dtype=float)
    if c.ndim == 1:
        return np.array([float(np.mean(c))], dtype=float)
    return np.mean(c, axis=0)


def _print_header(case: ValidityCase, expected: np.ndarray) -> None:
    c = np.asarray(case.c, dtype=float)
    print("f(x) = (1/N) * sum_i (x - c_i)^2")
    print(f"Кейс: {case.name}")
    print(f"  c.shape = {c.shape},  N = {len(c)}")
    print(f"  x0 = {case.x0}")
    print(f"  Очікуваний мінімум: x* = mean(c) = {expected}")
    print()


def _run_case_sequential(case: ValidityCase) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    expected = _case_expected(case)
    _print_header(case, expected)
    common: Dict[str, Any] = dict(
        learning_rate=case.learning_rate,
        max_iterations=case.max_iterations,
        tolerance=case.tolerance,
    )

    x, info = SequentialGradientDescent(**common).fit(
        case.x0, finite_sum_mean_squared_grad, c=case.c
    )
    print(
        f"Послідовна:  x = {x},  converged={info['converged']}, "
        f"iters={info['iterations']}"
    )
    print()
    print(f"|x - x*|_inf = {np.abs(x - expected).max():.2e}")
    ok = np.allclose(x, expected, atol=1e-5, rtol=1e-7)
    print(f"np.allclose(x, expected, atol=1e-5): {ok}")
    assert ok
    return x, info, expected


def _run_case_parallel(case: ValidityCase) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    _ensure_spawn()
    expected = _case_expected(case)
    _print_header(case, expected)
    common: Dict[str, Any] = dict(
        learning_rate=case.learning_rate,
        max_iterations=case.max_iterations,
        tolerance=case.tolerance,
    )

    x, info = DataParallelGradientDescent(**common, n_workers=N_WORKERS).fit(
        case.x0,
        partial_grad_fn=finite_sum_mean_squared_partial_grad,
        c=case.c,
    )
    print(
        f"Паралельна ({N_WORKERS} workers):  x = {x},  "
        f"converged={info['converged']}, iters={info['iterations']}"
    )
    print()
    print(f"|x - x*|_inf = {np.abs(x - expected).max():.2e}")
    ok = np.allclose(x, expected, atol=1e-5, rtol=1e-7)
    print(f"np.allclose(x, expected, atol=1e-5): {ok}")
    assert ok
    return x, info, expected


def _run_case_parallel_impl(
    case: ValidityCase, parallel_impl: str
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    _ensure_spawn()
    expected = _case_expected(case)
    _print_header(case, expected)
    common: Dict[str, Any] = dict(
        learning_rate=case.learning_rate,
        max_iterations=case.max_iterations,
        tolerance=case.tolerance,
    )

    x, info = DataParallelGradientDescent(
        **common, n_workers=N_WORKERS, parallel_impl=parallel_impl
    ).fit(
        case.x0,
        partial_grad_fn=finite_sum_mean_squared_partial_grad,
        c=case.c,
    )
    print(
        f"Паралельна ({N_WORKERS} workers, impl={parallel_impl}):  x = {x},  "
        f"converged={info['converged']}, iters={info['iterations']}"
    )
    print()
    print(f"|x - x*|_inf = {np.abs(x - expected).max():.2e}")
    ok = np.allclose(x, expected, atol=1e-5, rtol=1e-7)
    print(f"np.allclose(x, expected, atol=1e-5): {ok}")
    assert ok
    return x, info, expected


def run_sequential() -> None:
    for case in build_validity_cases():
        _run_case_sequential(case)
    _run_analytic_functions_sequential()


def run_parallel(parallel_impl: str = "map") -> None:
    _run_parallel_stress_only(parallel_impl=parallel_impl)


def run_all() -> None:
    run_sequential()
    _run_parallel_stress_compare()
    run_compare_parallel_impls()


def run_compare_parallel_impls() -> None:
    _ensure_spawn()
    case = build_parallel_stress_case()
    print("\=== map and recursive ===")
    x_map, _, expected = _run_case_parallel_impl(case, "map")
    x_rec, _, _ = _run_case_parallel_impl(case, "recursive")
    agree = np.max(np.abs(x_map - x_rec))
    ok_map = np.allclose(x_map, expected, atol=2e-5, rtol=1e-6)
    ok_rec = np.allclose(x_rec, expected, atol=2e-5, rtol=1e-6)
    ok_agree = np.allclose(x_map, x_rec, atol=1e-8, rtol=1e-7)
    print(
        f"max |x_map - x_recursive| = {agree:.2e}; "
        f"ok_map={ok_map}, ok_recursive={ok_rec}, agree={ok_agree}"
    )
    print("-" * 60)
    assert ok_map
    assert ok_rec
    assert ok_agree


def _run_analytic_functions_sequential() -> None:
    for case in build_analytic_function_cases():
        print(f"Кейс: {case.name}")
        x, info = SequentialGradientDescent(
            learning_rate=case.learning_rate,
            max_iterations=case.max_iterations,
            tolerance=case.tolerance,
        ).fit(case.x0, case.grad_fn)
        err = np.max(np.abs(x - case.expected))
        ok = np.allclose(x, case.expected, atol=1e-5, rtol=1e-7)
        print(
            f"  x={x}, expected={case.expected}, "
            f"iters={info['iterations']}, converged={info['converged']}, err={err:.2e}, ok={ok}"
        )
        assert ok


def _run_analytic_functions_via_parallel_solver() -> None:
    for case in build_analytic_function_cases():
        print(f"Кейс: {case.name}")
        x, info = DataParallelGradientDescent(
            learning_rate=case.learning_rate,
            max_iterations=case.max_iterations,
            tolerance=case.tolerance,
            n_workers=N_WORKERS,
        ).fit(case.x0, grad_fn=case.grad_fn)
        err = np.max(np.abs(x - case.expected))
        ok = np.allclose(x, case.expected, atol=1e-5, rtol=1e-7)
        print(
            f"  x={x}, expected={case.expected}, "
            f"iters={info['iterations']}, converged={info['converged']}, "
            f"parallel={info.get('parallel')}, err={err:.2e}, ok={ok}"
        )
        assert ok


def _run_analytic_functions_both() -> None:
    print("\n=== Прості функції іншого типу: узгодженість seq vs parallel-solver ===")
    for case in build_analytic_function_cases():
        print(f"Кейс: {case.name}")
        seq_solver = SequentialGradientDescent(
            learning_rate=case.learning_rate,
            max_iterations=case.max_iterations,
            tolerance=case.tolerance,
        )
        par_solver = DataParallelGradientDescent(
            learning_rate=case.learning_rate,
            max_iterations=case.max_iterations,
            tolerance=case.tolerance,
            n_workers=N_WORKERS,
        )
        x_seq, _ = seq_solver.fit(case.x0, case.grad_fn)
        x_par, _ = par_solver.fit(case.x0, grad_fn=case.grad_fn)

        ok_expected_seq = np.allclose(x_seq, case.expected, atol=1e-5, rtol=1e-7)
        ok_expected_par = np.allclose(x_par, case.expected, atol=1e-5, rtol=1e-7)
        ok_agree = np.allclose(x_seq, x_par, atol=1e-10, rtol=1e-9)
        print(
            f"  |x_seq-x*|_inf={np.max(np.abs(x_seq-case.expected)):.2e}, "
            f"|x_par-x*|_inf={np.max(np.abs(x_par-case.expected)):.2e}, "
            f"max|x_seq-x_par|={np.max(np.abs(x_seq-x_par)):.2e}"
        )
        print(
            f"  ok_seq={ok_expected_seq}, ok_par={ok_expected_par}, agree={ok_agree}"
        )
        assert ok_expected_seq
        assert ok_expected_par
        assert ok_agree


def _run_parallel_stress_only(parallel_impl: str = "map") -> None:
    case = build_parallel_stress_case()
    expected = _case_expected(case)
    print(f"Кейс: {case.name}, impl={parallel_impl}")
    x_par, info_par, _ = _run_case_parallel_impl(case, parallel_impl)
    err_par = np.max(np.abs(x_par - expected))
    ok_par = np.allclose(x_par, expected, atol=2e-5, rtol=1e-6)
    print(f"  n_workers_used={info_par.get('n_workers_used')}, err_par={err_par:.2e}, ok_par={ok_par}")
    assert ok_par


def _run_parallel_stress_compare() -> None:
    case = build_parallel_stress_case()
    expected = _case_expected(case)
    print(f"Кейс: {case.name}")

    common: Dict[str, Any] = dict(
        learning_rate=case.learning_rate,
        max_iterations=case.max_iterations,
        tolerance=case.tolerance,
    )
    x_seq, info_seq = SequentialGradientDescent(**common).fit(
        case.x0, finite_sum_mean_squared_grad, c=case.c
    )
    x_par, info_par = DataParallelGradientDescent(**common, n_workers=N_WORKERS).fit(
        case.x0,
        partial_grad_fn=finite_sum_mean_squared_partial_grad,
        c=case.c,
    )

    err_seq = np.max(np.abs(x_seq - expected))
    err_par = np.max(np.abs(x_par - expected))
    agree = np.max(np.abs(x_seq - x_par))
    ok_seq = np.allclose(x_seq, expected, atol=2e-5, rtol=1e-6)
    ok_par = np.allclose(x_par, expected, atol=2e-5, rtol=1e-6)
    ok_agree = np.allclose(x_seq, x_par, atol=1e-8, rtol=1e-7)
    print(
        f"  seq_iters={info_seq['iterations']}, par_iters={info_par['iterations']}, "
        f"n_workers_used={info_par.get('n_workers_used')}"
    )
    print(f"  err_seq={err_seq:.2e}, err_par={err_par:.2e}, max|x_seq-x_par|={agree:.2e}")
    print(f"  ok_seq={ok_seq}, ok_par={ok_par}, ok_agree={ok_agree}")

    assert ok_seq
    assert ok_par
    assert ok_agree


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Перевірка GD: finite-sum MSE + прості аналітичні функції."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=("seq", "par", "all", "par-compare"),
        help=(
            "seq — лише послідовна; par — лише паралельна; "
            "all — обидві та порівняння; par-compare — map vs recursive"
        ),
    )
    parser.add_argument(
        "--parallel-impl",
        default="map",
        choices=("map", "recursive"),
        help="Реалізація паралельної версії для mode=par",
    )
    args = parser.parse_args()

    if args.mode == "seq":
        run_sequential()
    elif args.mode == "par":
        run_parallel(parallel_impl=args.parallel_impl)
    elif args.mode == "par-compare":
        run_compare_parallel_impls()
    else:
        run_all()
