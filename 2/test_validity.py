"""
Перевірка коректності GD на кількох типах задач.

Finite-sum MSE: f(x) = (1/N) * sum_i ||x - c_i||^2  ->  x* = mean(c, axis=0).
Другий stress-кейс — лінійна регресія (МНК): рядок c = [a_i | b_i]

Сценарії (від простого до складного):
1) finite-sum MSE: 1D, 2D;
2) аналітичні функції:
    - зсунутий quartic 1D,
    - зсунутий змішаний поліном 1D,
    - логарифмічна функція.
3) для паралельної частини використовуються stress-тести на великому N для finite-sum
   (MSE та лінійна МНК-ціль), з порівнянням map і recursive.

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
    finite_sum_linear_least_squares_grad,
    finite_sum_mean_squared_grad,
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
    objective_name: str = "Finite-sum MSE"
    objective_tex: str = "f(x) = (1/N) * sum_i ||x - c_i||^2"
    kind: str = "mse"
    grad_fn: Any = finite_sum_mean_squared_grad


@dataclass(frozen=True)
class AnalyticFunctionCase:
    name: str
    x0: np.ndarray
    expected: np.ndarray
    learning_rate: float
    max_iterations: int
    tolerance: float
    grad_fn: Any
    objective_name: str = "Аналітична ціль"
    objective_tex: str = ""


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
        objective_name="Finite-sum MSE",
        objective_tex="f(x) = (1/N) * sum_i (x - c_i)^2",
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
        objective_name="Finite-sum MSE",
        objective_tex="f(x) = (1/N) * sum_i ||x - c_i||^2",
    )
    return [case_1d, case_2d]


def shifted_quartic_1d_grad(
    x: np.ndarray,
    c: Optional[np.ndarray] = None,
) -> np.ndarray:
    del c
    target = 1.5
    return 4.0 * (x - target) ** 3


def shifted_mixed_poly_1d_grad(
    x: np.ndarray,
    c: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    f(x) = (x + 2)^4 + 0.5 * (x + 2)^2,  x* = -2.
    """
    del c
    z = x + 2.0
    return 4.0 * z**3 + z


def shifted_log_smooth_1d_grad(
    x: np.ndarray,
    c: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    f(x) = (log(1 + exp(x)) - log(2))^2,  x* = 0.
    """
    del c
    log_term = np.log1p(np.exp(x))
    sigma = 1.0 / (1.0 + np.exp(-x))
    return 2.0 * (log_term - np.log(2.0)) * sigma


def build_analytic_function_cases() -> List[AnalyticFunctionCase]:
    return [
        AnalyticFunctionCase(
            name="Зсунутий quartic 1D",
            x0=np.array([4.0]),
            expected=np.array([1.5]),
            learning_rate=0.04,
            max_iterations=2000,
            tolerance=1e-8,
            grad_fn=shifted_quartic_1d_grad,
            objective_name="Зсунутий quartic",
            objective_tex="f(x) = (x - 1.5)^4,  x* = 1.5",
        ),
        AnalyticFunctionCase(
            name="Зсунутий змішаний поліном 1D",
            x0=np.array([0.0]),
            expected=np.array([-2.0]),
            learning_rate=0.005,
            max_iterations=5000,
            tolerance=1e-8,
            grad_fn=shifted_mixed_poly_1d_grad,
            objective_name="Зсунутий змішаний поліном",
            objective_tex="f(x) = (x + 2)^4 + 0.5 * (x + 2)^2,  x* = -2",
        ),
        AnalyticFunctionCase(
            name="Логарифмічна функція",
            x0=np.array([3.0]),
            expected=np.array([0.0]),
            learning_rate=0.2,
            max_iterations=2000,
            tolerance=1e-8,
            grad_fn=shifted_log_smooth_1d_grad,
            objective_name="Логарифмічна",
            objective_tex="f(x) = (log(1 + exp(x)) - log(2))^2,  x* = 0",
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
        objective_name="Finite-sum MSE",
        objective_tex="f(x) = (1/N) * sum_i ||x - c_i||^2",
        kind="mse",
        grad_fn=finite_sum_mean_squared_grad,
    )


def build_parallel_stress_linear_ls_case() -> ValidityCase:
    rng = np.random.default_rng(19)
    d = 3
    n = 20000
    a = rng.normal(loc=0.0, scale=1.0, size=(n, d))
    w_true = np.array([1.5, -0.8, 0.6], dtype=float)
    b = (a @ w_true + rng.normal(loc=0.0, scale=0.15, size=n)).reshape(n, 1)
    c = np.concatenate([a, b], axis=1)
    return ValidityCase(
        name="Stress / лінійна МНК, великий N 20000, d=3",
        c=c,
        x0=np.array([3.0, 2.0, -1.0]),
        learning_rate=0.25,
        max_iterations=400,
        tolerance=1e-7,
        objective_name="Finite-sum linear least squares",
        objective_tex="f(x) = (1/N) * sum_i (a_i^T x - b_i)^2  (останній стовпець c — b_i)",
        kind="linear_ls",
        grad_fn=finite_sum_linear_least_squares_grad,
    )


def build_parallel_stress_cases() -> List[ValidityCase]:
    return [
        build_parallel_stress_case(),
        build_parallel_stress_linear_ls_case(),
    ]


def _case_expected(case: ValidityCase) -> np.ndarray:
    c = np.asarray(case.c, dtype=float)
    if case.kind == "linear_ls":
        if c.ndim != 2 or c.shape[1] < 2:
            raise ValueError("linear_ls expects c with shape (N, d+1)")
        a_mat = c[:, : c.shape[1] - 1]
        b_vec = c[:, -1]
        x_star, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        return np.asarray(x_star, dtype=float).reshape(-1)
    if c.ndim == 1:
        return np.array([float(np.mean(c))], dtype=float)
    return np.mean(c, axis=0)


def _print_case_banner(case: ValidityCase) -> None:
    c = np.asarray(case.c, dtype=float)
    print("=" * 72)
    print(f"Ціль: {case.objective_name}")
    print(f"  {case.objective_tex}")
    print(f"Кейс: {case.name}")
    print(f"  c.shape = {c.shape},  N = {len(c)}")
    print(f"  x0 = {case.x0}")
    print(f"  grad_fn = {getattr(case.grad_fn, '__name__', str(case.grad_fn))}")
    print("  chunk_grad_fn = grad_fn (same function, chunk mode via indices)")


def _print_expected_footer(case: ValidityCase, expected: np.ndarray) -> None:
    if case.kind == "linear_ls":
        print(f"  Очікуваний мінімум (МНК / normal equations): x* = {expected}")
    else:
        print(f"  Очікуваний мінімум: x* = mean(c) = {expected}")
    print()


def _print_header(case: ValidityCase, expected: np.ndarray) -> None:
    _print_case_banner(case)
    _print_expected_footer(case, expected)


def _run_case_sequential(case: ValidityCase) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    expected = _case_expected(case)
    _print_header(case, expected)
    common: Dict[str, Any] = dict(
        learning_rate=case.learning_rate,
        max_iterations=case.max_iterations,
        tolerance=case.tolerance,
    )

    x, info = SequentialGradientDescent(**common).fit(case.x0, case.grad_fn, c=case.c)
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
        grad_fn=case.grad_fn,
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
        grad_fn=case.grad_fn,
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
    for case in build_parallel_stress_cases():
        print(f"\n=== Stress parallel ({parallel_impl}): {case.name} ===")
        _run_parallel_stress_only(case, parallel_impl=parallel_impl)


def run_all() -> None:
    run_sequential()
    for case in build_parallel_stress_cases():
        _run_parallel_stress_compare(case)
    run_compare_parallel_impls()


def run_compare_parallel_impls() -> None:
    _ensure_spawn()
    for case in build_parallel_stress_cases():
        print(f"\n=== map vs recursive: {case.name} ===")
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
        print(f"Ціль: {case.objective_name}")
        if case.objective_tex:
            print(f"  {case.objective_tex}")
        print(f"Кейс: {case.name}")
        print(f"  grad_fn = {getattr(case.grad_fn, '__name__', str(case.grad_fn))}")
        print(f"  x* (аналітично) = {case.expected}")
        print()
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
        print(f"Ціль: {case.objective_name}")
        if case.objective_tex:
            print(f"  {case.objective_tex}")
        print(f"Кейс: {case.name}")
        print(f"  grad_fn = {getattr(case.grad_fn, '__name__', str(case.grad_fn))}")
        print(f"  x* (аналітично) = {case.expected}")
        print()
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
        print(f"Ціль: {case.objective_name}")
        if case.objective_tex:
            print(f"  {case.objective_tex}")
        print(f"Кейс: {case.name}")
        print(f"  grad_fn = {getattr(case.grad_fn, '__name__', str(case.grad_fn))}")
        print(f"  x* (аналітично) = {case.expected}")
        print()
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


def _run_parallel_stress_only(case: ValidityCase, parallel_impl: str = "map") -> None:
    print(f"Кейс: {case.name}, impl={parallel_impl}")
    x_par, info_par, expected = _run_case_parallel_impl(case, parallel_impl)
    err_par = np.max(np.abs(x_par - expected))
    ok_par = np.allclose(x_par, expected, atol=2e-5, rtol=1e-6)
    print(f"  n_workers_used={info_par.get('n_workers_used')}, err_par={err_par:.2e}, ok_par={ok_par}")
    assert ok_par


def _run_parallel_stress_compare(case: ValidityCase) -> None:
    print(f"Кейс: {case.name}")

    common: Dict[str, Any] = dict(
        learning_rate=case.learning_rate,
        max_iterations=case.max_iterations,
        tolerance=case.tolerance,
    )
    expected = _case_expected(case)
    _print_header(case, expected)

    x_seq, info_seq = SequentialGradientDescent(**common).fit(case.x0, case.grad_fn, c=case.c)
    x_par, info_par = DataParallelGradientDescent(**common, n_workers=N_WORKERS).fit(
        case.x0,
        grad_fn=case.grad_fn,
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
        description=(
            "Перевірка GD: finite-sum (MSE, лінійна МНК) + прості аналітичні функції."
        )
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
