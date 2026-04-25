"""
Gradient Descent Optimization Algorithms

This module implements basic and parallelized gradient descent. The gradient
is supplied by the caller. For data-parallel finite-sum objectives, pass
``partial_grad_fn`` and ``c`` (or ``n_terms``) so partial gradients can be
aggregated across workers.
"""

from __future__ import annotations

import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Dict, List, Optional, Tuple

PartialGradFn = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray], int], np.ndarray]
GradFn = Callable[..., np.ndarray]


def finite_sum_mean_squared_grad(x: np.ndarray, c: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Gradient of f(x) = (1/N) * sum_i ||x - c_i||^2 (same shape as x).
    Requires ``c``.
    """
    if c is None:
        raise ValueError("finite_sum_mean_squared_grad requires c")
    c = np.asarray(c, dtype=float)
    n = c.shape[0]
    x = np.asarray(x, dtype=float)
    inv_n = 1.0 / n
    grad = np.zeros_like(x, dtype=float)
    for i in range(n):
        grad += inv_n * 2.0 * (x - c[i])
    return grad


def finite_sum_mean_squared_partial_grad(
    x: np.ndarray,
    indices: np.ndarray,
    c: Optional[np.ndarray],
    n_total: int,
) -> np.ndarray:
    """Partial gradient for terms i in ``indices`` (same objective as above)."""
    x = np.asarray(x, dtype=float)
    if c is None:
        raise ValueError("finite_sum_mean_squared_partial_grad requires c")
    c = np.asarray(c, dtype=float)
    if len(indices) == 0:
        return np.zeros_like(x, dtype=float)
    s = np.zeros_like(x, dtype=float)
    inv_n = 1.0 / n_total
    for i in indices:
        s += inv_n * 2.0 * (x - c[i])
    return s


def _partial_grad_call(args: Tuple) -> np.ndarray:
    """Worker for Pool; ``partial_grad_fn`` must be a picklable top-level function."""
    x, indices, c, n_total, partial_grad_fn = args
    idx = np.asarray(indices, dtype=int)
    return partial_grad_fn(np.asarray(x, dtype=float), idx, c, n_total)


class _AsyncPairSum:
    """Lazy pairwise sum over asynchronous worker results."""

    def __init__(self, left: Any, right: Any):
        self.left = left
        self.right = right

    def get(self) -> np.ndarray:
        return np.asarray(self.left.get(), dtype=float) + np.asarray(
            self.right.get(), dtype=float
        )


def _recursive_async_sum(pool: Pool, tasks: List[Tuple]) -> Any:
    """
    Submit tasks recursively and combine results as a reduction tree.
    """
    if not tasks:
        raise ValueError("tasks must not be empty")
    if len(tasks) == 1:
        return pool.apply_async(_partial_grad_call, (tasks[0],))
    mid = len(tasks) // 2
    left = _recursive_async_sum(pool, tasks[:mid])
    right = _recursive_async_sum(pool, tasks[mid:])
    return _AsyncPairSum(left, right)


class SequentialGradientDescent:
    def __init__(
        self,
        learning_rate: float,
        max_iterations: int,
        tolerance: float,
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def fit(
        self,
        x0: np.ndarray,
        grad_fn: GradFn,
        c: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        x = np.array(x0, dtype=float, copy=True)
        converged = False
        iterations = 0
        final_grad_norm = 0.0

        for iteration in range(self.max_iterations):
            grad = grad_fn(x, c)
            grad = np.asarray(grad, dtype=float)
            final_grad_norm = float(np.linalg.norm(grad))
            if final_grad_norm < self.tolerance:
                converged = True
                iterations = iteration + 1
                break
            x = x - self.learning_rate * grad
        else:
            iterations = self.max_iterations

        info: Dict[str, Any] = {
            "converged": converged,
            "iterations": iterations,
            "final_gradient_norm": final_grad_norm,
        }
        if c is not None:
            info["n_terms"] = int(np.asarray(c).shape[0])
        return x, info


class DataParallelGradientDescent:
    def __init__(
        self,
        learning_rate: float,
        max_iterations: int,
        tolerance: float,
        n_workers: Optional[int] = None,
        parallel_impl: str = "map",
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_workers = n_workers if n_workers is not None else cpu_count()
        self.parallel_impl = parallel_impl

    def fit(
        self,
        x0: np.ndarray,
        *,
        grad_fn: Optional[GradFn] = None,
        partial_grad_fn: Optional[PartialGradFn] = None,
        c: Optional[np.ndarray] = None,
        n_terms: Optional[int] = None,
        parallel_impl: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        If ``partial_grad_fn`` is set and ``c`` or ``n_terms`` defines the sum
        size, gradients are computed in parallel over index chunks (data
        parallelism). ``parallel_impl='map'`` uses ``Pool.map``; ``'recursive'``
        submits jobs recursively and reduces partial sums as a binary tree.
        Otherwise ``grad_fn`` must be provided and the step is sequential
        (no worker pool).
        """
        x = np.array(x0, dtype=float, copy=True)
        impl = self.parallel_impl if parallel_impl is None else parallel_impl
        if impl not in {"map", "recursive"}:
            raise ValueError("parallel_impl must be 'map' or 'recursive'")

        use_parallel = partial_grad_fn is not None and (c is not None or n_terms is not None)
        if use_parallel:
            if c is not None:
                c_arr = np.asarray(c, dtype=float)
                n = int(c_arr.shape[0])
                if n_terms is not None and n != int(n_terms):
                    raise ValueError("n_terms must match len(c) when both are given")
            else:
                c_arr = None
                if n_terms is None:
                    raise ValueError("partial_grad_fn requires c or n_terms")
                n = int(n_terms)
            if n < 1:
                raise ValueError("n_terms must be at least 1")

            n_proc = min(self.n_workers, max(1, n))
            chunks = [ch for ch in np.array_split(np.arange(n), n_proc) if len(ch) > 0]

            converged = False
            iterations = 0
            final_grad_norm = 0.0

            with Pool(processes=len(chunks)) as pool:
                for iteration in range(self.max_iterations):
                    tasks = [
                        (x, ch.tolist(), c_arr, n, partial_grad_fn) for ch in chunks
                    ]
                    if impl == "map":
                        parts = pool.map(_partial_grad_call, tasks)
                        grad = sum(parts)
                    else:
                        grad = _recursive_async_sum(pool, tasks).get()
                    final_grad_norm = float(np.linalg.norm(grad))
                    if final_grad_norm < self.tolerance:
                        converged = True
                        iterations = iteration + 1
                        break
                    x = x - self.learning_rate * grad
                else:
                    iterations = self.max_iterations
                    tasks = [
                        (x, ch.tolist(), c_arr, n, partial_grad_fn) for ch in chunks
                    ]
                    if impl == "map":
                        parts = pool.map(_partial_grad_call, tasks)
                        grad = sum(parts)
                    else:
                        grad = _recursive_async_sum(pool, tasks).get()
                    final_grad_norm = float(np.linalg.norm(grad))

            info: Dict[str, Any] = {
                "converged": converged,
                "iterations": iterations,
                "final_gradient_norm": final_grad_norm,
                "n_terms": n,
                "n_workers_used": len(chunks),
                "parallel": True,
                "parallel_impl": impl,
            }
            return x, info

        if grad_fn is None:
            raise ValueError(
                "Provide partial_grad_fn with c or n_terms for parallel mode, "
                "or grad_fn for sequential steps."
            )

        converged = False
        iterations = 0
        final_grad_norm = 0.0

        for iteration in range(self.max_iterations):
            grad = grad_fn(x, c)
            grad = np.asarray(grad, dtype=float)
            final_grad_norm = float(np.linalg.norm(grad))
            if final_grad_norm < self.tolerance:
                converged = True
                iterations = iteration + 1
                break
            x = x - self.learning_rate * grad
        else:
            iterations = self.max_iterations

        info = {
            "converged": converged,
            "iterations": iterations,
            "final_gradient_norm": final_grad_norm,
            "parallel": False,
        }
        if c is not None:
            info["n_terms"] = int(np.asarray(c).shape[0])
        return x, info
