"""
Gradient Descent Optimization Algorithms

This module implements basic and parallelized gradient descent algorithms
for finding optimal values of differentiable functions.
"""

import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any
from multiprocessing import Pool, cpu_count
import time




def _partial_grad_sum(args: Tuple) -> np.ndarray:
    x, indices, c, n_total = args
    x = np.asarray(x, dtype=float)
    if len(indices) == 0:
        return np.zeros_like(x, dtype=float)
    s = np.zeros_like(x, dtype=float)
    inv_n = 1.0 / n_total
    for i in indices:
        s += inv_n * 2.0 * (x - c[i])
    return s


class SequentialFiniteSumGradientDescent:
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
        c: np.ndarray,
        x0: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        c = np.asarray(c, dtype=float)
        n = c.shape[0]
        x = np.array(x0, dtype=float, copy=True)
        inv_n = 1.0 / n
        converged = False
        iterations = 0
        final_grad_norm = 0.0

        for iteration in range(self.max_iterations):
            grad = np.zeros_like(x, dtype=float)
            for i in range(n):
                grad += inv_n * 2.0 * (x - c[i])
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
            "n_terms": n,
        }
        return x, info


class DataParallelGradientDescent:
    def __init__(
        self,
        learning_rate: float,
        max_iterations: int,
        tolerance: float,
        n_workers: Optional[int] = None,
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_workers = n_workers if n_workers is not None else cpu_count()

    def fit(
        self,
        c: np.ndarray,
        x0: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        c = np.asarray(c, dtype=float)
        n = c.shape[0]
        x = np.array(x0, dtype=float, copy=True)
        n_proc = min(self.n_workers, max(1, n))
        chunks = [ch for ch in np.array_split(np.arange(n), n_proc) if len(ch) > 0]

        converged = False
        iterations = 0
        final_grad_norm = 0.0

        with Pool(processes=len(chunks)) as pool:
            for iteration in range(self.max_iterations):
                tasks = [(x, ch.tolist(), c, n) for ch in chunks]
                parts = pool.map(_partial_grad_sum, tasks)
                grad = sum(parts)
                final_grad_norm = float(np.linalg.norm(grad))
                if final_grad_norm < self.tolerance:
                    converged = True
                    iterations = iteration + 1
                    break
                x = x - self.learning_rate * grad
            else:
                iterations = self.max_iterations
                tasks = [(x, ch.tolist(), c, n) for ch in chunks]
                parts = pool.map(_partial_grad_sum, tasks)
                grad = sum(parts)
                final_grad_norm = float(np.linalg.norm(grad))

        info = {
            "converged": converged,
            "iterations": iterations,
            "final_gradient_norm": final_grad_norm,
            "n_terms": n,
            "n_workers_used": len(chunks),
        }
        return x, info
