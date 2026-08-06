"""Overnight sweep for f_alpha in fractal_initial_problem.ipynb.

This script keeps the notebook parameters fixed and searches nonzero
second-derivative fractal scale vectors of the symmetric form

    [a, b, c, c, b, a]

It writes every result immediately to CSV so a long run can be interrupted
without losing completed candidates.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
import time
import types
from pathlib import Path

import numpy as np


# Some project modules import matplotlib only for plotting side effects. The
# sweep itself does not need plotting, so keep it runnable in minimal envs.
if "matplotlib" not in sys.modules:
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub

from alpha_fractal_function import alpha_fractalize, alpha_fractalize_second_derivative
from fractal_SG_solver_new import H5_dd, ddphi, pointwise_fractal
from SG_solver import varphi


# Parameters copied from fractal_initial_problem.ipynb.
a = -1.0
b = 1.0
K = 8
s = 0.8
c_shape = 0.027
h = 0.01
tau = 0.01
T = 1.0

f_beta = [0.005, 0.0025, 0.0025, 0.005]
initial_fractal_iter = 4

f_alpha_subintervals = 6
f_alpha_iter = 2


def f(x: np.ndarray) -> np.ndarray:
    y = np.sin(np.pi * x)
    return np.where(np.isclose(y, 0.0, atol=1e-12), 0.0, y)


def g(x: np.ndarray) -> np.ndarray:
    return x * (1.0 - x**2)


def build_initial_profile(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sine_pi_fractal = alpha_fractalize(
        f,
        g,
        -1.0,
        1.0,
        len(f_beta),
        f_beta,
        initial_fractal_iter,
    )

    u0 = np.asarray([pointwise_fractal(xi, sine_pi_fractal) for xi in x])
    u0[0] = 0.0
    u0[-1] = 0.0

    def odd_dirichlet_extension(z: float) -> float:
        y = ((z + 1.0) % 4.0) - 1.0
        if y <= 1.0:
            return float(pointwise_fractal(y, sine_pi_fractal))
        return -float(pointwise_fractal(2.0 - y, sine_pi_fractal))

    exact = np.asarray(
        [
            0.5
            * (
                odd_dirichlet_extension(xi + T)
                + odd_dirichlet_extension(xi - T)
            )
            for xi in x
        ]
    )
    return u0, exact


def build_divided_difference_matrix(x: np.ndarray, k_idx: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(k_idx), len(x)))
    for row, kj in enumerate(k_idx):
        xm = x[kj - 1]
        x0 = x[kj]
        xp = x[kj + 1]
        denom = (x0 - xm) * (xp - x0) * (xp - xm)

        matrix[row, kj + 1] = 2.0 * (x0 - xm) / denom
        matrix[row, kj] = -2.0 * (xp - xm) / denom
        matrix[row, kj - 1] = 2.0 * (xp - x0) / denom
    return matrix


def d2_fractal_psi_matrix(x: np.ndarray, fractal_dd: dict[str, np.ndarray]) -> np.ndarray:
    p = fractal_dd["partition"]
    v = fractal_dd["values"]
    n = len(x) - 1
    psi = np.empty((len(x), len(x)))

    def q(z: float) -> float:
        return float(np.interp(z, p, v))

    for row, xx in enumerate(x):
        vals = np.empty(len(x))
        vals[0] = q(xx - x[1]) / (2.0 * (x[1] - x[0]))
        vals[1] = (
            (q(xx - x[2]) - q(xx - x[1])) / (2.0 * (x[2] - x[1]))
            - q(xx - x[1]) / (2.0 * (x[1] - x[0]))
        )

        for j in range(2, n - 1):
            vals[j] = (
                (q(xx - x[j + 1]) - q(xx - x[j])) / (2.0 * (x[j + 1] - x[j]))
                - (q(xx - x[j]) - q(xx - x[j - 1])) / (2.0 * (x[j] - x[j - 1]))
            )

        vals[n - 1] = (
            -q(xx - x[n - 1]) / (2.0 * (x[n] - x[n - 1]))
            - (q(xx - x[n - 1]) - q(xx - x[n - 2]))
            / (2.0 * (x[n - 1] - x[n - 2]))
        )
        vals[n] = q(xx - x[n - 1]) / (2.0 * (x[n] - x[n - 1]))
        psi[row, :] = vals

    return psi


def build_spatial_operator(
    x: np.ndarray,
    xk: np.ndarray,
    k_idx: np.ndarray,
    f_alpha: list[float],
) -> np.ndarray:
    fractal_dd = alpha_fractalize_second_derivative(
        ddphi,
        H5_dd,
        -2.0,
        2.0,
        f_alpha_subintervals,
        f_alpha,
        f_alpha_iter,
    )

    r = np.abs(xk[:, None] - xk[None, :])
    A = varphi(r, s)
    A_inv = np.linalg.inv(A)

    divided_difference = build_divided_difference_matrix(x, k_idx)
    phi_part = varphi(x[:, None] - xk[None, :], s)
    rbf_correction = np.sqrt(s**2 + (x[:, None] - xk[None, :]) ** 2)
    psi_dd = d2_fractal_psi_matrix(x, fractal_dd)

    # For any grid vector u:
    # alpha = A^{-1} D u,
    # u_xx = Psi'' u + (Phi - Psi'' R) alpha.
    return psi_dd + (phi_part - psi_dd @ rbf_correction) @ A_inv @ divided_difference


def solve_for_alpha(
    x: np.ndarray,
    xk: np.ndarray,
    k_idx: np.ndarray,
    u0: np.ndarray,
    exact: np.ndarray,
    f_alpha: list[float],
) -> tuple[float, float, float]:
    spatial_operator = build_spatial_operator(x, xk, k_idx, f_alpha)

    Nt = int(round(T / tau))
    u_prev = u0.copy()
    u_cur = u0 + 0.5 * tau**2 * (spatial_operator @ u0)
    u_cur[0] = 0.0
    u_cur[-1] = 0.0

    for _ in range(1, Nt):
        u_next = 2.0 * u_cur - u_prev + tau**2 * (spatial_operator @ u_cur)
        u_next[0] = 0.0
        u_next[-1] = 0.0
        u_prev, u_cur = u_cur, u_next

    abs_err = np.abs(u_cur - exact)
    linf = float(np.max(abs_err))
    paper_rms = float(np.sqrt(np.sum(abs_err**2)) / len(x))
    standard_rms = float(np.sqrt(np.sum(abs_err**2) / len(x)))
    return linf, paper_rms, standard_rms


def candidate_f_alphas() -> list[list[float]]:
    coarse_values = [
        -0.005,
        -0.003,
        -0.002,
        -0.001,
        -0.0005,
        -0.0002,
        -0.0001,
        0.0,
        0.0001,
        0.0002,
        0.0005,
        0.001,
        0.002,
        0.003,
        0.005,
    ]

    candidates = []
    for left, mid, center in itertools.product(coarse_values, repeat=3):
        alpha = [left, mid, center, center, mid, left]
        if any(abs(v) > 0.02 for v in alpha):
            continue
        if all(v == 0.0 for v in alpha):
            continue
        candidates.append(alpha)

    # Include the exact vectors you have already tried for easier comparison.
    candidates.extend(
        [
            [-0.0002, 0.0, -0.0001, -0.0001, 0.0, -0.0002],
            [0.002, 0.0, -0.0001, -0.0001, 0.0, 0.002],
            [0.0, 0.0, -0.0001, -0.0001, 0.0, 0.0],
        ]
    )

    seen = set()
    unique = []
    for alpha in candidates:
        key = tuple(alpha)
        if key not in seen:
            seen.add(key)
            unique.append(alpha)
    unique.sort(key=lambda alpha: (max(abs(v) for v in alpha), sum(abs(v) for v in alpha)))
    return unique


def read_completed(output_path: Path) -> set[tuple[float, ...]]:
    if not output_path.exists():
        return set()

    completed = set()
    with output_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            completed.add(
                tuple(float(row[f"alpha_{i}"]) for i in range(f_alpha_subintervals))
            )
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/f_alpha_sweep_results.csv"),
        help="CSV path for all sweep results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for a short test run.",
    )
    args = parser.parse_args()

    n = int(round((b - a) / h))
    if n % K != 0:
        raise ValueError("n must be divisible by K.")

    N = n // K
    x = np.linspace(a, b, n + 1)
    k_idx = np.concatenate(([1], K * np.arange(1, N - 1), [n - 1])).astype(int)
    xk = x[k_idx]

    u0, exact = build_initial_profile(x)
    candidates = candidate_f_alphas()
    if args.limit is not None:
        candidates = candidates[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(args.output)
    write_header = not args.output.exists()

    best_linf = math.inf
    best_paper_rms = math.inf
    start = time.time()

    with args.output.open("a", newline="") as handle:
        fieldnames = [
            "rank_order",
            "alpha_0",
            "alpha_1",
            "alpha_2",
            "alpha_3",
            "alpha_4",
            "alpha_5",
            "f_Linf",
            "f_RMS_paper",
            "f_RMS_standard",
            "elapsed_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for idx, alpha in enumerate(candidates, start=1):
            key = tuple(alpha)
            if key in completed:
                continue

            try:
                linf, paper_rms, standard_rms = solve_for_alpha(
                    x, xk, k_idx, u0, exact, alpha
                )
            except Exception as exc:  # keep overnight run moving
                print(f"[{idx}/{len(candidates)}] failed {alpha}: {exc}", flush=True)
                continue

            elapsed = time.time() - start
            writer.writerow(
                {
                    "rank_order": idx,
                    **{f"alpha_{i}": alpha[i] for i in range(f_alpha_subintervals)},
                    "f_Linf": linf,
                    "f_RMS_paper": paper_rms,
                    "f_RMS_standard": standard_rms,
                    "elapsed_seconds": elapsed,
                }
            )
            handle.flush()

            if linf < best_linf:
                best_linf = linf
                best_paper_rms = paper_rms
                print(
                    "[best Linf] "
                    f"{idx}/{len(candidates)} alpha={alpha} "
                    f"Linf={linf:.12g} paper_RMS={paper_rms:.12g}",
                    flush=True,
                )
            elif idx % 25 == 0:
                print(
                    f"[{idx}/{len(candidates)}] latest Linf={linf:.12g}; "
                    f"best Linf={best_linf:.12g}, best paper_RMS={best_paper_rms:.12g}",
                    flush=True,
                )

    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
