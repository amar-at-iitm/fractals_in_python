"""Reusable Sine-Gordon MQ quasi-interpolation helpers.

These functions were copied from ``SG_solver.ipynb`` so they can be imported
from other scripts and notebooks.
"""

import numpy as np

__all__ = [
    "varphi",
    "phi",
    "psi",
    "LD_operator",
    "second_divided_difference",
    "rbf_matrix",
    "rbf_correction",
    "error_function",
    "L_W2",
    "d2_phi",
    "d2_psi",
    "d2_L_W2",
]


def varphi(r, s):
    """Eq. (7): varphi(r) = s^2 / (s^2 + r^2)^(3/2)."""
    return s**2 / (s**2 + r**2) ** 1.5


def phi(a, b, c):
    """Eq. (10): phi_i(a) = sqrt(c^2 + (a - b)^2)."""
    return np.sqrt(c**2 + (a - b) ** 2)


def psi(a, j, x, c):
    """Compute psi_j(a) from Eq. (12) on the node set x."""
    n = len(x) - 1

    if j == 0:
        return 0.5 + (phi(a, x[1], c) - (a - x[0])) / (2 * (x[1] - x[0]))

    if j == 1:
        return (
            (phi(a, x[2], c) - phi(a, x[1], c)) / (2 * (x[2] - x[1]))
            - (phi(a, x[1], c) - (a - x[0])) / (2 * (x[1] - x[0]))
        )

    if j == n - 1:
        return (
            ((x[n] - a) - phi(a, x[n - 1], c)) / (2 * (x[n] - x[n - 1]))
            - (phi(a, x[n - 1], c) - phi(a, x[n - 2], c))
            / (2 * (x[n - 1] - x[n - 2]))
        )

    if j == n:
        return 0.5 + (phi(a, x[n - 1], c) - (x[n] - a)) / (
            2 * (x[n] - x[n - 1])
        )

    return (
        (phi(a, x[j + 1], c) - phi(a, x[j], c)) / (2 * (x[j + 1] - x[j]))
        - (phi(a, x[j], c) - phi(a, x[j - 1], c)) / (2 * (x[j] - x[j - 1]))
    )


def LD_operator(a, i, x, f, c):
    """
    Wu-Schaback interpolation operator.

    (L_D f)(a) = sum_j f_j psi_j(a). The ``i`` argument is kept for
    compatibility with the notebook function signature.
    """
    d = []
    for j in range(len(x)):
        d.append(psi(a, j, x, c))
    return np.asarray(f) @ np.asarray(d)


def second_divided_difference(x, f, kj):
    """
    Approximate f''(x_kj) using the three-point nonuniform formula.

    ``kj`` must be a strictly interior grid index.
    """
    if kj <= 0 or kj >= len(x) - 1:
        raise ValueError("kj must be a strictly interior grid index.")

    xm = x[kj - 1]
    x0 = x[kj]
    xp = x[kj + 1]

    fm = f[kj - 1]
    f0 = f[kj]
    fp = f[kj + 1]

    numerator = 2.0 * ((x0 - xm) * fp - (xp - xm) * f0 + (xp - x0) * fm)
    denominator = (x0 - xm) * (xp - x0) * (xp - xm)

    return numerator / denominator


def rbf_matrix(xk, s):
    """Return A_ij = varphi(|x_{k_i} - x_{k_j}|)."""
    r = np.abs(xk[:, None] - xk[None, :])
    return varphi(r, s)


def rbf_correction(a, xk, alpha, s):
    """Return R(a) = sum_j alpha_j sqrt(s^2 + (a - x_{k_j})^2)."""
    m = []
    for j in range(len(xk)):
        m.append(np.sqrt(s**2 + (a - xk[j]) ** 2))
    return np.asarray(m) @ alpha


def error_function(i, x, f, xk, alpha, s):
    """Return e(x_i) = f(x_i) - R(x_i)."""
    return f[i] - rbf_correction(x[i], xk, alpha, s)


def L_W2(i, x, f, xk, alpha, s, c):
    """
    Wu-Schaback L_W2 operator from Eq. (17).

    L_W2 f(x_i) = sum_j alpha_j sqrt(s^2 + (x_i - x_{k_j})^2)
                  + L_D e(x_i).
    """
    rbf_part = rbf_correction(x[i], xk, alpha, s)

    e_vals = []
    for p in range(len(x)):
        e_vals.append(error_function(p, x, f, xk, alpha, s))

    ld_part = LD_operator(x[i], i, x, e_vals, c)
    return rbf_part + ld_part


def d2_phi(a, b, c):
    """Second derivative of phi_i(a) with center b."""
    return c**2 / (c**2 + (a - b) ** 2) ** 1.5


def d2_psi(a, j, x, c):
    """Second derivative of psi_j(a)."""
    n = len(x) - 1

    if j == 0:
        return d2_phi(a, x[1], c) / (2 * (x[1] - x[0]))

    if j == 1:
        return (
            (d2_phi(a, x[2], c) - d2_phi(a, x[1], c)) / (2 * (x[2] - x[1]))
            - d2_phi(a, x[1], c) / (2 * (x[1] - x[0]))
        )

    if j == n - 1:
        return (
            -d2_phi(a, x[n - 1], c) / (2 * (x[n] - x[n - 1]))
            - (d2_phi(a, x[n - 1], c) - d2_phi(a, x[n - 2], c))
            / (2 * (x[n - 1] - x[n - 2]))
        )

    if j == n:
        return d2_phi(a, x[n - 1], c) / (2 * (x[n] - x[n - 1]))

    return (
        (d2_phi(a, x[j + 1], c) - d2_phi(a, x[j], c))
        / (2 * (x[j + 1] - x[j]))
        - (d2_phi(a, x[j], c) - d2_phi(a, x[j - 1], c))
        / (2 * (x[j] - x[j - 1]))
    )


def d2_L_W2(i, x, f, xk, alpha, s, c):
    """
    Second derivative of the Wu-Schaback operator from Eq. (17A).

    (L_W2 f)''(x_i) = sum_j alpha_j varphi(x_i - x_{k_j})
                      + sum_p e(x_p) psi_p''(x_i).
    """
    xx = x[i]

    rbf_part = 0.0
    for j in range(len(alpha)):
        rbf_part += alpha[j] * varphi(xx - xk[j], s)

    e_vals = np.zeros(len(x))
    for p in range(len(x)):
        e_vals[p] = error_function(p, x, f, xk, alpha, s)

    ld_part = 0.0
    for p in range(len(x)):
        ld_part += e_vals[p] * d2_psi(xx, p, x, c)

    return rbf_part + ld_part
