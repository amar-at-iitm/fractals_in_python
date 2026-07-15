import numpy as np
import matplotlib.pyplot as plt


def phi_i(z, xi, c):
    z = np.asarray(z, dtype=float)
    return np.sqrt(c**2 + (z - xi)**2)

def dphi_i(z, xi, c):
    z = np.asarray(z, dtype=float)
    return (z - xi) / np.sqrt(c**2 + (z - xi)**2)

def ddphi_i(z, xi, c):
    z = np.asarray(z, dtype=float)
    return c**2 / (c**2 + (z - xi)**2)**1.5


def H5(z, x1, xN, xi, c):
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    phi1   = phi_i(x1, xi, c)
    phiN   = phi_i(xN, xi, c)
    phi1d  = dphi_i(x1, xi, c)
    phiNd  = dphi_i(xN, xi, c)
    phi1dd = ddphi_i(x1, xi, c)
    phiNdd = ddphi_i(xN, xi, c)

    h1 = (phiN - phi1 - phi1d*dx - 0.5*phi1dd*dx**2) / dx**3
    h2 = (3*(phi1 - phiN) + 2*(phi1d + 0.5*phiNd)*dx + 0.5*phi1dd*dx**2) / dx**4
    h3 = (6*(phiN - phi1) - 3*(phi1d + phiNd)*dx + 0.5*(phiNdd - phi1dd)*dx**2) / dx**5

    dz = z - x1
    return (phi1
            + phi1d*dz
            + 0.5*phi1dd*dz**2
            + h1*dz**3
            + h2*dz**3*(z - xN)
            + h3*dz**3*(z - xN)**2)

def H5_dd(z, x1, xN, xi, c):
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    phi1   = phi_i(x1, xi, c)
    phiN   = phi_i(xN, xi, c)
    phi1d  = dphi_i(x1, xi, c)
    phiNd  = dphi_i(xN, xi, c)
    phi1dd = ddphi_i(x1, xi, c)
    phiNdd = ddphi_i(xN, xi, c)

    h1 = (phiN - phi1 - phi1d*dx - 0.5*phi1dd*dx**2) / dx**3
    h2 = (3*(phi1 - phiN) + 2*(phi1d + 0.5*phiNd)*dx + 0.5*phi1dd*dx**2) / dx**4
    h3 = (6*(phiN - phi1) - 3*(phi1d + phiNd)*dx + 0.5*(phiNdd - phi1dd)*dx**2) / dx**5

    dz = z - x1
    w = z - xN

    return (
        phi1dd
        + 6.0*h1*dz
        + h2*(6.0*dz*w + 6.0*dz**2)
        + h3*(6.0*dz*w**2 + 12.0*dz**2*w + 2.0*dz**3)
    )



def fractal_second_derivative(partition, xi, c, alpha, n_points=2000, n_iter=50):
    partition = np.asarray(partition, dtype=float)
    a = partition[0]
    b = partition[-1]
    N = len(partition) - 1

    if np.isscalar(alpha):
        alpha = np.full(N, alpha, dtype=float)
    else:
        alpha = np.asarray(alpha, dtype=float)

    if len(alpha) != N:
        raise ValueError("Length of alpha must be len(partition)-1.")
    if np.any(np.abs(alpha) >= 1):
        raise ValueError("All alpha values must satisfy |alpha_i| < 1.")

    x = np.linspace(a, b, n_points)
    ddphi = ddphi_i(x, xi, c)
    ydd = ddphi.copy()

    for _ in range(n_iter):
        ydd_new = np.empty_like(ydd)

        for k in range(N):
            x_left = partition[k]
            x_right = partition[k + 1]

            if k < N - 1:
                mask = (x >= x_left) & (x < x_right)
            else:
                mask = (x >= x_left) & (x <= x_right)

            u = a + (x[mask] - x_left) * (b - a) / (x_right - x_left)
            s2 = ((b - a) / (x_right - x_left))**2

            ydd_u = np.interp(u, x, ydd)
            Hdd_u = H5_dd(u, a, b, xi, c)

            ydd_new[mask] = ddphi_i(x[mask], xi, c) + alpha[k] * s2 * (ydd_u - Hdd_u)

        ydd = ydd_new

    return ydd
