import numpy as np
from SG_solver import d2_phi, varphi, error_function 


def phi(z, c):
    return np.sqrt(c**2 + z**2)

def dphi(z, c):
    return z / np.sqrt(c**2 + z**2)

def ddphi(z, c):
    return c**2 / (c**2 + z**2)**1.5

def H5(z, xN, x1, c):
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    phi1   = phi(x1, c)
    phiN   = phi(xN, c)
    phi1d  = dphi(x1, c)
    phiNd  = dphi(xN, c)
    phi1dd = ddphi(x1, c)
    phiNdd = ddphi(xN, c)

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


def H5_dd(z, x1, xN, c):
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    phi1   = phi(x1, c)
    phiN   = phi(xN, c)
    phi1d  = dphi(x1, c)
    phiNd  = dphi(xN, c)
    phi1dd = ddphi(x1, c)
    phiNdd = ddphi(xN, c)

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



def build_fractal_second_derivative(
    x,
    c,
    f_alpha,
    n_points=2000,
    n_iter=50,
):
    x = np.asarray(x, dtype=float)

    a = x[0]
    b = x[-1]
    N = len(x) - 1

    if np.isscalar(f_alpha):
        f_alpha = np.full(N, f_alpha)
    else:
        f_alpha = np.asarray(f_alpha)

    partition = np.linspace(a, b, n_points)

    ydd = ddphi(partition, c)

    for _ in range(n_iter):

        ydd_new = np.empty_like(ydd)

        for k in range(N):

            xl = x[k]
            xr = x[k + 1]

            if k < N - 1:
                mask = (partition >= xl) & (partition < xr)
            else:
                mask = (partition >= xl) & (partition <= xr)

            u = a + (partition[mask] - xl) * (b - a) / (xr - xl)

            scale = ((b - a)/(xr - xl))**2

            ydd_u = np.interp(u, partition, ydd)

            Hdd = H5_dd(u, a, b, c)

            ydd_new[mask] = (
                ddphi(partition[mask], c)
                + f_alpha[k] * scale * (ydd_u - Hdd)
            )

        ydd = ydd_new

    return {
        "partition": partition,
        "values": ydd,
    }



#-----------------------------------------------------------------------
# Second derivative of Fractal L_W2
#-----------------------------------------------------------------------
# def d2_fractal_L_W2(i, x, f, xk, alpha, s, c, f_alpha, n_points=2000, n_iter=50):
#     #               i, x, f, xk, alpha, s, c
#     xx = x[i]

#     rbf_part = 0.0
#     for j in range(len(alpha)):
#         rbf_part += alpha[j] * varphi(xx - xk[j], s)

#     e_vals = np.zeros(len(x))
#     for p in range(len(x)):
#         e_vals[p] = error_function(p, x, f, xk, alpha, s)

#     ld_part = 0.0
#     for p in range(len(x)):
#         ld_part += e_vals[p] * d2_fractal_psi(xx, p, x, c, f_alpha, n_points, n_iter)

#     return rbf_part + ld_part


def d2_fractal_phi_pointwise(z, fractal_dd):
    return np.interp(z, fractal_dd["partition"], fractal_dd["values"])

def d2_fractal_psi(a, j, x, fractal_dd):
    """Second derivative of the fractal basis function psi_j."""

    n = len(x) - 1

    if j == 0:
        return (
            d2_fractal_phi_pointwise(a - x[1], fractal_dd)
            / (2 * (x[1] - x[0]))
        )

    if j == 1:
        return (
            (
                d2_fractal_phi_pointwise(a - x[2], fractal_dd)
                - d2_fractal_phi_pointwise(a - x[1], fractal_dd)
            )
            / (2 * (x[2] - x[1]))
            - d2_fractal_phi_pointwise(a - x[1], fractal_dd)
            / (2 * (x[1] - x[0]))
        )

    if j == n - 1:
        return (
            -d2_fractal_phi_pointwise(a - x[n - 1], fractal_dd)
            / (2 * (x[n] - x[n - 1]))
            - (
                d2_fractal_phi_pointwise(a - x[n - 1], fractal_dd)
                - d2_fractal_phi_pointwise(a - x[n - 2], fractal_dd)
            )
            / (2 * (x[n - 1] - x[n - 2]))
        )

    if j == n:
        return (
            d2_fractal_phi_pointwise(a - x[n - 1], fractal_dd)
            / (2 * (x[n] - x[n - 1]))
        )

    return (
        (
            d2_fractal_phi_pointwise(a - x[j + 1], fractal_dd)
            - d2_fractal_phi_pointwise(a - x[j], fractal_dd)
        )
        / (2 * (x[j + 1] - x[j]))
        - (
            d2_fractal_phi_pointwise(a - x[j], fractal_dd)
            - d2_fractal_phi_pointwise(a - x[j - 1], fractal_dd)
        )
        / (2 * (x[j] - x[j - 1]))
    )


def d2_fractal_L_W2(i, x, f, xk, alpha, s, fractal_dd):
    #               i, x, f, xk, alpha, s, c
    xx = x[i]

    rbf_part = 0.0
    for j in range(len(alpha)):
        rbf_part += alpha[j] * varphi(xx - xk[j], s)

    e_vals = np.zeros(len(x))
    for p in range(len(x)):
        e_vals[p] = error_function(p, x, f, xk, alpha, s)

    ld_part = 0.0
    for p in range(len(x)):
        ld_part += e_vals[p] * d2_fractal_psi(xx, p, x, fractal_dd)

    return rbf_part + ld_part