import numpy as np
import time

def is_symmetric(A, tol=1e-12):
    return np.allclose(A, A.T, atol=tol)


def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
    
def forward_substitution(L, b):

    n = len(b)

    y = np.zeros(n)

    for i in range(n):

        y[i] = ( b[i] - np.dot(L[i,:i], y[:i])) / L[i,i]

    return y


def backward_substitution(U, y):

    n = len(y)

    x = np.zeros(n)

    for i in range(n-1, -1, -1):

        x[i] = ( y[i] - np.dot(U[i,i+1:], x[i+1:])) / U[i,i]

    return x



def cholesky_solver(A, b):

    L = np.linalg.cholesky(A)

    y = forward_substitution(L, b)

    w = backward_substitution(L.T, y)

    return w


def conjugate_gradient( A, b, tol=1e-10, maxiter=None):

    n = len(b)

    if maxiter is None:
        maxiter = n

    x = np.zeros(n)

    r = b - A @ x

    p = r.copy()

    rsold = np.dot(r, r)

    for k in range(maxiter):

        Ap = A @ p

        alpha = rsold / np.dot(p, Ap)

        x = x + alpha * p

        r = r - alpha * Ap

        rsnew = np.dot(r, r)

        if np.sqrt(rsnew) < tol:

            return x, k + 1

        beta = rsnew / rsold

        p = r + beta * p

        rsold = rsnew

    return x, maxiter


def solve_spd(A, b, threshold=3000, cg_tol=1e-10):

    n = A.shape[0]

    if not is_symmetric(A):
        raise ValueError("Matrix is not symmetric")

    if not is_positive_definite(A):
        raise ValueError("Matrix is not SPD")

    if n < threshold:
        x = cholesky_solver(A, b)

    else:
        x, iterations = conjugate_gradient( A, b, tol=cg_tol)


    return x