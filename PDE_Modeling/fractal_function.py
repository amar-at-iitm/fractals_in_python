import numpy as np

def fractalize(X, Y, alpha, n_iter):

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    N = len(X) - 1

    if type(alpha) is float:
        alpha = np.full(N - 1, alpha)

    alpha = np.asarray(alpha, dtype=float)

    if len(alpha) != N:
        raise ValueError("alpha must have length len(X)-1.")

    x0, xN = X[0], X[-1]
    y0, yN = Y[0], Y[-1]

    # Compute affine map coefficients
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)

    for n in range(N):
        xn0 = X[n]
        xn1 = X[n + 1]

        yn0 = Y[n]
        yn1 = Y[n + 1]

        a[n] = (xn1 - xn0) / (xN - x0)
        b[n] = (xN * xn0 - x0 * xn1) / (xN - x0)

        c[n] = (yn1 - yn0 - alpha[n] * (yN - y0)) / (xN - x0)
        d[n] = (xN * yn0 - x0 * yn1 - alpha[n] * (xN * y0 - x0 * yN)) / (xN - x0)

    # Affine map
    def w(n, point):
        x, y = point
        return np.array([
            a[n] * x + b[n],
            alpha[n] * y + c[n] * x + d[n]
        ])

    # Initial points
    current_points = np.column_stack((X, Y))

    # Iterate the IFS
    #----------------------------------------------
    for _ in range(n_iter):
        new_points = []

        for n in range(N):
            for point in current_points:
                new_points.append(w(n, point))

        current_points = np.asarray(new_points)
    #----------------------------------------------



    # Sort by x-coordinate
    order = np.argsort(current_points[:, 0])
    # print(order)
    current_points = current_points[order]
    # print(current_points)

    x = current_points[:,0]
    r = current_points[:,1]
    # # Remove duplicate x-values
    # x_unique, idx = np.unique(current_points[:, 0], return_index=True)
    # y_unique = current_points[idx, 1]

    # # Interpolate onto the original X grid
    # fractal_Y = np.interp(X, x_unique, y_unique)

    return {
            "partition": x,
            "values": r,
        }

    # return x, r