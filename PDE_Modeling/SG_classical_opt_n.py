import numpy as np
import matplotlib.pyplot as plt
from SG_solver import rbf_matrix, second_divided_difference, d2_L_W2

import os
from tqdm import tqdm

a = -1.0
b = 1.0
#######################
# K = 8
# s = 0.8
# c = 0.027 
# ################
# h = 0.01
# tau = 0.01
################



def exact_u(x, t):
    return 0.5 * (np.sin(np.pi * (x + t)) + np.sin(np.pi * (x - t)))

def classical_optimization(c, s, K, h, tau, n, T):
    global best_Linf_error, best_RMS_error


    N = n // K
    Nt = int(round(T / tau))
    # Nt = 0

    x = np.linspace(a, b, n + 1)

    k_idx = np.concatenate(([1], K * np.arange(1, N - 1), [n - 1])).astype(int)

    xk = x[k_idx]
    xkj = []
    for j in range(1, N+1):
        if j == 1:
            value = float(x[1])
            xkj.append(value)
        elif 1 < j < N:
            value = a + ((j - 1) * (b - a)) / N
            xkj.append(value)
        elif j == N:
            value = float(x[n - 1])
            xkj.append(value)

    U = np.zeros((Nt + 2, len(x)))
    U[0, :] = exact_u(x, 0)
    U[0, 0] = 0.0
    U[0, -1] = 0.0

    A0 = rbf_matrix(xk, s)
    rhs0 = np.asarray([second_divided_difference(x, U[0, :], kj) for kj in k_idx])
    alpha0 = np.linalg.solve(A0, rhs0)

    uxx0 = np.zeros(len(x))
    for i in range(len(x)):
        uxx0[i] = d2_L_W2(i, x, U[0, :], xk, alpha0, s, c)

    U[1, :] = U[0, :] + 0.5 * tau**2 * uxx0
    U[1, 0] = 0.0
    U[1, -1] = 0.0

    #------------------------------------
    for d in tqdm(range(1, Nt + 1), desc="Time stepping", unit="step"):
        A = rbf_matrix(xk, s)
        rhs_d = np.asarray([second_divided_difference(x, U[d], kj) for kj in k_idx])
        alpha = np.linalg.solve(A, rhs_d)

        uxx = np.zeros(len(x))
        for i in range( len(x)):
            uxx[i] = d2_L_W2(i, x, U[d], xk, alpha, s, c)


        U[d + 1, :] = 2.0 * U[d, :] - U[d - 1, :] + tau**2 * uxx

        # Dirichlet boundary conditions
        U[d + 1, 0] = 0.0
        U[d + 1, -1] = 0.0
    #-----------------------------------


    u_num = U[Nt + 1, :]
    u_ex = exact_u(x, T)

    err = u_num - u_ex
    abs_err = np.abs(err)

    Linf_error = np.max(abs_err)
    RMS_error = np.sqrt(np.mean(abs_err**2))

    print(f"Parameters: c={c}, s={s}, K={K}, h={h}, tau={tau}")
    print(f"Linf error: {Linf_error}")
    print(f"RMS error: {RMS_error}")

    if Linf_error < best_Linf_error:
        print("Updating best errors...")
        best_Linf_error = Linf_error
        best_RMS_error = RMS_error
        print(f"New best Linf error: {best_Linf_error}")
        print(f"New best RMS error: {best_RMS_error}")

        file_name = f"Local_best_results_T_{T}.txt"
        with open(file_name, "w") as f:
            f.write(f"At time T={T}:\n")
            f.write(f"Best Linf error: {best_Linf_error}\n")
            f.write(f"Best RMS error: {best_RMS_error}\n")
            f.write(f"Parameters: c={c}, s={s}, K={K}, h={h}, tau={tau}\n")




T = 0.25

C = [0.02, 0.027, 0.03, 0.035, 0.04]  # 0.02, 0.025, 0.027,
S = [0.9, 0.7, 0.8, 0.6]
k = [4, 5, 8, 10]  #
H = [0.1, 0.05, 0.02, 0.01]  #
Tau = [0.01]  # 0.01, 0.008

time =[0.25, 0.5, 0.75, 1]  # 
for T in tqdm(time, desc="Time stepping", unit="step"):
    best_Linf_error = float('inf')
    best_RMS_error = float('inf')
    print(f"Started Working for T={T}...")
    for c in tqdm(C, desc="Working for c", unit="value"):
        print(f"Started Working for c={c}...")
        for s in tqdm(S, desc="Working for s", unit="value"):
            print(f"Started Working for s={s}...")     
            for K in tqdm(k, desc="Working for K", unit="value"):
                print(f"Started Working for K={K}...")
                for h in tqdm(H, desc="Working for h", unit="value"):
                    n = round((b - a) / h)
                    if not np.isclose(a + n*h, b):
                        raise ValueError("h must divide b-a exactly.")
                    n = int(n)

                    if n % K != 0:
                        print(f"K={K} does not divide n={n} evenly, skipping...")
                        continue  # Skip this combination if n is not divisible by K


                    for tau in tqdm(Tau, desc="Working for tau", unit="value"):

                        classical_optimization(c, s, K, h, tau, n, T)
