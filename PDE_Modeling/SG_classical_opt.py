import numpy as np
import matplotlib.pyplot as plt
from SG_solver import rbf_matrix, second_divided_difference, d2_L_W2

import os
import wandb
from tqdm import tqdm

from class_sweep_config import sweep_config

wandb.login(key="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")

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
T = 0.25



def exact_u(x, t):
    return 0.5 * (np.sin(np.pi * (x + t)) + np.sin(np.pi * (x - t)))

# time =[0.25]  # , 0.5, 0.75, 1
# for T in tqdm(time, desc="Time stepping", unit="step"):
best_Linf_error = float('inf')
best_RMS_error = float('inf')

def classical_optimization():
    global best_Linf_error, best_RMS_error

    wandb.init()
    config = wandb.config

    run_name = f"c-{config.c}_h-{config.h}_s-{config.s}_K-{config.K}_tau-{config.tau}"
    wandb.run.name = run_name

    c = config.c
    s = config.s
    h = config.h
    K = config.K
    tau = config.tau

    n = round((b - a) / h)
    if not np.isclose(a + n*h, b):
        raise ValueError("h must divide b-a exactly.")
    n = int(n)

    if n % K != 0:
        raise ValueError("n must be divisible by K because the paper defines N = n / K.")

    N = n // K
    # Nt = int(round(T / tau))
    Nt = 0

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
    U[0, :] = exact_u(x, T)
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

    u_num = U[Nt + 1, :]
    u_ex = exact_u(x, T)

    err = u_num - u_ex
    abs_err = np.abs(err)

    Linf_error = np.max(abs_err)
    RMS_error = np.sqrt(np.mean(abs_err**2))

    if Linf_error < best_Linf_error:
        best_Linf_error = Linf_error
        best_RMS_error = RMS_error
        print(f"New best Linf error: {best_Linf_error}")
        print(f"New best RMS error: {best_RMS_error}")

        with open("best_results.txt", "w") as f:
            f.write(f"Best Linf error: {best_Linf_error}\n")
            f.write(f"Best RMS error: {best_RMS_error}\n")
            f.write(f"Parameters: c={c}, s={s}, K={K}, h={h}, tau={tau}\n")

    wandb.log({
        "c": c,
        "s": s,
        "K": K,
        "h": h,
        "tau": tau,
        "RMS_error": RMS_error,
        "Linf_error": Linf_error
    })

    wandb.finish()

# ---------------------------------------------------
# Run sweep
# ---------------------------------------------------
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="SG_classical_optimization")
    wandb.agent(sweep_id, function=classical_optimization, count = 10)
    print("Sweep complete")