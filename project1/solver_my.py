# %%
import numpy as np


def solve_ivp(func, t_span, y0, t_eval, args, method="Euler"):
    """
    A general solver to solve IVPs.
    func: function with variables t, y, and args.
    t_span: [t0, tf], initial time and final.
    y0: initial y. Need to use 1d list or ndarray(1,:).
    t_eval: seperation time of steps.
    args: parameters used in func.
    method: "Euler"(default), "RK2", and "RK4".
    return: (y(t), t), y[:,i] is evaluation of y0[i].
    """
    dt = t_eval
    t0 = t_span[0]
    tf = t_span[1] + dt

    t = np.arange(t0, tf, dt)
    y = np.zeros((np.size(t), np.size(y0)))
    y[0, :] = np.array(y0)

    if method == "Euler":
        for i in range(np.size(t) - 1):
            y[i + 1, :] = y[i, :] + dt * func(t[i], y[i, :])
        return t, y
    elif method == "RK2":
        for i in range(np.size(t) - 1):
            k1 = func(t[i], y[i, :])
            k2 = func(t[i] + dt, y[i, :] + dt * k1)
            y[i + 1, :] = y[i, :] + 0.5 * dt * (k1 + k2)
        return t, y
    elif method == "RK4":
        for i in range(np.size(t) - 1):
            k1 = func(t[i], y[i, :])
            k2 = func(t[i] + 0.5 * dt, y[i, :] + 0.5 * dt * k1)
            k3 = func(t[i] + 0.5 * dt, y[i, :] + 0.5 * dt * k2)
            k4 = func(t[i] + dt, y[i, :] + dt * k3)
            y[i + 1, :] = y[i, :] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return t, y
    else:
        print("Check your method!")
        return


# %%
