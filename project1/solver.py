import numpy as np
import matplotlib.pyplot as plt

"""

This program solves Initial Value Problems (IVP).
We support three numerical meothds: Euler, Rk2, and Rk4

Example Usage:

    def func(t,y,a,b,c):
        yderive = np.zeros(len(y))
        yderive[0] = 0
        yderive[1] = a, ...
        return yderive

    y0  = [0,1]
    t_span = (0,1)
    t_eval =np.linspace(0,1,100)

    sol = solve_ivp(func, t_span, y0, 
                    method="RK4",t_eval=t_eval, args=(K,M))


    See `solve_ivp` for detailed description. 

Author: Kuo-Chuan Pan, NTHU 2022.10.06
                            2024.03.08
For the course, computational physics

"""


def solve_ivp(func, t_span, y0, method, t_eval, args):
    """
    Solve Initial Value Problems.

    :param func: a function to describe the derivative of the desired function
    :param t_span: 2-tuple of floats. the time range to compute the IVP, (t0, tf)
    :param y0: an array. The initial state
    :param method: string. Numerical method to compute.
                   We support "Euler", "RK2" and "RK4".
    :param t_eval: array_like. Times at which to store the computed solution,
                   must be sorted and lie within t_span.
    :param *args: extra arguments for the derive func.

    :return: array_like. solutions.

    Note: the structe of this function is to mimic the scipy.integrate
          In the numerical scheme we designed, we didn't check the consistentcy between
          t_span and t_eval. Be careful.

    """

    sol = np.zeros((len(y0), len(t_eval)))  # define the shape of the solution

    # Evaluation
    for i, t in enumerate(t_eval):  # i:index, t=t_eval[i]
        if i == 0:
            time = t
            y = y0
        else:
            dt = t - time  # dt may be different.
            y = _update(func, y, dt, time, method, *args)
            # Use t=time and y(notice that y is at t=time now) to evaluate next y.
        sol[:, i] = y
        time = t  # Shift to next moment.

    return sol


def _update(derive_func, y0, dt, t, method, *args):
    """
    Update the IVP with different numerical method

    :param derive_func: the derivative of the function y'
    :param y0: the initial conditions at time t
    :param dt: the time step dt
    :param t: the time
    :param method: the numerical method
    :param *args: extral parameters for the derive_func

    :return: the next step condition y

    """

    if method == "Euler":
        ynext = _update_euler(derive_func, y0, dt, t, *args)
    elif method == "RK2":
        ynext = _update_rk2(derive_func, y0, dt, t, *args)
    elif method == "RK4":
        ynext = _update_rk4(derive_func, y0, dt, t, *args)
    else:
        print("Error: mysolve doesn't supput the method", method)
        quit()
    return ynext


def _update_euler(derive_func, y0, dt, t, *args):
    """
    Update the IVP with the Euler's method

    :return: the next step solution y

    """

    y = y0 + dt * derive_func(t, y0, *args)

    return y  # <- change here. just a placeholder


def _update_rk2(derive_func, y0, dt, t, *args):
    """
    Update the IVP with the RK2 method

    :return: the next step solution y
    """

    k1 = derive_func(t, y0, *args)
    k2 = derive_func(t + dt, y0 + dt * k1, *args)
    y = y0 + 0.5 * dt * (k1 + k2)

    return y  # <- change here. just a placeholder


def _update_rk4(derive_func, y0, dt, t, *args):
    """
    Update the IVP with the RK4 method

    :return: the next step solution y
    """

    k1 = derive_func(t, y0, *args)
    k2 = derive_func(t + 0.5 * dt, y0 + 0.5 * dt * k1, *args)
    k3 = derive_func(t + 0.5 * dt, y0 + 0.5 * dt * k2, *args)
    k4 = derive_func(t + dt, y0 + dt * k3, *args)
    y = y0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y  # <- change here. just a placeholder


if __name__ == "__main__":

    """

    Testing solver.solve_ivp()

    Kuo-Chuan Pan 2022.10.07

    """

    def oscillator(t, y, K, M):
        """
        The derivate function for an oscillator
        In this example, we set

        y[0] = x
        y[1] = v

        yderive[0] = x' = v
        yderive[1] = v' = a

        :param t: the time
        :param y: the initial condition y
        :param K: the spring constant
        :param M: the mass of the oscillator

        """
        yf = np.zeros(np.shape(y))
        yf[0] = y[1]
        yf[1] = -K / M * y[0]

        return yf  # <- change here. just a placeholder

    t_span = (0, 10)
    y0 = np.array([1, 0])
    t_eval = np.linspace(0, 10, 100)

    K = 1
    M = 1

    sol = solve_ivp(oscillator, t_span, y0, method="RK4", t_eval=t_eval, args=(K, M))

    plt.plot(t_eval, sol[0, :])
    plt.show()
    print("Done!")
