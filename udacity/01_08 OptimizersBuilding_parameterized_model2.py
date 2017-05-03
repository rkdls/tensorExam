import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo


def fit_line(data, error_func):
    """fit a line to given data, using a supplied error function.
    :parameters
    data: 2D array where each row is a point(X0,Y)
    error_func : function that computes the error between a line and observed data
    
    Returns line that minimizes the error function.
    """
    l = np.float32([0, np.mean(data[:, 1])])

    x_ends = np.float32([-5, 5])
    plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth=2.0, label="Initial guess")

    result = spo.minimize(error_func, l, args=(data,), method="SLSQP", options={"disp": True})  # display True
    return result.x


def error(line, data):
    """compute error between given line model and observed data
    line: tuple/list/array (C0, C1) where C0 is slope and C1 is Y-intercept
    data : 2D array where each row is a point(X,Y)
    
    return error as a single real value.
    """

    err = np.sum((data[:, 1] - (line[0] * data[:, 0] + line[1])) ** 2)
    return err


def run_f():
    l_orig = np.float32([4, 2])
    print("Original line: C0={}, C1={}".format(l_orig[0], l_orig[1]))
    Xorig = np.linspace(0, 10, 21)
    Yorig = l_orig[0] * Xorig + l_orig[1]
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original line")

    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label="data points")
    l_fit = fit_line(data, error)
    print('fitted line C0={}, C1={}'.format(l_fit[0], l_fit[1]))

    plt.plot(data[:, 0], l_fit[0] * data[:, 0] + l_fit[1], 'r--', label="fitted data")
    plt.legend(loc="upper right")
    plt.show()

    # print(noise)


run_f()
