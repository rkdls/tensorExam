import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo


def fit_line(data, error_func, degree=3):
    """fit a line to given data, using a supplied error function.
    :parameters
    data: 2D array where each row is a point(X0,Y)
    error_func : function that computes the error between a line and observed data
    
    Returns line that minimizes the error function.
    """
    Cguess = np.poly1d(np.ones(degree +1, dtype=np.float32))

    x = np.linspace(-5,5,21)

    plt.plot(x, np.polyval(Cguess, x), 'm--', linewidth=2.0, label="Initial guess")

    result = spo.minimize(error_func, Cguess, args=(data,), method="SLSQP", options={"disp": True})  # display True
    return np.poly1d(result.x)


def error_poly(C, data):
    """compute error between given line model and observed data
    :parameter
    C: numpy.poly1d object or equivalent array representing polynomial coefficeints
    data : 2D array where each row is a point(X,Y)
    
    Return error as a single real value.
    """

    err = np.sum((data[:, 1] - np.polyval(C, data[:0])) ** 2)
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
    l_fit = fit_line(data, error_poly)
    print('fitted line C0={}, C1={}'.format(l_fit[0], l_fit[1]))

    plt.plot(data[:, 0], l_fit[0] * data[:, 0] + l_fit[1], 'r--', label="fitted data")
    plt.legend(loc="upper right")
    plt.show()


run_f()
