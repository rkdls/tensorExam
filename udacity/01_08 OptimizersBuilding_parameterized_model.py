import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo


def f(x):
    Y = (x - 1.5) ** 2 + 0.5
    print("X = {}, Y= {}".format(x, Y))
    return Y


def run_f():
    Xguess = 2.0
    min_result = spo.minimize(f, Xguess, method='SLSQP', options={'disp': True})
    print('minima found ')
    print('X = {}, Y={}'.format(min_result.x, min_result.fun))

    Xplot = np.linspace(0.5, 2.5, 21)
    Yplot = f(Xplot)
    plt.plot(Xplot, Yplot)
    plt.plot(min_result.x, min_result.fun, 'ro')
    plt.title("minima of an objective function")
    plt.show()

run_f()
