import pandas as pd
import matplotlib.pyplot as plt

import importlib

get_data, plot_data = importlib.import_module('01_05Incomplete_data').get_data, importlib.import_module(
    '01_05Incomplete_data').plot_data


def compute_daily_returns(df):
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns.ix[0, :] = 0
    return daily_returns


def run_stock():
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY', 'FB']
    df = get_data(symbols, dates)
    # plot_data(df)

    daily_returns = compute_daily_returns(df)
    # plot_data(daily_returns, title='daily_returns', ylabel='daily returns')

    # daily_returns.hist(bins=20)
    # mean = daily_returns['SPY'].mean()
    # print(mean)
    # std = daily_returns['SPY'].std()
    # plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
    # plt.axvline(std, color='r', linestyle='dashed', linewidth=2)
    # plt.axvline(-std, color='r', linestyle='dashed', linewidth=2)
    # kurt = daily_returns.kurtosis()
    daily_returns['SPY'].hist(bins=20, label="SPY")
    daily_returns['FB'].hist(bins=20, label="FB")
    print(daily_returns['SPY'].kurtosis())
    print(daily_returns['FB'].kurtosis())
    print(daily_returns.kurtosis())
    plt.legend(loc="upper right")
    plt.show()


run_stock()
