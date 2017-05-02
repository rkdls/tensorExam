import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    symbols = ['SPY', 'FB', 'GLD']
    df = get_data(symbols, dates)
    # plot_data(df)

    daily_returns = compute_daily_returns(df)

    # daily_returns.plot(kind='scatter', x='SPY', y='FB')
    beta_SPY, alpha_FB = np.polyfit(daily_returns['SPY'], daily_returns['FB'], deg=1)
    # print(beta_SPY, alpha_FB)
    # plt.plot(daily_returns['SPY'], beta_SPY * daily_returns['SPY'] + alpha_FB, '-', color='r')
    # plt.show()
    daily_returns.plot(kind='scatter', x='SPY', y='GLD')
    beta_SPY, alpha_GLD = np.polyfit(daily_returns['SPY'], daily_returns['GLD'], deg=1)
    plt.plot(daily_returns['SPY'], beta_SPY * daily_returns['SPY'] + alpha_GLD, '-', color='r')
    print(beta_SPY, alpha_FB)

    print(daily_returns.corr(method='pearson'))
    # plot_data(daily_returns, title='daily_returns', ylabel='daily returns')
    plt.show()


run_stock()
