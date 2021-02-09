import pandas as pd
import numpy as np

def rank(df):
    """
    Cross-sectional percentile rank.

    :param df:
    :return: 
    """
    return df.rank(pct=True)


def stddev(df, d):
    """
    Rolling standard deviation over the last d days.

    :param df:
    :param d:
    :return:
    """
    return df.rolling(d).std()


def delta(df, d):
    """
    today’s value of x minus the value of x d days ago
    """
    return df - df.shift(d)


def corr(x, y, d):
    """
    time-serial correlation of x and y for the past d days 
    """
    return x.rolling(d).corr(y)


def ts_max(df, d=10):
    """
    The rolling max over the last d days. 

    :param df: data frame containing prices
    :param d: number of days to look back (rolling window)
    :return: Pandas series
    """
    return df.rolling(d).max()


def ts_min(df, d=10):
    """
    The rolling min over the last d days. 

    :param df: data frame containing prices
    :param d: number of days to look back (rolling window)
    :return: Pandas series
    """
    return df.rolling(d).min()


def ts_argmax(df, d):
    """
    Gets the day, ts_max(x, d) occured on.

    :param df: dataframe
    :param d: number of days to look back (rolling window)
    :return: Pandas Series
    """
    return df.rolling(d).apply(np.argmax).add(1)


