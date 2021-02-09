import pandas as pd
import numpy as np

def rank(df):
    """
    Cross-sectional percentile rank.

    :param df:
    :return: 
    """
    return df.rank(axis=1, pct=True)


def stddev(df, d):
    """
    Rolling standard deviation over the last d days.

    :param df:
    :param d:
    :return:
    """
    return df.rolling(d).std()


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


