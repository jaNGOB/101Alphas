import utils as u
import pandas as pd
import numpy as np

 
# Alpha01
def alpha001(df):
    """
    Alpha#1
    (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5) 

    :param df: dataframe
    :return: 
    """
    pass
    
def alpha002(df):
    """
    Alpha#2
    (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    """
    tmp_1 = u.rank(u.delta(np.log(df.volume), 2))
    tmp_2 = u.rank(((df.close - df.open) / df.open))
    return (-1 * u.corr(tmp_1, tmp_2, 6))


def alpha003(df):
    """
    Alpha#3
    (-1 * correlation(rank(open), rank(volume), 10))
    """
    return (-1 * u.corr(u.rank(df.open), u.rank(df.volume), 10))


def alpha004(df):
    """
    Alpha #4
    (-1 * Ts_Rank(rank(low), 9))
    """
    return (-1 * u.ts_rank(u.rank(df.low), 9))