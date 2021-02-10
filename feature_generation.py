import utils as u
import pandas as pd
import numpy as np

 
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

def alphas005(df):
    """
    Alpha#5
    (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))) 
    """
    pass

def alpha006(df):
    """
    Alpha#6
    (-1 * correlation(open, volume, 10)) 
    """
    return (-1 * u.corr(df.open, df.volume, 10))

def alpha007(df):
    """
    Alpha#7
    ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : 
    (-1 * 1)) 
    """
    pass

def alpha008(df):
    """
    Alpha#8
    (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * 
    sum(returns, 5)), 10))))
    """
    temp1 = (u.ts_sum(df.open, 5) * u.ts_sum(df.returns, 5))
    temp2 = u.delay((u.ts_sum(df.open, 5) * u.ts_sum(df.returns, 5)), 10)
    return (-1 * u.rank(temp1 - temp2))


