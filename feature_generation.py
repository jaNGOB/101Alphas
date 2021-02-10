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

def alpha009(df):
    """
    Alpha#9
    ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : 
    ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))) 
    """
    tempd1 = u.delta(df.close, 1)
    tempmin = u.ts_min(tempd1, 5)
    tempmax = u.ts_max(tempd1, 5)
    return pd.Series(np.where(tempmin > 0, tempd1, np.where(tempmax < 0, tempd1, (-1 * tempd1))), df.index)

def alpha010(df):
    """
    Alpha#10
    rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)
    ? delta(close, 1) : (-1 * delta(close, 1)))))
    """
    tempd1 = u.delta(df.close, 1)
    tempmin = u.ts_min(tempd1, 4)
    tempmax = u.ts_max(tempd1, 4)
    return u.rank(pd.Series(np.where(tempmin > 0, tempd1, np.where(tempmax < 0, tempd1, (-1 * tempd1))), df.index))

def alpha011(df):
    """
    Alpha#11
    ((rank(ts_max((vwap - close), 3)) + 
    rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3))) 
    """
    temp1 = u.rank(u.ts_max((df.vwap - df.close), 3))
    temp2 = u.rank(u.ts_min((df.vwap - df.close), 3))
    temp3 = u.rank(u.delta(df.volume, 3))
    return temp1 + (temp2 * temp3)

def alpha012(df):
    """
    Alpha#12
    (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    """
    return (np.sign(u.delta(df.volume, 1)) * (-1 * u.delta(df.close, 1)))

def alpha013(df):
    """
    Alpha#13
    (-1 * rank(covariance(rank(close), rank(volume), 5)))
    """
    return (-1 * u.rank(u.cov(u.rank(df.close), u.rank(df.volume), 5)))

def alpha014(df):
    """
    Alpha#14
    ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)) 
    """
    return ((-1 * u.rank(u.delta(df.returns, 3))) * u.corr(df.open, df.volume, 10))

def alpha015(df):
    """
    Alpha#15
    (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)) 
    """
    return (-1 * u.ts_sum(u.corr(u.rank(df.high), u.rank(df.volume), 3), 3))

def alpha016(df):
    """
    Alpha#16
    (-1 * rank(covariance(rank(high), rank(volume), 5))) 
    """
    return (-1 * u.rank(u.cov(u.rank(df.high), u.rank(df.volume), 5)))

def alpha017(df):
    """
    Alpha#17
    (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *
    rank(ts_rank((volume / adv20), 5))) 
    """
    pass

def alpha018(df):
    """
    Alpha#18
    (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + 
    correlation(close, open, 10))))
    """
    temp1 = u.stddev(abs((df.close - df.open)), 5 )
    temp2 = df.close - df.open
    temp3 = u.corr(df.close, df.open, 10)
    return (-1 * u.rank(temp1 + temp2 + temp3))
    
def alpha019(df):
    """
    Alpha#19
    ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * 
    (1 + rank((1 + sum(returns, 250)))))
    """
    temp1 = (-1 * np.sign(((df.close - u.delay(df.close, 7)) + u.delta(df.close, 7))))
    temp2 = (1 + u.rank((1 + u.ts_sum(df.returns, 250))))
    return (temp1 * temp2)

def alpha020(df):
    """
    Alpha#20
    (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * 
    rank((open - delay(low, 1)))) 
    """
    temp1 = (-1 * u.rank((df.open - u.delay(df.high, 1))))
    temp2 = u.rank((df.open - u.delay(df.close, 1)))
    temp3 = u.rank((df.open - u.delay(df.low, 1)))
    return (temp1 * temp2 * temp3)

def alpha021(df):
    """
    Alpha#21
    ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : 
    (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 
    1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1)))) 
    """
    decision1 = (u.ts_sum(df.close, 8) / 8 + u.stddev(df.close, 8)) < (u.ts_sum(df.close, 2) / 2)
    decision2 = (u.ts_sum(df.close, 2) / 2 < (u.ts_sum(df.close, 8) / 8) - u.stddev(df.close, 8))
    decision3 = ((1 < (df.volume / df.adv20)) or ((df.volume / df.adv20) == 1))
    return np.where(decision1, (-1 * 1), np.where(decision2, 1, np.where(decision3, 1, (-1 * 1))))

def alpha022(df):
    """
    Alpha#22
    (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    """
    return (-1 * (u.delta(u.corr(df.high, df.volume, 5), 5) * u.rank(u.stddev(df.close, 20))))

def alpha023(df):
    """
    Alpha#23
    (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0) 
    """
    return pd.Series(np.where((u.ts_sum(df.high, 20) / 20) < df.high, (-1 * u.delta(df.high, 2)), 0), df.index)

def alpha024(df):
    """
    Alpha#24
    Can be shortened without the || (or operator) and just use the <= statement.
    ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||
    ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? 
    (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3))) 
    """
    decision = u.delta((u.ts_sum(df.close, 100) / 100), 100) / u.delay(df.close, 100) <= 0.05
    if_true = (-1 * (df.close - u.ts_min(df.close, 100)))
    if_false = (-1 * u.delta(df.close, 3))
    return pd.Series(np.where(decision, if_true, if_false), df.index)

def alpha025(df):
    """
    Alpha#25
    rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    """
    return u.rank(((((-1 * df.returns) * df.adv20) * df.vwap) * (df.high - df.close)))

def alpha026(df):
    """
    Alpha#26
    (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)) 
    """
    return (-1 * u.ts_max(u.corr(u.ts_rank(df.volume, 5), u.ts_rank(df.high, 5), 5), 3)) 