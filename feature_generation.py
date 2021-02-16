"""
This file implements most alphas of the paper
101 Alphas by Zura Kakushadze
https://arxiv.org/ftp/arxiv/papers/1601/1601.00991.pdf

February 2021
Jan Gobeli
"""


import utils as u
import pandas as pd
import numpy as np

 
def alpha1(df):
    """
    Alpha#1
    (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5) 

    :param df: dataframe
    :return: 
    """
    temp1 = pd.Series(np.where((df.returns < 0), u.stddev(df.returns, 20), df.close), index = df.index)
    return (u.rank(u.ts_argmax(temp1**2, 5)) - 0.5) 

    
def alpha2(df):
    """
    Alpha#2
    (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    """
    tmp_1 = u.rank(u.delta(np.log(df.volume), 2))
    tmp_2 = u.rank(((df.close - df.open) / df.open))
    return (-1 * u.corr(tmp_1, tmp_2, 6))


def alpha3(df):
    """
    Alpha#3
    (-1 * correlation(rank(open), rank(volume), 10))
    """
    return (-1 * u.corr(u.rank(df.open), u.rank(df.volume), 10))


def alpha4(df):
    """
    Alpha #4
    (-1 * Ts_Rank(rank(low), 9))
    """
    return (-1 * u.ts_rank(u.rank(df.low), 9))


def alpha5(df):
    """
    Alpha#5
    (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))) 
    """
    return (u.rank((df.open - (u.ts_sum(df.vwap, 10) / 10))) * (-1 * abs(u.rank((df.close - df.vwap))))) 


def alpha6(df):
    """
    Alpha#6
    (-1 * correlation(open, volume, 10)) 
    """
    return (-1 * u.corr(df.open, df.volume, 10))


def alpha7(df):
    """
    Alpha#7
    ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : 
    (-1 * 1)) 
    """
    iftrue = ((-1 * u.ts_rank(abs(u.delta(df.close, 7)), 60)) * np.sign(u.delta(df.close, 7)))
    return pd.Series(np.where(u.adv(df, 20) < df.volume, iftrue, (-1 * 1)), index=df.index)


def alpha8(df):
    """
    Alpha#8
    (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * 
    sum(returns, 5)), 10))))
    """
    temp1 = (u.ts_sum(df.open, 5) * u.ts_sum(df.returns, 5))
    temp2 = u.delay((u.ts_sum(df.open, 5) * u.ts_sum(df.returns, 5)), 10)
    return (-1 * u.rank(temp1 - temp2))


def alpha9(df):
    """
    Alpha#9
    ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : 
    ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))) 
    """
    tempd1 = u.delta(df.close, 1)
    tempmin = u.ts_min(tempd1, 5)
    tempmax = u.ts_max(tempd1, 5)
    return pd.Series(np.where(tempmin > 0, tempd1, np.where(tempmax < 0, tempd1, (-1 * tempd1))), df.index)


def alpha10(df):
    """
    Alpha#10
    rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)
    ? delta(close, 1) : (-1 * delta(close, 1)))))
    """
    tempd1 = u.delta(df.close, 1)
    tempmin = u.ts_min(tempd1, 4)
    tempmax = u.ts_max(tempd1, 4)
    return u.rank(pd.Series(np.where(tempmin > 0, tempd1, np.where(tempmax < 0, tempd1, (-1 * tempd1))), df.index))


def alpha11(df):
    """
    Alpha#11
    ((rank(ts_max((vwap - close), 3)) + 
    rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3))) 
    """
    temp1 = u.rank(u.ts_max((df.vwap - df.close), 3))
    temp2 = u.rank(u.ts_min((df.vwap - df.close), 3))
    temp3 = u.rank(u.delta(df.volume, 3))
    return temp1 + (temp2 * temp3)


def alpha12(df):
    """
    Alpha#12
    (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    """
    return (np.sign(u.delta(df.volume, 1)) * (-1 * u.delta(df.close, 1)))

def alpha13(df):
    """
    Alpha#13
    (-1 * rank(covariance(rank(close), rank(volume), 5)))
    """
    return (-1 * u.rank(u.cov(u.rank(df.close), u.rank(df.volume), 5)))

def alpha14(df):
    """
    Alpha#14
    ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)) 
    """
    return ((-1 * u.rank(u.delta(df.returns, 3))) * u.corr(df.open, df.volume, 10))


def alpha15(df):
    """
    Alpha#15
    (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)) 
    """
    return (-1 * u.ts_sum(u.corr(u.rank(df.high), u.rank(df.volume), 3), 3))

def alpha16(df):
    """
    Alpha#16
    (-1 * rank(covariance(rank(high), rank(volume), 5))) 
    """
    return (-1 * u.rank(u.cov(u.rank(df.high), u.rank(df.volume), 5)))

def alpha17(df):
    """
    Alpha#17
    (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *
    rank(ts_rank((volume / adv20), 5))) 
    """  
    temp1 = (-1 * u.rank(u.ts_rank(df.close, 10)))
    temp2 = u.rank(u.delta(u.delta(df.close, 1), 1))
    temp3 = u.rank(u.ts_rank((df.volume / u.adv(df, 20)), 5))
    return ((temp1 * temp2) * temp3)

def alpha18(df):
    """
    Alpha#18
    (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + 
    correlation(close, open, 10))))
    """
    temp1 = u.stddev(abs((df.close - df.open)), 5 )
    temp2 = df.close - df.open
    temp3 = u.corr(df.close, df.open, 10)
    return (-1 * u.rank(temp1 + temp2 + temp3))
    
def alpha19(df):
    """
    Alpha#19
    ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * 
    (1 + rank((1 + sum(returns, 250)))))
    """
    temp1 = (-1 * np.sign(((df.close - u.delay(df.close, 7)) + u.delta(df.close, 7))))
    temp2 = (1 + u.rank((1 + u.ts_sum(df.returns, 250))))
    return (temp1 * temp2)

def alpha20(df):
    """
    Alpha#20
    (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * 
    rank((open - delay(low, 1)))) 
    """
    temp1 = (-1 * u.rank((df.open - u.delay(df.high, 1))))
    temp2 = u.rank((df.open - u.delay(df.close, 1)))
    temp3 = u.rank((df.open - u.delay(df.low, 1)))
    return (temp1 * temp2 * temp3)

def alpha21(df):
    """
    Alpha#21
    ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : 
    (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 
    1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1)))) 
    """
    decision1 = (u.ts_sum(df.close, 8) / 8 + u.stddev(df.close, 8)) < (u.ts_sum(df.close, 2) / 2)
    decision2 = (u.ts_sum(df.close, 2) / 2 < (u.ts_sum(df.close, 8) / 8) - u.stddev(df.close, 8))
    decision3 = ((1 < (df.volume / u.adv(df, 20))) | ((df.volume / u.adv(df, 20)) == 1))
    return np.where(decision1, (-1 * 1), np.where(decision2, 1, np.where(decision3, 1, (-1 * 1))))

def alpha22(df):
    """
    Alpha#22
    (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    """
    return (-1 * (u.delta(u.corr(df.high, df.volume, 5), 5) * u.rank(u.stddev(df.close, 20))))

def alpha23(df):
    """
    Alpha#23
    (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0) 
    """
    return pd.Series(np.where((u.ts_sum(df.high, 20) / 20) < df.high, (-1 * u.delta(df.high, 2)), 0), df.index)

def alpha24(df):
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

def alpha25(df):
    """
    Alpha#25
    rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    """
    return u.rank(((((-1 * df.returns) * u.adv(df, 20)) * df.vwap) * (df.high - df.close)))

def alpha26(df):
    """
    Alpha#26
    (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)) 
    """
    return (-1 * u.ts_max(u.corr(u.ts_rank(df.volume, 5), u.ts_rank(df.high, 5), 5), 3)) 

def alpha27(df):
    """
    Alpha#27
    ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1) 
    """
    temp = np.where((0.5 < u.rank((u.ts_sum(u.corr(u.rank(df.volume), u.rank(df.vwap), 6), 2) / 2.0))), (-1 * 1),  1)
    return pd.Series(temp, index=df.index)

def alpha28(df):
    """  
    Alpha#28
    scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    """
    return u.scale(((u.corr(u.adv(df, 20), df.low, 5) + ((df.high + df.low) / 2)) - df.close))

def alpha29(df):
    """
    Alpha#29
    (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),
    5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5)) 
    """
    temp1 = u.scale(np.log(u.ts_sum(u.ts_min(u.rank(u.rank((-1 * u.rank(u.delta((df.close - 1), 5))))), 2), 1)))
    temp2 = u.product(u.rank(u.rank(temp1)), 1)
    temp3 = u.ts_rank(u.delay((-1 * df.returns), 6), 5)
    return (np.where(temp1 < temp2, temp1, temp2) + temp3)

def alpha30(df):
    """
    Alpha#30
     (((1.0 - rank(((sign((close - delay(close, 1))) + 
     sign((delay(close, 1) - delay(close, 2)))) +
     sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20)) 
    """
    return (((1.0 - u.rank(((np.sign((df.close - u.delay(df.close, 1))) \
            + np.sign((u.delay(df.close, 1) - u.delay(df.close, 2))))   \
            + np.sign((u.delay(df.close, 2) - u.delay(df.close, 3)))))) \
            * u.ts_sum(df.volume, 5)) / u.ts_sum(df.volume, 20))

def alpha31(df):
    """
    Alpha#31
    ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) 
    + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    """
    temp1 = u.rank(u.rank(u.rank(u.decay_linear((-1 * u.rank(u.rank(u.delta(df.close, 10)))), 10))))
    temp2 = u.rank((-1 * u.delta(df.close, 3))) + np.sign(u.scale(u.corr(u.adv(df, 20), df.low, 12)))
    return temp1 + temp2

def alpha32(df):
    """
    Alpha#32
    (scale(((sum(close, 7) / 7) - close)) + 
    (20 * scale(correlation(vwap, delay(close, 5), 230)))) 
    """
    temp1 = u.scale(((u.ts_sum(df.close, 7) / 7) - df.close))
    temp2 = (20 * u.scale(u.corr(df.vwap, u.delay(df.close, 5), 230)))
    return temp1 + temp2

def alpha33(df):
    """
    Alpha#33
    rank((-1 * ((1 - (open / close))^1)))
    """
    return u.rank((-1 * ((1 - (df.open / df.close))**1)))

def alpha34(df):
    """
    Alpha#34
    rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1))))) 
    """
    return u.rank(((1 - u.rank((u.stddev(df.returns, 2) / u.stddev(df.returns, 5)))) \
            + (1 - u.rank(u.delta(df.close, 1)))))

def alpha35(df):
    """
    Alpha#35
    ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * 
    (1 - Ts_Rank(returns, 32)))
    """
    return ((u.ts_rank(df.volume, 32) * (1 - u.ts_rank(((df.close + df.high) - df.low), 16))) \
            * (1 - u.ts_rank(df.returns, 32)))

def alpha36(df):
    """
    Alpha#36
    (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + 
    (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + 
    rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open))))) 
    """
    temp1 = (2.21 * u.rank(u.corr((df.close - df.open), u.delay(df.volume, 1), 15)))
    temp2 = (0.7 * u.rank((df.open - df.close)))
    temp3 = (0.73 * u.rank(u.ts_rank(u.delay((-1 * df.returns), 6), 5)))
    temp4 = u.rank(abs(u.corr(df.vwap, u.adv(df, 20), 6)))
    temp5 = (0.6 * u.rank((((sum(df.close, 200) / 200) - df.open) * (df.close - df.open))))
    return ((((temp1 + temp2) + temp3) + temp4) + temp5)

def alpha37(df):
    """
    Alpha#37
    (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close))) 
    """
    return (u.rank(u.corr(u.delay((df.open - df.close), 1), df.close, 200)) + u.rank((df.open - df.close)))

def alpha38(df):
    """
    Alpha#38
    ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open))) 
    """
    return ((-1 * u.rank(u.ts_rank(df.close, 10))) * u.rank((df.close / df.open)))

def alpha39(df):
    """
    Alpha#39

    """
    temp = (-1 * u.rank((u.delta(df.close, 7) * (1 - u.rank(u.decay_linear((df.volume / u.adv(df, 20)), 9))))))
    return (temp * (1 + u.rank(u.ts_sum(df.returns, 250))))

def alpha40(df):
    """
    Alpha#40
    ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    """
    return ((-1 * u.rank(u.stddev(df.high, 10))) * u.corr(df.high, df.volume, 10))

def alpha41(df):
    """
    Alpha#41
    (((high * low)^0.5) - vwap) 
    """
    return (((df.high * df.low)**0.5) - df.vwap)

def alpha42(df):
    """
    Alpha#42
    (rank((vwap - close)) / rank((vwap + close))) 
    """
    return (u.rank((df.vwap - df.close)) / u.rank((df.vwap + df.close)))

def alpha43(df):
    """
    Alpha#43
    (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8)) 
    """
    return (u.ts_rank((df.volume / u.adv(df, 20)), 20) * u.ts_rank((-1 * u.delta(df.close, 7)), 8))

def alpha44(df):
    """
    Alpha#44
    (-1 * correlation(high, rank(volume), 5))
    """
    return (-1 * u.corr(df.high, u.rank(df.volume), 5))

def alpha45(df):
    """
    Alpha#45
    (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) 
    * rank(correlation(sum(close, 5), sum(close, 20), 2)))) 
    """
    temp1 = u.rank((u.ts_sum(u.delay(df.close, 5), 20) / 20))
    temp2 = u.corr(df.close, df.volume, 2)
    temp3 = u.rank(u.corr(u.ts_sum(df.close, 5), u.ts_sum(df.close, 20), 2))
    return (-1 * ((temp1 * temp2) * temp3))

def alpha46(df):
    """
    Alpha#46
    ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : 
    (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :
    ((-1 * 1) * (close - delay(close, 1)))))
    """
    decision1 = (0.25 < (((u.delay(df.close, 20) - u.delay(df.close, 10)) / 10) - ((u.delay(df.close, 10) - df.close) / 10)))
    decision2 = ((((u.delay(df.close, 20) - u.delay(df.close, 10)) / 10) - ((u.delay(df.close, 10) - df.close) / 10)) < 0)
    iffalse = ((-1 * 1) * (df.close - u.delay(df.close, 1)))
    return pd.Series(np.where(decision1, (-1 * 1), np.where(decision2, 1, iffalse)), index=df.index)

def alpha47(df):
    """
    Alpha#47
    ((((rank((1 / close)) * volume) / adv20) * 
    ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5)))) 
    """
    temp1 = ((u.rank((1 / df.close)) * df.volume) / u.adv(df, 20))
    temp2 = ((df.high * u.rank((df.high - df.close))) / (u.ts_sum(df.high, 5) / 5))
    return ((temp1 * temp2) - u.rank((df.vwap - u.delay(df.vwap, 5))))

def alpha48(df):
    """
    Alpha#48
    (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *
    delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
    """
    pass

def alpha49(df):
    """
    Alpha#49
    (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 
    (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    """
    temp1 = ((u.delay(df.close, 20) - u.delay(df.close, 10)) / 10)
    temp2 = ((u.delay(df.close, 10) - df.close) / 10)
    return pd.Series(np.where(((temp1 - temp2) < (-1 * 0.1)), 1, ((-1 * 1) * (df.close - u.delay(df.close, 1)))), index=df.index)

def alpha50(df):
    """
    Alpha#50
    (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    """
    return (-1 * u.ts_max(u.rank(u.corr(u.rank(df.volume), u.rank(df.vwap), 5)), 5))

def alpha51(df):
    """
    Alpha#51
    (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) 
    < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    """
    condition = ((((u.delay(df.close, 20) - u.delay(df.close, 10)) / 10) \
        - ((u.delay(df.close, 10) - df.close) / 10)) < (-1 * 0.05))
    return pd.Series(np.where(condition, 1, ((-1 * 1) * (df.close - u.delay(df.close, 1)))), df.index)

def alpha52(df):
    """
    Alpha#52
    ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) 
    - sum(returns, 20)) / 220))) * ts_rank(volume, 5)) 
    """
    temp1 = ((-1 * u.ts_min(df.low, 5)) + u.delay(u.ts_min(df.low, 5), 5))
    temp2 = u.rank(((u.ts_sum(df.returns, 240) - u.ts_sum(df.returns, 20)) / 220))
    return ((temp1 * temp2) * u.ts_rank(df.volume, 5))

def alpha53(df):
    """
    Alpha#53
    (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    """
    return (-1 * u.delta((((df.close - df.low) - (df.high - df.close)) / (df.close - df.low)), 9))

def alpha54(df):
    """
    Alpha#54
    ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    """
    return ((-1 * ((df.low - df.close) * (df.open**5))) / ((df.low - df.high) * (df.close**5)))

def alpha55(df):
    """
    Alpha#55
    (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) 
    - ts_min(low, 12)))), rank(volume), 6)) 
    """
    temp1 = (df.close - u.ts_min(df.low, 12))
    temp2 = (u.ts_max(df.high, 12) - u.ts_min(df.low,12))
    return (-1 * u.corr(u.rank((temp1 / temp2)), u.rank(df.volume), 6))

def alpha56(df):
    """
    Alpha#56
    (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap))))) 
    No MarketCap Available
    """
    pass

def alpha57(df):
    """
    Alpha#57
    (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)))) 
    """
    return (0 - (1 * ((df.close - df.vwap) / u.decay_linear(u.rank(u.ts_argmax(df.close, 30)), 2))))

def alpha58(df):
    """
    Alpha#58
    (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), 
    volume, 3.92795), 7.89291), 5.50322))
    """
    pass

def alpha59(df):
    """
    Alpha#59
    (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *
    (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
    """
    pass

def alpha60(df):
    """
    Alpha#60
    (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) 
    - scale(rank(ts_argmax(close, 10))))))
    """
    temp1 = u.scale(u.rank(((((df.close - df.low) - (df.high - df.close)) / (df.high - df.low)) * df.volume)))
    return (0 - (1 * ((2 * temp1) - u.scale(u.rank(u.ts_argmax(df.close, 10))))))

def alpha61(df):
    """
    Alpha#61
    (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))

    Rounded the days to int since partial lookback 
    """
    return (u.rank((df.vwap - u.ts_min(df.vwap, 16))) < u.rank(u.corr(df.vwap, u.adv(df, 180), 18)))

def alpha62(df):
    """
    Alpha#62
    ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) 
    + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1) 
    """
    temp1 = u.rank(u.corr(df.vwap, u.ts_sum(u.adv(df, 20), 22), 10))
    temp2 = u.rank(((u.rank(df.open) + u.rank(df.open)) < (u.rank(((df.high + df.low) / 2)) + u.rank(df.high))))
    return ((temp1 < temp2) * -1) 

def alpha63(df):
    """
    Alpha#63
    ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))
    - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), 
    sum(adv180, 37.2467), 13.557), 12.2883))) * -1) 
    """
    pass

def alpha64(df):
    """
    Alpha#64
    ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),
    sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) 
    + (vwap * (1 -0.178404))), 3.69741))) * -1) 
    """
    temp1 = u.ts_sum(((df.open * 0.178404) + (df.low * (1 - 0.178404))), 13)
    temp2 = u.rank(u.corr(temp1, u.ts_sum(u.adv(df, 120), 18), 17))
    temp3 = u.rank(u.delta(((((df.high + df.low) / 2) * 0.178404) + (df.vwap * (1 - 0.178404))), 4))
    return ((temp2 < temp3) * -1)

def alpha65(df):
    """
    Alpha#65
    ((rank(correlation(((open * 0.00817205) + 
    (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1) 
    """
    temp1 = (df.open * 0.00817205) + (df.vwap * (1 - 0.00817205))
    temp2 = u.rank((df.open - u.ts_min(df.open, 14)))
    return ((u.rank(u.corr(temp1, u.ts_sum(u.adv(df, 60), 9), 6)) < temp2) * -1)

def alpha66(df):
    """
    Alpha#66
    ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) 
    + Ts_Rank(decay_linear(((((low * 0.96633) 
    + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1) 
    """
    temp1 = u.rank(u.decay_linear(u.delta(df.vwap, 4), 7.23052))
    temp2 = (((df.low * 0.96633) + (df.low * (1 - 0.96633))) - df.vwap)
    temp3 = (df.open - ((df.high + df.low) / 2))
    return ((temp1 + u.ts_rank(u.decay_linear((temp2 / temp3), 11.4157), 7)) * -1)

def alpha67(df):
    """
    Alpha#67
    ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap,
    IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1) 
    """
    pass

def alpha68(df):
    """
    Alpha#68
    ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) 
    < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    """
    temp1 = u.ts_rank(u.corr(u.rank(df.high), u.rank(u.adv(df,15)), 9), 14)
    temp2 = u.rank(u.delta(((df.close * 0.518371) + (df.low * (1 - 0.518371))), 1))
    return u.rank(u.delta(((df.close * 0.518371) + (df.low * (1 - 0.518371))), 1))

def alpha69(df):
    """
    Alpha#69
    ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412),
    4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),
    9.0615)) * -1) 
    """
    pass

def alpha70(df):
    """
    Alpha#70
    ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close,
    IndClass.industry), adv50, 17.8256), 17.9171)) * -1) 
    """
    pass

def alpha71(df):
    """
    Alpha#71
    max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), 
    Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) 
    - (vwap + vwap)))^2), 16.4662), 4.4388))
    """
    temp1 = u.corr(u.ts_rank(df.close, 3), u.ts_rank(u.adv(df,180), 12), 18)
    temp2 = u.ts_rank(u.decay_linear((u.rank(((df.low + df.open) - (df.vwap + df.vwap)))**2), 16.4662), 4)
    return pd.Series(np.where(temp1 > temp2, temp1, temp2), df.index)

def alpha72(df):
    """
    Alpha#72
    (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /
    rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), 
    Ts_Rank(volume, 18.5188), 6.86671),2.95011))) 
    """
    temp1 = u.rank(u.decay_linear(u.corr(((df.high + df.low) / 2), u.adv(df, 40), 9), 10))
    temp2 = u.rank(u.decay_linear(u.corr(u.ts_rank(df.vwap, 4), u.ts_rank(df.volume, 19), 7),2.95011)) 
    return (temp1 / temp2) 

def alpha73(df):
    """
    Alpha#73
    (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),
    Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) 
    / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    """
    temp1 = u.rank(u.decay_linear(u.delta(df.vwap, 5), 2.91864))
    temp2 = u.delta(((df.open * 0.147155) + (df.low * (1 - 0.147155))), 2)
    temp3 = ((df.open * 0.147155) + (df.low * (1 - 0.147155)))
    temp4 = u.ts_rank(u.decay_linear(((temp2 / temp3) * -1), 2), 17)
    return pd.Series(np.where(temp1 > temp4, temp1 * -1, temp4 * -1), df.index)


def alpha74(df):
    """
    Alpha#74
    ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) 
    < rank(correlation(rank(((high * 0.0261661) 
    + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)
    """
    temp1 = u.rank(u.corr(df.close, u.ts_sum(u.adv(df, 30), 37), 15))
    temp2 = u.rank(u.corr(u.rank(((df.high * 0.0261661) + (df.vwap * (1 - 0.0261661)))), u.rank(df.volume), 11)) 
    return ((temp1 < temp2) * -1)


def alpha75(df):
    """
    Alpha#75(df)
    (rank(correlation(vwap, volume, 4.24304))
    < rank(correlation(rank(low), rank(adv50), 12.4413)))
    """
    temp1 = u.rank(u.corr(df.vwap, df.volume, 4))
    temp2 = u.rank(u.corr(u.rank(df.low), u.rank(u.adv(df, 50)), 12))
    return (temp1 < temp2)


def alpha76(df):
    """
    Alpha#76
    (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),
    Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81,
    8.14941), 19.569), 17.1543), 19.383)) * -1) 
    """
    pass


def alpha77(df):
    """
    Alpha#77
    min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),
    rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125))) 
    """
    temp1 = u.rank(u.decay_linear(((((df.high + df.low) / 2) + df.high) - (df.vwap + df.high)), 20.0451))
    temp2 = u.rank(u.decay_linear(u.corr(((df.high + df.low) / 2), u.adv(df, 40), 3), 5.64125))
    return pd.Series(np.where(temp1 > temp2, temp1, temp2), index=df.index)


def alpha78(df):
    """
    Alpha#78
    (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),
    sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
    """
    temp1 = u.ts_sum(((df.low * 0.352233) + (df.vwap * (1 - 0.352233))), 20)
    temp2 = u.rank(u.corr(u.rank(df.vwap), u.rank(df.volume), 6))
    temp3 = u.rank(u.corr(temp1, u.ts_sum(u.adv(df, 40), 20), 7))
    return (temp3**temp2)


def alpha79(df):
    """
    Alpha#79
    (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),
    IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), 
    Ts_Rank(adv150, 9.18637), 14.6644))) 
    """
    pass


def alpha80(df):
    """
    Alpha#80
    ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),
    IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
    """
    pass


def alpha81(df):
    """
    Alpha#81
    ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),
    8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    """
    temp = u.rank(np.log(u.product(u.rank(u.rank(u.corr(df.vwap, u.ts_sum(u.adv(df, 10), 50), 8))**4), 15)))
    return ((temp < u.rank(u.corr(u.rank(df.vwap), u.rank(df.volume), 5))) * -1)


def alpha82(df):
    """
    Alpha#82
    (min(rank(decay_linear(delta(open, 1.46063), 14.8717)),
    Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), 
    ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
    """
    pass


def alpha83(df):
    """
    Alpha#83
    ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) 
    / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
    """
    temp1 = u.rank(u.delay(((df.high - df.low) / (u.ts_sum(df.close, 5) / 5)), 2)) * u.rank(u.rank(df.volume))
    temp2 = (((df.high - df.low) / (u.ts_sum(df.close, 5) / 5)) / (df.vwap - df.close))
    return (temp1 / temp2)


def alpha84(df):
    """
    Alpha#84
    SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), 
    delta(close, 4.96796)) 
    """
    return (u.ts_rank((df.vwap - u.ts_max(df.vwap, 15)), 21)**u.delta(df.close, 5))


def alpha85(df):
    """
    Alpha#85
    (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,
    9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), 
    Ts_Rank(volume, 10.1595), 7.11408))) 
    """
    temp1 = u.rank(u.corr(((df.high * 0.876703) + (df.close * (1 - 0.876703))), u.adv(df, 30), 10))
    temp2 = u.rank(u.corr(u.ts_rank(((df.high + df.low) / 2), 4), u.ts_rank(df.volume, 10), 7))
    return (temp1**temp2)


def alpha86(df):
    """
    Alpha#86
    ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) 
    < rank(((open + close) - (vwap + open)))) * -1)
    """
    temp1 = u.ts_rank(u.corr(df.close, u.ts_sum(u.adv(df, 20), 15), 6), 20)
    temp2 = u.rank(((df.open + df.close) - (df.vwap + df.open)))
    return ((temp1 < temp2) * -1)

def alpha87(df):
    """
    Alpha#87
    (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))),
    1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,
    IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
    """
    pass

def alpha88(df):
    """
    Alpha#88
    min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))),
    8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60,
    20.6966), 8.01266), 6.65053), 2.61957)) 
    """
    temp1 = u.rank(u.decay_linear(((u.rank(df.open) + u.rank(df.low)) - (u.rank(df.high) + u.rank(df.close))), 8))
    temp2 = u.ts_rank(u.decay_linear(u.corr(u.ts_rank(df.close, 8), u.ts_rank(u.adv(df, 60), 21), 8), 6.65053), 3)
    return pd.Series(np.where(temp1 < temp2, temp1, temp2), index=df.index)

def alpha89(df):
    """
    Alpha#89
    (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10,
    6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap,
    IndClass.industry), 3.48158), 10.1466), 15.3012)) 
    """
    pass

def alpha90(df):
    """
    Alpha#90
    ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40,
    IndClass.subindustry), low, 5.38375), 3.21856)) * -1) 
    """
    pass

def alpha91(df):
    """
    Alpha#91
    ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close,
    IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -
    rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1) 
    """

def alpha92(df):
    """
    Alpha#92
    min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221),
    18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024),
    6.80584))
    """
    temp1 = u.ts_rank(u.decay_linear(((((df.high + df.low) / 2) + df.close) < (df.low + df.open)), 14.7221), 19)
    temp2 = u.ts_rank(u.decay_linear(u.corr(u.rank(df.low), u.rank(u.adv(df, 30)), 8), 6.94024), 7)
    return pd.Series(np.where(temp1 < temp2, temp1, temp2), index=df.index)

def alpha93(df):
    """
    Alpha#93
    (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81,
    17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) 
    + (vwap * (1 - 0.524434))), 2.77377), 16.2664))) 
    """
    pass

def alpha94(df):
    """
    Alpha#94
    ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap,
    19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1) 
    """
    temp1 = u.rank((df.vwap - u.ts_min(df.vwap, 12)))
    temp2 = u.ts_rank(u.corr(u.ts_rank(df.vwap, 20), u.ts_rank(u.adv(df, 60), 4), 18), 3)
    return ((temp1**temp2) * -1)

def alpha95(df):
    """
    Alpha#95
    (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low)
    / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584)) 
    """
    temp1 = u.rank((df.open - u.ts_min(df.open, 12)))
    temp2 = u.corr(u.ts_sum(((df.high + df.low) / 2), 19), u.ts_sum(u.adv(df, 40), 19), 13)
    return (temp1 < u.ts_rank((u.rank(temp2)**5), 12))

def alpha96(df):
    """
    Alpha#96
    (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878),
    4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404),
    Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1) 
    """
    temp1 = u.ts_rank(u.decay_linear(u.corr(u.rank(df.vwap), u.rank(df.volume), 4), 4.16783), 8)
    temp2 = u.corr(u.ts_rank(df.close, 7), u.ts_rank(u.adv(df, 60), 4), 4)
    temp3 = u.ts_rank(u.decay_linear(u.ts_argmax(temp2, 13), 14.0365), 13)
    return pd.Series(np.where(temp1 > temp3, temp1, temp3), index=df.index)

def alpha97(df):
    """
    Alpha#97
    ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),
    IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low,
    7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1) 
    """
    pass

def alpha98(df):
    """
    Alpha#98
    (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -
    rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571),
    6.95668), 8.07206))) 
    """
    temp1 = u.ts_rank(u.ts_argmin(u.corr(u.rank(df.open), u.rank(u.adv(df, 15)), 21), 9), 7)
    temp2 = u.rank(u.decay_linear(temp1, 8.07206))
    temp3 = u.rank(u.decay_linear(u.corr(df.vwap, u.ts_sum(u.adv(df, 5), 26), 5), 7)) 
    return (temp3 - temp2) 

def alpha99(df):
    """
    Alpha#99
    ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) 
    < rank(correlation(low, volume, 6.28259))) * -1) 
    """
    temp1 = u.rank(u.corr(u.ts_sum(((df.high + df.low) / 2), 20), u.ts_sum(u.adv(df, 60), 20), 9))
    temp2 = u.rank(u.corr(df.low, df.volume, 6))
    return pd.Series(np.where(temp1 < temp2, temp1 * -1, temp2 * -1), index=df.index)

def alpha100(df):
    """
    Alpha#100
    (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) 
    - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) 
    - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))),
    IndClass.subindustry))) * (volume / adv20))))
    """
    pass

def alpha101(df):
    """
    Alpha#101
    ((close - open) / ((high - low) + .001)) 
    """
    return ((df.close - df.open) / ((df.high - df.low) + .001)) 
