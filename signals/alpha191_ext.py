import pandas as pd
import numpy as np

def rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    横截面排序 (Cross-Sectional Rank)
    返回按升序排列的横截面分位数 (0, 1]
    
    参数:
    df: 索引为交易日，列为股票代码的 DataFrame
    """
    return df.rank(axis=1, pct=True)

def decay_linear(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    线性衰减加权移动平均 (Linear Decay Moving Average)
    距离当前越近的数据权重越高。
    权重 w = [1, 2, ..., window] / sum(1..window)
    """
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()
    
    def apply_weights(x):
        if len(x) < window or np.isnan(x).any():
            return np.nan
        return np.dot(x, weights)
        
    return df.rolling(window).apply(apply_weights, raw=True)

def ts_argmax(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    时序最大值位置 (Time-Series ArgMax)
    返回过去 window 天内的最大值发生在距离今天几天前。
    返回值范围: [0, window - 1]。0 表示今天最大，window-1 表示窗口期第一天最大。
    """
    def calc_argmax(x):
        if len(x) < window or np.isnan(x).all():
            return np.nan
        return window - 1 - np.nanargmax(x)
        
    return df.rolling(window).apply(calc_argmax, raw=True)

def correlation(df1: pd.DataFrame, df2: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    两序列滚动时序相关系数 (Time-Series Correlation)
    计算 df1 和 df2 在过去 window 天的皮尔逊相关系数。
    """
    return df1.rolling(window).corr(df2)

def delay(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序滞后 (Time-Series Delay)"""
    return df.shift(d)

def delta(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序差分 (Time-Series Delta)"""
    return df - df.shift(d)

def scale(df: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    """
    横截面缩放 (Cross-Sectional Scale)
    将每一行（截面）绝对值之和缩放到 a。
    """
    abs_sum = df.abs().sum(axis=1)
    abs_sum = abs_sum.replace(0, np.nan)
    return df.div(abs_sum, axis=0) * a
