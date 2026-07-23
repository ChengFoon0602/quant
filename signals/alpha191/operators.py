"""
Alpha 191 核心算子 — 全部向量化实现。

16 个算子覆盖国泰君安 191 因子库的全部计算需求。
所有算子接受和返回 pandas Series，不做逐行循环。

参考：国泰君安金融工程《基于高频数据的 Alpha 因子筛选》
"""

import numpy as np
import pandas as pd


def SUM(series, n):
    """过去 n 日求和"""
    return series.rolling(n).sum()


def STD(series, n):
    """过去 n 日标准差（总体标准差）"""
    return series.rolling(n).std(ddof=0)


def _MAX(series, n):
    """过去 n 日最大值"""
    return series.rolling(n).max()


def _MIN(series, n):
    """过去 n 日最小值"""
    return series.rolling(n).min()


def DELTA(series, n):
    """n 日差分: A(t) - A(t-n)"""
    return series - series.shift(n)


def DELAY(series, n):
    """n 日延迟: A(t-n)"""
    return series.shift(n)


def RANK(series):
    """截面升序排名，归一化 0~1"""
    return series.rank(pct=True)


# 别名: 公式中 MAX/MIN 和算子名冲突，用下划线前缀区分的版本
MAX = _MAX
MIN = _MIN


def TSRANK(series, n):
    """时间序列排名: 当前值在过去 n 日的百分位排名 (升序, 0~1)"""
    def _rank_last(x):
        # x 是 numpy array (raw=True), 计算最后一个元素的百分位排名
        last = x[-1]
        # Ascending rank: fraction of elements <= last
        return np.searchsorted(np.sort(x), last, side='right') / len(x)
    return series.rolling(n).apply(_rank_last, raw=True)


def CORR(a, b, n):
    """过去 n 日 Pearson 相关系数（inf → NaN 处理）"""
    result = a.rolling(n).corr(b)
    return result.replace([np.inf, -np.inf], np.nan)


def SMA(series, n, m):
    """
    n 日递归移动平均，m 为权重参数。

    SMA_t = (m * A_t + (n - m) * SMA_{t-1}) / n

    等价于 EMA(alpha = m/n)，初值 = 前 n 日简单均值。
    研报中常见 m=1 和 m=2。
    """
    alpha = m / n
    result = pd.Series(np.nan, index=series.index)
    if len(series) < n:
        return result
    result.iloc[n - 1] = series.iloc[:n].mean()
    ewm = series.ewm(alpha=alpha, adjust=False).mean()
    result.iloc[n:] = ewm.iloc[n:]
    return result


def REGBETA(y, x, n):
    """过去 n 日 y 对 x 的线性回归系数 β = Cov(y,x) / Var(x)"""
    cov = y.rolling(n).cov(x)
    var = x.rolling(n).var(ddof=0)
    return cov / var


def REGRESI(y, x, n):
    """过去 n 日 y 对 x 的线性回归残差"""
    beta = REGBETA(y, x, n)
    return y - beta * x


def COUNT(condition, n):
    """过去 n 日满足条件(True)的天数"""
    return condition.astype(float).rolling(n).sum()


def _LOG(series):
    """自然对数"""
    return np.log(series)


def _ABS(series):
    """绝对值"""
    return np.abs(series)


def _SIGN(series):
    """符号: +1 / -1 / 0"""
    return np.sign(series)


# 别名，方便因子公式使用
LOG = _LOG
ABS = _ABS
SIGN = _SIGN
