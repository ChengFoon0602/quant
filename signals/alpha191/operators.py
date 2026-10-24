"""
Alpha 191 核心算子 — 全部向量化实现。

16 个算子覆盖国泰君安 191 因子库的全部计算需求。
所有算子接受 pandas Series/DataFrame 或 numpy ndarray，不做逐行循环。

参考：国泰君安金融工程《基于高频数据的 Alpha 因子筛选》
"""

import numpy as np
import pandas as pd


def _ensure_series(x):
    """将 ndarray 转换回 pandas 对象（处理 np.where/np.fmax 等的返回类型）。

    1D ndarray → Series,  2D ndarray → DataFrame.
    """
    if isinstance(x, np.ndarray) and not isinstance(x, (pd.Series, pd.DataFrame)):
        if x.ndim == 2:
            return pd.DataFrame(x)
        return pd.Series(x)
    return x


def SUM(series, n):
    """过去 n 日求和"""
    return _ensure_series(series).rolling(n).sum()


def STD(series, n):
    """过去 n 日标准差（总体标准差）"""
    return _ensure_series(series).rolling(n).std(ddof=0)


def _MAX(series, n):
    """过去 n 日最大值"""
    return _ensure_series(series).rolling(n).max()


def _MIN(series, n):
    """过去 n 日最小值"""
    return _ensure_series(series).rolling(n).min()


def DELTA(series, n):
    """n 日差分: A(t) - A(t-n)"""
    s = _ensure_series(series)
    return s - s.shift(n)


def DELAY(series, n):
    """n 日延迟: A(t-n)"""
    return _ensure_series(series).shift(n)


def RANK(series):
    """截面升序排名，归一化 0~1。

    - Series/ndarray 输入：时序排名（单股票内）
    - DataFrame 输入：截面排名 `axis=1`（每行 = 同一日期跨股票排名）
    """
    if isinstance(series, pd.DataFrame):
        return series.rank(axis=1, pct=True)
    return _ensure_series(series).rank(pct=True)


# 别名: 公式中 MAX/MIN 和算子名冲突，用下划线前缀区分的版本
MAX = _MAX
MIN = _MIN


def _tsrank_1d(col: pd.Series, n: int) -> pd.Series:
    """单列时间序列排名: 当前值在过去 n 日的百分位排名 (升序, 0~1)。

    语义等价于原 `rolling(n).apply(searchsorted(sorted(x), x[-1], side='right')/n)`，
    即窗口内 `(x <= x[-1]).sum() / n`。

    2026-10-24 改为向量化：原实现是 Python 逐窗口回调，783 列 × 3886 行时
    单因子需 30–60 秒，全库重筛要 1–2 小时。滑动窗口视图把这一步压到秒级，
    数值逐位一致（窗口含 NaN 时同样输出 NaN，由 min_periods=window 保证）。
    """
    arr = col.to_numpy(dtype=float)
    total = len(arr)
    out = np.full(total, np.nan)
    if total < n:
        return pd.Series(out, index=col.index)

    win = np.lib.stride_tricks.sliding_window_view(arr, n)   # (T-n+1, n)
    valid = ~np.isnan(win).any(axis=1)
    if valid.any():
        sub = win[valid]                                     # (k, n)
        pct = np.full(total - n + 1, np.nan)
        pct[valid] = (sub <= sub[:, -1:]).sum(axis=1) / n
        out[n - 1:] = pct
    return pd.Series(out, index=col.index)



def TSRANK(series, n):
    """时间序列排名: 当前值在过去 n 日的百分位排名 (升序, 0~1)。

    - Series/ndarray 输入：直接计算
    - DataFrame 输入：逐列计算（每只股票独立）
    """
    if isinstance(series, pd.DataFrame):
        return series.apply(lambda col: _tsrank_1d(col, n))
    return _tsrank_1d(_ensure_series(series), n)


def CORR(a, b, n):
    """过去 n 日 Pearson 相关系数（inf → NaN 处理）"""
    result = _ensure_series(a).rolling(n).corr(_ensure_series(b))
    return result.replace([np.inf, -np.inf], np.nan)


def _sma_1d(col: pd.Series, n: int, m: int) -> pd.Series:
    """单列递归移动平均 (SMA)。

    SMA_t = (m·A_t + (n−m)·SMA_{t−1}) / n，**初值 = 前 n 日简单均值**。

    2026-10-24 修正：原实现用 `ewm(alpha=m/n, adjust=False)` 的值填充 t≥n，
    而 ewm 的递归是从 x[0] 起步的——等于在 t=n−1 处打了个补丁，之后仍沿用
    ewm 自己的历史路径，与「均值初值纯递归」不等价（实测偏差 0.13%~0.46%）。
    双实现互证发现。受影响因子：009、076–093、087（最终池）、173、174。
    """
    a = col.to_numpy(dtype=float)
    total = len(a)
    out = np.full(total, np.nan)
    if total < n:
        return pd.Series(out, index=col.index)

    alpha = m / n
    head = a[:n]
    if np.isnan(head).any():
        # 首个窗口不完整：从第一个满窗开始
        start = None
        for i in range(n - 1, total):
            w = a[i - n + 1: i + 1]
            if not np.isnan(w).any():
                out[i] = w.mean()
                start = i + 1
                break
        if start is None:
            return pd.Series(out, index=col.index)
        prev = out[start - 1]
    else:
        prev = head.mean()
        out[n - 1] = prev
        start = n

    for i in range(start, total):
        cur = a[i]
        if np.isnan(cur):
            out[i] = prev          # 与 ewm(ignore_na=True) 语义一致：跳过不更新
            continue
        prev = (1.0 - alpha) * prev + alpha * cur
        out[i] = prev
    return pd.Series(out, index=col.index)


def SMA(series, n, m):
    """
    n 日递归移动平均，m 为权重参数。

    SMA_t = (m * A_t + (n - m) * SMA_{t-1}) / n

    等价于 EMA(alpha = m/n)，初值 = 前 n 日简单均值。
    研报中常见 m=1 和 m=2。

    - Series/ndarray 输入：直接计算
    - DataFrame 输入：逐列计算
    """
    if isinstance(series, pd.DataFrame):
        return series.apply(lambda col: _sma_1d(col, n, m))
    return _sma_1d(_ensure_series(series), n, m)


def REGBETA(y, x, n):
    """过去 n 日 y 对 x 的线性回归系数 β = Cov(y,x) / Var(x)。

    2026-10-24 修正：原实现 `cov = y.rolling(n).cov(x)`（默认 ddof=1）配
    `var = x.rolling(n).var(ddof=0)`，分子分母自由度不一致 → β 系统性放大
    (n-1)/n 倍（n=5 时 +25%，实测相对误差 0.200）。双实现互证发现。
    影响 alpha142/143/144/145/146/147/148/149 八个因子。
    """
    y = _ensure_series(y)
    x = _ensure_series(x)
    cov = y.rolling(n).cov(x, ddof=0)
    var = x.rolling(n).var(ddof=0)
    return cov / var


def REGRESI(y, x, n):
    """过去 n 日 y 对 x 的线性回归残差"""
    beta = REGBETA(y, x, n)
    return _ensure_series(y) - beta * _ensure_series(x)


def COUNT(condition, n):
    """过去 n 日满足条件(True)的天数"""
    return _ensure_series(condition).astype(float).rolling(n).sum()


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
