"""
crowding.py — 因子拥挤度时序指标（方向C 核心，量价/基本面两池通用）。

拥挤度 = 风格因子的横截面集中度/极端度随时间的变化（Lou & Polk 2013 风格）。
不是 trading signal，是市场结构测度：多少人挤在同一条腿 → 拥挤后表现如何。

四个指标（全部月末采样，复用方向2 的 month_end_dates）：
  C1 factor_exposure_extreme_ratio : 月末截面 |z|>2 股票占比（极端暴露集中度）
  C2 style_homogeneity             : 滚动 12 月两两 Rank IC |均值|
                                     风格间同涨同跌 = 拥挤
  C3 turnover_crowding             : 因子收益与市场换手代理的滚动相关
  C4 factor_return_spike           : 因子多空收益 rolling 波动（正尖峰 = 崩溃前兆）

通用化设计（2026-08-14 扩展）：
  - 所有函数接受 factor_cols 参数（默认量价 16 因子，可传基本面 20 因子）
  - 输入统一为长表（MultiIndex date,symbol，列=因子），wide_to_long 负责把
    compute_factor_tensor 的 {field: date×symbol} 宽表转成长表
  - align_direction 做方向翻转：负方向因子（如 pbMRQ 低=好）×−1 使「高 z = 好方向」，
    跨因子比较拥挤度才一致。量价池（X_matrix 已标准化）不需要。
  - style_homogeneity 用 factor_ret.shape[1]（不再硬编码 16）

方法论纪律：
  - 全部用截至 t 的数据算（无前瞻）。C1 用 t 月末截面，C2-C4 用 t 及之前的
    历史滚动窗口。条件收益分析在 report.py 用领先-滞后（拥挤度 t → 收益 t+1..t+12）。
  - 因子收益：月末截面 zscore 多空（top20% - bottom20%）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    Z_EXTREME_THRESHOLD, STYLE_LOOKBACK, SPIKE_LOOKBACK, FACTOR_COLS,
)


def load_factor_matrix(path) -> pd.DataFrame:
    """读取 X_matrix.csv → 月末截面长表 pivot 为 date×factor 宽表 + 返回原始长表。"""
    X = pd.read_csv(path, index_col=[0, 1], parse_dates=[0])
    X.index = X.index.set_levels(X.index.levels[1].astype(str), level=1)
    return X


def wide_to_long(tensor: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{field: date×symbol 宽表} → 长表（MultiIndex date,symbol，列=因子）。

    用于把 compute_factor_tensor 的输出（基本面因子）对齐到 X_matrix 长表格式。
    """
    frames = [df.stack().rename(f) for f, df in tensor.items()]
    return pd.concat(frames, axis=1)


def align_direction(tensor: dict[str, pd.DataFrame],
                    direction_map: dict[str, str]) -> dict[str, pd.DataFrame]:
    """方向翻转：direction='-' 的因子 ×−1，使「高 z = 好方向」跨因子一致。

    direction_map: {field: '+'|'-'}。负方向因子（如 pbMRQ 低估值=好）需翻转，
    否则因子拥挤度「高 z」含义不一致（对估值因子会变成「贵=拥挤」而非「便宜=拥挤」）。
    """
    out = {}
    for f, df in tensor.items():
        sign = -1.0 if direction_map.get(f) == "-" else 1.0
        out[f] = df * sign if sign < 0 else df
    return out


def month_end_dates(daily_index) -> pd.DatetimeIndex:
    """每月最后一个交易日的日期序列（复用方向2 语义）。"""
    s = pd.Series(daily_index, index=daily_index)
    grouped = s.groupby([s.dt.year, s.dt.month]).last()
    return pd.DatetimeIndex(grouped.values)


def factor_exposure_extreme_ratio(X_long: pd.DataFrame, med: pd.DatetimeIndex,
                                  factor_cols: list[str] | None = None,
                                  z_thresh: float = Z_EXTREME_THRESHOLD) -> pd.Series:
    """C1: 月末截面 |z-score|>2 的股票占比（对 factor_cols）。

    逐月末，对每只股票 factor_cols 因子暴露做横截面 zscore，统计任一因子 |z|>2 的股票比例。
    高 = 很多人持有极端风格暴露 = 拥挤。
    """
    if factor_cols is None:
        factor_cols = FACTOR_COLS
    ratios = []
    dates = []
    for d in med:
        cross = X_long.xs(d, level="date")
        if len(cross) < 30:
            continue
        fc = [c for c in factor_cols if c in cross.columns]
        if not fc:
            continue
        fz = cross[fc].sub(cross[fc].mean()).div(cross[fc].std())
        extreme = (fz.abs() > z_thresh).any(axis=1)
        ratios.append(extreme.mean())
        dates.append(d)
    return pd.Series(ratios, index=pd.DatetimeIndex(dates), name="C1_extreme_exposure")


def factor_monthly_returns(X_long: pd.DataFrame, med: pd.DatetimeIndex,
                           close_matrix: pd.DataFrame, fwd_days: int = 21,
                           factor_cols: list[str] | None = None) -> pd.DataFrame:
    """factor_cols 因子的月末多空收益（top20% - bottom20% zscore 等权）。

    fwd_return(close, 21) 对齐月末：因子暴露 t → 收益 t+1..t+22。
    输出：date×factor 的多空收益（不含成本，仅测度结构用）。
    """
    if factor_cols is None:
        factor_cols = FACTOR_COLS
    # fwd_return(close, 21) = close(t+22)/close(t+1)-1（与方向2 purify.fwd_return 一致）
    fwd = close_matrix.shift(-(fwd_days + 1)) / close_matrix.shift(-1) - 1
    rets = {}
    for f in factor_cols:
        fr = []
        for d in med:
            if d not in X_long.index.get_level_values(0):
                continue
            cross = X_long.xs(d, level="date")
            if f not in cross.columns:
                continue
            fv = cross[f]
            if fv.notna().sum() < 30:
                continue
            z = fv.sub(fv.mean()).div(fv.std())
            top = z >= z.quantile(0.8)
            bot = z <= z.quantile(0.2)
            fwd_d = fwd.loc[d] if d in fwd.index else None
            if fwd_d is None or fwd_d.notna().sum() < 30:
                continue
            # 按 symbol 对齐（top/bot 来自 X 的 symbol 集合，fwd 来自 close 全集合）
            top = top.reindex(fwd_d.index).fillna(False).astype(bool)
            bot = bot.reindex(fwd_d.index).fillna(False).astype(bool)
            r_top = fwd_d[top].mean()
            r_bot = fwd_d[bot].mean()
            if np.isnan(r_top) or np.isnan(r_bot):
                continue
            fr.append((d, r_top - r_bot))
        if fr:
            rets[f] = pd.Series(dict(fr), name=f)
    return pd.DataFrame(rets).sort_index()


def style_homogeneity(factor_ret: pd.DataFrame, med: pd.DatetimeIndex,
                      lookback: int = STYLE_LOOKBACK) -> pd.Series:
    """C2: 滚动 12 月两两 Rank IC 绝对值均值 → 风格同质化。

    因子收益两两相关越高 = 风格间同涨同跌 = 拥挤（大家都在同一维度）。
    n 用 factor_ret.shape[1]（通用，不再硬编码 16）。
    """
    corrs = factor_ret.rank().rolling(lookback).corr()
    n = factor_ret.shape[1]
    means = []
    for d in med:
        if d not in corrs.index:
            continue
        pairs = [corrs.loc[d].iloc[i, j].real for i in range(n)
                 for j in range(i + 1, n)]
        if pairs:
            means.append((d, np.nanmean(np.abs(pairs))))
    return pd.Series(dict(means), name="C2_style_homogeneity").sort_index()


def turnover_crowding(factor_ret: pd.DataFrame, turnover: pd.Series,
                      med: pd.DatetimeIndex,
                      lookback: int = STYLE_LOOKBACK) -> pd.Series:
    """C3: 因子收益与市场换手代理的滚动相关（绝对因子收益均值 vs 换手）。

    因子收益脉冲 + 高换手 = 量能驱动的拥挤（追涨杀跌在风格层面）。
    """
    abs_ret = factor_ret.abs().mean(axis=1)
    tv = turnover.reindex(abs_ret.index).ffill()
    c3 = abs_ret.rolling(lookback).corr(tv)
    return c3.reindex([d for d in med if d in c3.index]).rename("C3_turnover_crowding")


def factor_return_spike(factor_ret: pd.DataFrame, med: pd.DatetimeIndex,
                        lookback: int = SPIKE_LOOKBACK) -> pd.Series:
    """C4: 因子多空收益 rolling 波动（跨因子均值）。正尖峰 = 拥挤崩溃前兆。"""
    vol = factor_ret.rolling(lookback).std().mean(axis=1)
    return vol.reindex([d for d in med if d in vol.index]).rename("C4_return_spike")


def compute_all(X_long: pd.DataFrame, close_matrix: pd.DataFrame,
                turnover: pd.Series, factor_cols: list[str] | None = None) -> pd.DataFrame:
    """计算全部 4 个拥挤度指标 → 月末时序表。"""
    med = month_end_dates(X_long.index.get_level_values(0).unique())
    fr = factor_monthly_returns(X_long, med, close_matrix, factor_cols=factor_cols)
    med_ret = fr.index
    ts = pd.DataFrame(index=med_ret)
    ts["C1_extreme_exposure"] = factor_exposure_extreme_ratio(X_long, med_ret, factor_cols)
    ts["C2_style_homogeneity"] = style_homogeneity(fr, med_ret)
    ts["C3_turnover_crowding"] = turnover_crowding(fr, turnover, med_ret)
    ts["C4_return_spike"] = factor_return_spike(fr, med_ret)
    ts.index.name = "date"
    return ts
