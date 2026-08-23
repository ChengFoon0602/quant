"""
risk/crowding.py — 风格因子拥挤度测度与极端尾部风险预警引擎。

拥挤度测度（Lou & Polk 2013 范式与 A 股适应性拓展）：
  测度风格因子的横截面暴露集中度与相关性时序演变，作为市场结构输入与尾部风险预警。

四大核心维度（月末采样）：
  C1 factor_exposure_extreme_ratio : 月末截面 |z|>2 股票占比（极端风格暴露集中度）
  C2 style_homogeneity             : 滚动 12 月因子间两两 Rank IC 绝对值均值（同质化交易测度）
  C3 turnover_crowding             : 因子多空收益与市场换手代理的滚动相关性
  C4 factor_return_spike           : 因子多空收益滚动波动率（尖峰测度）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def month_end_dates(daily_index: pd.DatetimeIndex | pd.Index) -> pd.DatetimeIndex:
    """提取每月最后一个交易日的日期序列。"""
    s = pd.Series(pd.to_datetime(daily_index), index=pd.to_datetime(daily_index))
    grouped = s.groupby([s.dt.year, s.dt.month]).last()
    return pd.DatetimeIndex(grouped.values).sort_values()


def wide_to_long(tensor: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """将 {field: date×symbol 宽表} 字典转换为 MultiIndex (date, symbol) 的长表 DataFrame。"""
    frames = [df.stack().rename(f) for f, df in tensor.items()]
    return pd.concat(frames, axis=1)


def align_direction(
    tensor: Dict[str, pd.DataFrame],
    direction_map: Dict[str, str],
) -> Dict[str, pd.DataFrame]:
    """对负方向因子进行符号翻转（direction='-' 乘以 -1），统一语义为「高 z = 好方向/高暴露」。"""
    out = {}
    for f, df in tensor.items():
        sign = -1.0 if direction_map.get(f) == "-" else 1.0
        out[f] = df * sign if sign < 0 else df
    return out


def factor_exposure_extreme_ratio(
    X_long: pd.DataFrame,
    med: pd.DatetimeIndex,
    factor_cols: Optional[List[str]] = None,
    z_thresh: float = 2.0,
    min_stocks: int = 30,
) -> pd.Series:
    """C1: 月末截面 |z-score| > z_thresh 的股票占比。

    高值表示截面上有异常高比例的股票持有极端风格暴露，风格高度集中。
    """
    if factor_cols is None:
        factor_cols = list(X_long.columns)

    ratios = []
    dates = []
    for d in med:
        if d not in X_long.index.get_level_values(0):
            continue
        cross = X_long.xs(d, level=0)
        if len(cross) < min_stocks:
            continue
        fc = [c for c in factor_cols if c in cross.columns]
        if not fc:
            continue
        fz = cross[fc].sub(cross[fc].mean()).div(cross[fc].std().replace(0, np.nan))
        extreme = (fz.abs() > z_thresh).any(axis=1)
        ratios.append(extreme.mean())
        dates.append(d)

    return pd.Series(ratios, index=pd.DatetimeIndex(dates), name="C1_extreme_exposure").sort_index()


def factor_monthly_returns(
    X_long: pd.DataFrame,
    med: pd.DatetimeIndex,
    close_matrix: pd.DataFrame,
    fwd_days: int = 21,
    factor_cols: Optional[List[str]] = None,
    top_q: float = 0.20,
    bottom_q: float = 0.20,
    min_stocks: int = 30,
) -> pd.DataFrame:
    """计算因子在月末截面的未来月度多空收益序列（Top 分位 - Bottom 分位等权收益）。

    无未来函数对齐：因子暴露 t 月末 -> 收益 close(t + fwd_days + 1) / close(t + 1) - 1。
    """
    if factor_cols is None:
        factor_cols = list(X_long.columns)

    fwd = close_matrix.shift(-(fwd_days + 1)) / close_matrix.shift(-1) - 1
    rets = {}

    for f in factor_cols:
        fr = []
        for d in med:
            if d not in X_long.index.get_level_values(0):
                continue
            cross = X_long.xs(d, level=0)
            if f not in cross.columns:
                continue
            fv = cross[f].dropna()
            if len(fv) < min_stocks:
                continue

            top_thr = fv.quantile(1.0 - top_q)
            bot_thr = fv.quantile(bottom_q)

            top_symbols = fv[fv >= top_thr].index
            bot_symbols = fv[fv <= bot_thr].index

            if d not in fwd.index:
                continue
            fwd_d = fwd.loc[d]

            r_top = fwd_d.reindex(top_symbols).mean()
            r_bot = fwd_d.reindex(bot_symbols).mean()

            if np.isnan(r_top) or np.isnan(r_bot):
                continue
            fr.append((d, r_top - r_bot))

        if fr:
            rets[f] = pd.Series(dict(fr), name=f)

    return pd.DataFrame(rets).sort_index()


def style_homogeneity(
    factor_ret: pd.DataFrame,
    med: pd.DatetimeIndex,
    lookback: int = 12,
) -> pd.Series:
    """C2: 滚动 lookback 月因子间两两 Rank IC 绝对值均值。

    因子收益两两相关性越高，说明各风格因子同涨同跌，市场在风格维度趋于同质化拥挤。
    """
    if factor_ret.empty or factor_ret.shape[1] < 2:
        return pd.Series(dtype=float, index=med, name="C2_style_homogeneity")

    corrs = factor_ret.rank().rolling(lookback, min_periods=max(3, lookback // 2)).corr()
    n = factor_ret.shape[1]
    means = {}

    for d in med:
        if d not in corrs.index:
            continue
        try:
            mat = corrs.loc[d].values
            pairs = [abs(mat[i, j]) for i in range(n) for j in range(i + 1, n) if not np.isnan(mat[i, j])]
            if pairs:
                means[d] = float(np.mean(pairs))
        except Exception:
            continue

    return pd.Series(means, name="C2_style_homogeneity").sort_index()


def turnover_crowding(
    factor_ret: pd.DataFrame,
    turnover_series: pd.Series,
    med: pd.DatetimeIndex,
    lookback: int = 12,
) -> pd.Series:
    """C3: 绝对因子收益均值与市场换手率序列的滚动相关性。

    正相关越高表示因子超额收益与高换手同步脉冲，表明量能资金追逐该类风格。
    """
    if factor_ret.empty:
        return pd.Series(dtype=float, index=med, name="C3_turnover_crowding")

    abs_ret = factor_ret.abs().mean(axis=1)
    tv = turnover_series.reindex(abs_ret.index).ffill()
    c3 = abs_ret.rolling(lookback, min_periods=max(3, lookback // 2)).corr(tv)
    return c3.reindex([d for d in med if d in c3.index]).rename("C3_turnover_crowding").sort_index()


def factor_return_spike(
    factor_ret: pd.DataFrame,
    med: pd.DatetimeIndex,
    lookback: int = 6,
) -> pd.Series:
    """C4: 因子多空收益滚动波动率（跨因子均值）。正向波动尖峰常伴随极端拥挤或崩溃。"""
    if factor_ret.empty:
        return pd.Series(dtype=float, index=med, name="C4_return_spike")

    vol = factor_ret.rolling(lookback, min_periods=max(2, lookback // 2)).std().mean(axis=1)
    return vol.reindex([d for d in med if d in vol.index]).rename("C4_return_spike").sort_index()


def compute_crowding_indicators(
    X_long: pd.DataFrame,
    close_matrix: pd.DataFrame,
    turnover_series: pd.Series,
    factor_cols: Optional[List[str]] = None,
    z_thresh: float = 2.0,
    style_lookback: int = 12,
    spike_lookback: int = 6,
    min_stocks: int = 30,
) -> pd.DataFrame:
    """端到端计算四大拥挤度指标 (C1-C4)。

    Returns
    -------
    pd.DataFrame
        包含 C1_extreme_exposure, C2_style_homogeneity, C3_turnover_crowding, C4_return_spike。
    """
    dates = X_long.index.get_level_values(0).unique()
    med = month_end_dates(dates)

    fr = factor_monthly_returns(X_long, med, close_matrix, factor_cols=factor_cols, min_stocks=min_stocks)
    med_ret = fr.index if not fr.empty else med

    c1 = factor_exposure_extreme_ratio(X_long, med_ret, factor_cols=factor_cols, z_thresh=z_thresh, min_stocks=min_stocks)
    c2 = style_homogeneity(fr, med_ret, lookback=style_lookback)
    c3 = turnover_crowding(fr, turnover_series, med_ret, lookback=style_lookback)
    c4 = factor_return_spike(fr, med_ret, lookback=spike_lookback)

    df = pd.DataFrame({
        "C1_extreme_exposure": c1,
        "C2_style_homogeneity": c2,
        "C3_turnover_crowding": c3,
        "C4_return_spike": c4,
    }, index=med_ret).sort_index()
    df.index.name = "date"
    return df


def compute_composite_crowding(
    indicators_df: pd.DataFrame,
    expanding_min: int = 24,
) -> pd.DataFrame:
    """计算时序标准化（Expanding Z-Score）后的综合拥挤度指标。

    Returns
    -------
    pd.DataFrame
        原指标对应的 _z 列以及 composite_z (等权综合拥挤度 Z-score)。
    """
    df = indicators_df.copy()
    z_cols = []

    for col in ["C1_extreme_exposure", "C2_style_homogeneity", "C3_turnover_crowding", "C4_return_spike"]:
        if col in df.columns:
            s = df[col]
            # Expanding 均值与标准差，防止前瞻
            exp_mean = s.expanding(min_periods=expanding_min).mean()
            exp_std = s.expanding(min_periods=expanding_min).std().replace(0, np.nan)
            z_col = f"{col}_z"
            df[z_col] = (s - exp_mean) / exp_std
            z_cols.append(z_col)

    if z_cols:
        df["composite_z"] = df[z_cols].mean(axis=1)
    else:
        df["composite_z"] = np.nan

    return df


def detect_extreme_events(
    composite_z: pd.Series,
    quantile_thresh: float = 0.90,
    merge_window_months: int = 6,
) -> List[Tuple[pd.Timestamp, float]]:
    """检测综合拥挤度处于历史高位 (>quantile_thresh 分位数) 的极端事件，并合并相邻月份。

    Returns
    -------
    List[Tuple[pd.Timestamp, float]]
        (事件发生月末日期, 当期 composite_z) 列表。
    """
    s = composite_z.dropna().sort_index()
    if s.empty:
        return []

    cutoff = s.quantile(quantile_thresh)
    extreme_dates = s[s >= cutoff]

    events = []
    last_date = None

    for d, val in extreme_dates.items():
        dt = pd.Timestamp(d)
        if last_date is None:
            events.append((dt, float(val)))
            last_date = dt
        else:
            # 判断月份间隔
            month_diff = (dt.year - last_date.year) * 12 + (dt.month - last_date.month)
            if month_diff > merge_window_months:
                events.append((dt, float(val)))
                last_date = dt
            else:
                # 在同一窗口内保留更高的 z 值
                if val > events[-1][1]:
                    events[-1] = (dt, float(val))
                    last_date = dt

    return events
