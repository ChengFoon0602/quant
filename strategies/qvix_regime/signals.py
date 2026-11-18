"""
strategies/qvix_regime/signals.py — QVIX 波动率状态信号。

## 核心约束：分位必须滚动

`qvix_regime` 的全部信号都由**滚动分位**构造，绝不使用全样本分位。
全样本分位（`s.rank(pct=True)`）在 t 日已经用到 t 之后的观测值，是典型的
未来函数 —— 探索阶段看分布形状可以用，进入回测必须换成 `rolling_percentile`。

## 时点约定

`build_weight_portfolio` 内部 `W_lag = W_held.shift(1)`，即 t 日目标权重
只在 t+1 日产生收益。因此本模块的 gate 允许使用 **t 日收盘**的 QVIX，
与 `market_trend_gate()` 用 t 日收盘价判 MA 的口径一致，不构成前视。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def rolling_percentile(
    s: pd.Series,
    window: int = 252,
    min_periods: int = 120,
) -> pd.Series:
    """滚动历史分位：当日值在过去 `window` 个观测中的分位（0~1）。

    实现为「过去窗口内严格小于当日值的占比」，逐窗口计算，不使用未来数据。
    `window=252` 对应约一年。
    """
    def _pct(a: np.ndarray) -> float:
        return float((a[:-1] < a[-1]).mean()) if len(a) > 1 else np.nan

    return s.rolling(window, min_periods=min_periods).apply(_pct, raw=True)


def qvix_binary_gate(
    iv: pd.Series,
    quantile: float = 0.90,
    window: int = 252,
    min_periods: int = 120,
) -> pd.Series:
    """二值闸门：滚动分位低于 `quantile` 时开仓（1.0），否则清仓（0.0）。

    默认 0.90 来自 report G3 的尾部观测：IV 处于历史前 10% 时，
    未来 60 日收益显著为负，而 10%~90% 区间内无单调区分度。
    """
    pct = rolling_percentile(iv, window=window, min_periods=min_periods)
    # 分位无法计算时（样本前段）默认开仓，避免人为空仓抬升夏普
    return (pct < quantile).astype(float).where(pct.notna(), 1.0)


def qvix_position_scale(
    iv: pd.Series,
    window: int = 252,
    min_periods: int = 120,
    floor: float = 0.0,
) -> pd.Series:
    """连续仓位系数：分位越高仓位越低，`scale = 1 - pct`，下限 `floor`。

    用于检验「渐进减仓」是否优于「二值清仓」。
    """
    pct = rolling_percentile(iv, window=window, min_periods=min_periods)
    scale = (1.0 - pct).clip(lower=floor, upper=1.0)
    return scale.where(pct.notna(), 1.0)


def iv_rv_spread(iv: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """方差风险溢价代理：IV − RV（点）。

    正值表示期权相对历史波动「贵」，通常对应恐慌情绪；
    负值出现在急跌初期（RV 上冲快于 IV）。
    """
    from data.qvix_fetcher import realized_vol

    return iv - realized_vol(close, window=window)


def calibrate_quantile_for_exposure(
    iv: pd.Series,
    target_mean_gate: float,
    window: int = 252,
    min_periods: int = 120,
    lo: float = 0.01,
    hi: float = 0.999,
    tol: float = 1e-3,
) -> float:
    """二分搜索阈值，使二值闸门的**平均持仓比例**等于 `target_mean_gate`。

    用途：`build_weight_portfolio` 的 gate 只作用于权益腿，关闸越多则权益
    平均暴露越低、波动越低、夏普可能机械性上升。要在同等暴露下比较两个
    gate 的优劣，必须先做暴露匹配（见 report G4「匹配仓位对照组」）。

    Notes
    -----
    暴露的定义必须与 `qvix_binary_gate()` 完全一致：**分位为 NaN 的样本段
    按开仓（1.0）计**。若只在 `pct.dropna()` 上算暴露，样本前 252 天会被
    漏掉，导致搜索区间错位、阈值卡在边界（实测曾返回 0.50 并给出错误暴露）。
    """
    if not 0.0 < target_mean_gate <= 1.0:
        raise ValueError(f"target_mean_gate 必须在 (0, 1]，得到 {target_mean_gate}")

    pct = rolling_percentile(iv, window=window, min_periods=min_periods)
    if pct.dropna().empty:
        raise ValueError("分位序列为空，无法校准阈值")

    def _exposure(q: float) -> float:
        g = (pct < q).astype(float).where(pct.notna(), 1.0)
        return float(g.mean())

    # 目标超出可达范围时直接返回端点，避免二分退化成一个错误值
    if _exposure(hi) < target_mean_gate:
        return float(hi)
    if _exposure(lo) > target_mean_gate:
        return float(lo)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        exposure = _exposure(mid)
        if abs(exposure - target_mean_gate) < tol:
            return float(mid)
        if exposure < target_mean_gate:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def combine_gates(*gates: Optional[pd.Series]) -> pd.Series:
    """多个闸门取交集（全部为 1 才开仓）。

    缺失的闸门（None）被忽略；返回序列索引为各闸门索引的并集，缺失值按 1.0 处理
    （与 `build_weight_portfolio` 内部 `gate.reindex(...).fillna(1.0)` 语义一致）。
    """
    valid = [g for g in gates if g is not None]
    if not valid:
        raise ValueError("至少需要一个闸门")
    idx = valid[0].index
    for g in valid[1:]:
        idx = idx.union(g.index)
    out = pd.Series(1.0, index=idx)
    for g in valid:
        out = out * g.reindex(idx).fillna(1.0)
    return out
