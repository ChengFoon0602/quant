"""
event_study.py — 极端拥挤尾部风险事件研究（方向C 延伸 ②）。

把「综合拥挤度 >90 分位 → 后续因子回撤」从描述变成量化事件研究：
以极端拥挤日为锚，统计其后 3/6/12 月因子收益，bootstrap 检验是否显著异于常态。
这是弧线里唯一有实用价值的产出——风控输入（非 alpha，非 trading signal）。

方法：
  - extreme_events: 综合拥挤度 >threshold 分位的事件日，相邻 6 月内合并为一个事件
  - event_windows: 每个事件后 horizons 月因子累计收益（领先-滞后，无前瞻）
  - event_study_bootstrap: bootstrap 检验事件后收益 vs 常态分布（随机锚点对照）

用法:
    from event_study import extreme_events, event_windows, event_study_bootstrap
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def composite_crowding(crowd_ts: pd.DataFrame) -> pd.Series:
    """综合拥挤度 = C1-C4 标准化均值。"""
    z = crowd_ts.sub(crowd_ts.mean()).div(crowd_ts.std())
    return z.mean(axis=1)


def extreme_events(composite: pd.Series, threshold: float = 0.9,
                   min_gap_months: int = 6) -> list[pd.Timestamp]:
    """综合拥挤度 >threshold 分位的事件日列表（相邻 min_gap_months 月内合并）。

    只取每个事件簇的最高点作为代表日（避免同一段拥挤期重复计数）。
    """
    thr = composite.quantile(threshold)
    high = composite[composite > thr]
    if high.empty:
        return []
    # 事件簇：时间差超过 min_gap_months 月视为新事件
    events = []
    cluster = []
    for d in high.index:
        if cluster and (d - cluster[-1]).days > 30 * min_gap_months:
            # 簇内取最高拥挤度日
            events.append(max(cluster, key=lambda x: composite[x]))
            cluster = []
        cluster.append(d)
    if cluster:
        events.append(max(cluster, key=lambda x: composite[x]))
    return events


def event_windows(event_dates: list[pd.Timestamp], factor_ret: pd.DataFrame,
                  horizons: list[int] | None = None) -> pd.DataFrame:
    """每个事件后 horizons 月因子累计收益（跨因子均值，领先-滞后无前瞻）。

    factor_ret: date×factor 月末多空收益（月频）。h 月窗口 = 事件后 1..h 月收益累乘。
    输出：DataFrame(event_date, horizon, cum_ret, event_peaks)。
    """
    if horizons is None:
        horizons = [3, 6, 12]
    mean_ret = factor_ret.mean(axis=1)  # 跨因子月均多空收益
    rows = []
    for ev in event_dates:
        pos = mean_ret.index.get_indexer([ev], method="nearest")[0]
        for h in horizons:
            end = min(pos + h, len(mean_ret) - 1)
            if end <= pos:
                continue
            win = mean_ret.iloc[pos + 1:end + 1]
            if len(win) == 0:
                continue
            cum = float((1 + win).prod() - 1)
            rows.append({"event_date": ev, "horizon": h, "cum_ret": cum})
    return pd.DataFrame(rows)


def event_study_bootstrap(events_fwd: pd.Series, normal_series: pd.Series,
                          n_boot: int = 10000, block: int = 12, seed: int = 0) -> dict:
    """bootstrap 检验：事件后收益均值是否显著异于常态分布。

    events_fwd: 事件后某 horizon 的累计收益序列（可能很少，如实标注）
    normal_series: 全部月末的因子月均收益（常态池，用于 block bootstrap 对照）
    返回 {n_events, event_mean, normal_mean, p_value(单侧: 事件更差), std_normal}
    """
    rng = np.random.default_rng(seed)
    arr = normal_series.dropna().values
    n = len(arr)
    boot_means = []
    for _ in range(n_boot):
        # block bootstrap（随机起点，block 大小滚动拼接至 n）
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = []
        for s in starts:
            idx.extend(range(s, min(s + block, n)))
        idx = np.array(idx[:n])
        boot_means.append(np.mean(arr[idx]))
    boot_means = np.array(boot_means)
    n_ev = len(events_fwd)
    ev_mean = events_fwd.mean()
    # 单侧 p：常态 bootstrap 分布中「均值 ≤ 事件均值」的比例（事件是否显著更差/更好）
    p_worse = float((boot_means <= ev_mean).mean())
    return {
        "n_events": n_ev,
        "event_mean": ev_mean,
        "normal_mean": float(np.mean(arr)),
        "normal_std": float(np.std(arr)),
        "p_worse": p_worse,
        "std_boot": float(boot_means.std()),
    }


def summarize(event_df: pd.DataFrame, normal_series: pd.Series,
              n_boot: int = 10000) -> pd.DataFrame:
    """按 horizon 汇总事件研究结果（事件均值 + bootstrap 对照）。"""
    rows = []
    for h, grp in event_df.groupby("horizon"):
        ev = grp["cum_ret"]
        # 常态对照：全部月末因子月均收益 → h 月累乘的分布
        m = normal_series.dropna()
        boot = np.random.default_rng(0)
        n = len(m)
        normal_h = []
        for _ in range(500):
            s = boot.integers(0, n, size=h)
            normal_h.append(float((1 + m.iloc[s]).prod() - 1))
        normal_h = np.array(normal_h)
        p = float((normal_h >= ev.mean()).mean())  # 常态 ≥ 事件 的比例 = 事件显著差于常态
        rows.append({
            "horizon_months": h, "n_events": len(ev),
            "event_mean_cum": ev.mean(), "event_median": ev.median(),
            "normal_mean_cum": normal_h.mean(), "normal_std": normal_h.std(),
            "p_worse_than_normal": p,
        })
    return pd.DataFrame(rows)
