"""
report.py — 方向C 中证500 因子拥挤度时序研究（市场结构测度）驱动。

流程：
  1. 读 X_matrix.csv（16 因子暴露）+ PIT 面板（close/volume/member）
  2. 算 4 个拥挤度指标（C1-C4）→ 月末时序 crowding_time_series.csv
  3. 条件收益分析：拥挤度分桶（Q1-Q4）→ 未来 12 月因子收益
  4. 图：拥挤度综合时序 + 条件收益柱状图
  5. print 全部关键数值（供 report.md 引用）

用法:
    cd strategies/zz500_crowding_trial && python report.py
"""

from __future__ import annotations

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    THIS_DIR, FEATURE_SEL_DIR, PROJECT_ROOT, FIGURES_DIR,
    PIT_TRIAL_DIR, INDEX, MARKET_EVENTS, CONDITION_LOOKBACK,
    FACTOR_COLS,
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FEATURE_SEL_DIR))

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from build_pit_matrix import load_pit_panel
from crowding import compute_all, load_factor_matrix, month_end_dates


def section(title: str):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def conditional_return_analysis(crowd_ts: pd.DataFrame, factor_ret: pd.DataFrame,
                                lookback: int = CONDITION_LOOKBACK) -> pd.DataFrame:
    """拥挤度分桶 → 未来 lookback 月平均因子收益。

    用综合拥挤度（C1-C4 标准化均值）在 t 分桶，收益 = t+1..t+lookback 的
    factor_ret 均值（领先-滞后，无前瞻）。输出每桶的均值/t 值/样本数。
    """
    # 综合拥挤度（C1-C4 标准化均值）
    z = crowd_ts.sub(crowd_ts.mean()).div(crowd_ts.std())
    composite = z.mean(axis=1)
    # 未来收益（滚动求和，向前）
    ret_roll = factor_ret.mean(axis=1)  # 16 因子月均多空收益
    fwd = ret_roll.shift(-1).rolling(lookback).mean().shift(-(lookback - 1))[::-1]

    # 对齐
    common = composite.index.intersection(fwd.dropna().index)
    comp = composite.loc[common]
    fw = fwd.loc[common]

    # 分桶
    q = pd.qcut(comp.rank(method="first"), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    rows = []
    for bucket in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        sel = fw[q == bucket]
        rows.append({
            "bucket": bucket,
            "mean": sel.mean(), "t": sel.mean() / (sel.std() / np.sqrt(len(sel))) if len(sel) > 1 else np.nan,
            "n": len(sel), "median": sel.median(),
        })
    return pd.DataFrame(rows)


def plot_crowding_timeseries(crowd_ts: pd.DataFrame):
    """图1: 4 拥挤度指标标准化时序 + 市场大事件标注。"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    z = crowd_ts.sub(crowd_ts.mean()).div(crowd_ts.std())

    ax = axes[0]
    for col in crowd_ts.columns:
        ax.plot(z.index, z[col].values, label=col, linewidth=1.2)
    for ev_date, ev_name in MARKET_EVENTS.items():
        ax.axvline(pd.Timestamp(ev_date), color="red", linestyle="--", linewidth=0.8)
        ax.text(pd.Timestamp(ev_date), ax.get_ylim()[1] * 0.95, ev_name,
                rotation=90, fontsize=8, color="red")
    ax.set_title("中证500 因子拥挤度时序（C1-C4 标准化，2010-2025）")
    ax.set_xlabel("date"); ax.set_ylabel("z-score")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[1]
    composite = z.mean(axis=1)
    ax.plot(composite.index, composite.values, color="#c44e52", linewidth=1.5,
            label="综合拥挤度（C1-C4 均值）")
    for ev_date, ev_name in MARKET_EVENTS.items():
        ax.axvline(pd.Timestamp(ev_date), color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("综合拥挤度 + 市场大事件（2015 股灾 / 2021 核心资产 / 2024 微盘）")
    ax.set_xlabel("date"); ax.set_ylabel("综合拥挤度 z-score")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "01_crowding_timeseries.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {path}")


def plot_conditional_returns(cond_df: pd.DataFrame):
    """图2: 拥挤度分桶的条件因子收益柱状图。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    buckets = cond_df["bucket"]
    vals = cond_df["mean"] * 100  # %
    colors = ["#c44e52" if b == "Q4_high" else "#4c72b0" for b in buckets]
    ax.bar(buckets, vals, color=colors)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(f"拥挤度分桶 → 未来 {CONDITION_LOOKBACK} 月因子多空收益")
    ax.set_xlabel("拥挤度分桶（Q1=低 → Q4=高）"); ax.set_ylabel("未来月均收益 (%)")
    ax.grid(True, alpha=0.3, axis="y")
    # t 值标注
    for i, r in cond_df.iterrows():
        ax.text(i, (r["mean"]) * 100 + 0.05, f"t={r['t']:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    path = FIGURES_DIR / "02_conditional_returns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {path}")


def plot_crowding_comparison(crowd_v: pd.DataFrame, crowd_f: pd.DataFrame):
    """图3: 量价 vs 基本面综合拥挤度对比时序。"""
    fig, ax = plt.subplots(figsize=(14, 5))
    z_v = crowd_v.sub(crowd_v.mean()).div(crowd_v.std()).mean(axis=1)
    z_f = crowd_f.sub(crowd_f.mean()).div(crowd_f.std()).mean(axis=1)
    ax.plot(z_v.index, z_v.values, label="量价 16 因子（动量类）", color="#c44e52", linewidth=1.3)
    ax.plot(z_f.index, z_f.values, label="基本面 20 因子（估值/质量）", color="#4c72b0", linewidth=1.3)
    for ev_date, ev_name in MARKET_EVENTS.items():
        ax.axvline(pd.Timestamp(ev_date), color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("量价 vs 基本面因子综合拥挤度对比（2010-2025）")
    ax.set_xlabel("date"); ax.set_ylabel("综合拥挤度 z-score")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "03_crowding_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {path}")


def plot_event_study(ev_summary: pd.DataFrame):
    """图4: 极端拥挤事件后 3/6/12 月因子收益 vs 常态。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ev_summary))
    w = 0.35
    ev = ev_summary["event_mean_cum"].values
    nm = ev_summary["normal_mean_cum"].values
    ax.bar(x - w / 2, ev, w, label="极端拥挤后", color="#c44e52")
    ax.bar(x + w / 2, nm, w, label="常态对照", color="#4c72b0")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} 月" for h in ev_summary["horizon_months"]])
    ax.set_ylabel("累计因子收益")
    ax.set_title("极端拥挤（>90 分位）后因子收益 vs 常态（bootstrap 对照）")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    for i, r in ev_summary.iterrows():
        ax.text(i - w / 2, r["event_mean_cum"] + 0.01, f"n={int(r['n_events'])}",
                ha="center", fontsize=7)
        ax.text(i + w / 2, r["normal_mean_cum"] + 0.01, f"p={r['p_worse_than_normal']:.2f}",
                ha="center", fontsize=7)
    plt.tight_layout()
    path = FIGURES_DIR / "04_event_study.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {path}")


def main():
    section("方向C：中证500 因子拥挤度时序研究")
    print("[1] 加载数据（量价池）...")
    X_long = load_factor_matrix(PIT_TRIAL_DIR / "X_matrix.csv")
    close, volume, member = load_pit_panel(INDEX)
    turnover = pd.read_csv(PIT_TRIAL_DIR / "X_matrix.csv") \
        .groupby("date")["market_turnover_20d"].last()  # 市场换手代理
    turnover.index = pd.to_datetime(turnover.index)
    print(f"  X_matrix: {X_long.shape} | 面板 {close.shape}")

    print("[2] 计算量价池拥挤度（C1-C4）...")
    from crowding import compute_all, factor_monthly_returns
    crowd_v = compute_all(X_long, close, turnover)
    crowd_v.to_csv(THIS_DIR / "crowding_time_series.csv")
    med = month_end_dates(X_long.index.get_level_values(0).unique())
    fr_v = factor_monthly_returns(X_long, med, close)
    print(f"  量价时序: {crowd_v.shape}（{crowd_v.index[0].date()} → {crowd_v.index[-1].date()}）")
    for col in crowd_v.columns:
        print(f"    {col}: mean={crowd_v[col].mean():.4f}")

    print("[3] 条件收益（量价池，拥挤度分桶 → 未来 12 月）...")
    cond = conditional_return_analysis(crowd_v, fr_v)
    print(cond.to_string(index=False))

    print("[4] 基本面池拥挤度（对比）...")
    from fundamental_crowding import load_fundamental_long, FUNDAMENTAL_COLS
    X_fund, close_f = load_fundamental_long()
    from build_pit_matrix import build_market_features_pit
    _, vol_f, mem_f = load_pit_panel(INDEX)
    mkt_f = build_market_features_pit(close_f, vol_f, mem_f)
    crowd_f = compute_all(X_fund, close_f, mkt_f["market_turnover_20d"], factor_cols=FUNDAMENTAL_COLS)
    crowd_f.to_csv(THIS_DIR / "fundamental_crowding_time_series.csv")
    med_f = month_end_dates(X_fund.index.get_level_values(0).unique())
    fr_f = factor_monthly_returns(X_fund, med_f, close_f, factor_cols=FUNDAMENTAL_COLS)
    cond_f = conditional_return_analysis(crowd_f, fr_f)
    print("  基本面池条件收益：")
    print(cond_f.to_string(index=False))

    print("[5] 事件研究（极端拥挤 >90 分位 → 后续因子收益）...")
    from event_study import composite_crowding, extreme_events, event_windows, summarize
    for name, crowd, fr in [("量价", crowd_v, fr_v), ("基本面", crowd_f, fr_f)]:
        comp = composite_crowding(crowd)
        events = extreme_events(comp)
        ev_df = event_windows(events, fr)
        ev_sum = summarize(ev_df, fr.mean(axis=1))
        print(f"  [{name}] 极端事件 {len(events)} 个: "
              + ", ".join(f"{e.date()}" for e in events))
        print(ev_sum.to_string(index=False))
        ev_sum.to_csv(THIS_DIR / f"event_study_{name}.csv", index=False)

    print("[6] 生成图表...")
    plot_crowding_timeseries(crowd_v)
    plot_conditional_returns(cond)
    plot_crowding_comparison(crowd_v, crowd_f)
    # 事件研究图（量价池）
    comp_v = composite_crowding(crowd_v)
    ev_sum_v = summarize(event_windows(extreme_events(comp_v), fr_v), fr_v.mean(axis=1))
    plot_event_study(ev_sum_v)

    section("完成")
    print(f"时序保存: {THIS_DIR}/crowding_time_series.csv + fundamental_crowding_time_series.csv")


if __name__ == "__main__":
    main()
