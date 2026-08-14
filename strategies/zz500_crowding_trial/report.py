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


def main():
    section("方向C：中证500 因子拥挤度时序研究")
    print("[1] 加载数据...")
    X_long = load_factor_matrix(PIT_TRIAL_DIR / "X_matrix.csv")
    close, volume, member = load_pit_panel(INDEX)
    turnover = pd.read_csv(PIT_TRIAL_DIR / "X_matrix.csv") \
        .groupby("date")["market_turnover_20d"].last()  # 市场换手代理
    turnover.index = pd.to_datetime(turnover.index)
    print(f"  X_matrix: {X_long.shape} | 面板 {close.shape}")

    print("[2] 计算拥挤度指标（C1-C4）...")
    crowd_ts = compute_all(X_long, close, turnover)
    crowd_ts.to_csv(THIS_DIR / "crowding_time_series.csv")
    print(f"  时序: {crowd_ts.shape}（{crowd_ts.index[0].date()} → {crowd_ts.index[-1].date()}）")
    for col in crowd_ts.columns:
        print(f"    {col}: mean={crowd_ts[col].mean():.4f} std={crowd_ts[col].std():.4f} "
              f"max={crowd_ts[col].max():.4f}")

    print("[3] 条件收益分析（拥挤度分桶 → 未来 12 月因子收益）...")
    factor_ret = load_factor_matrix(PIT_TRIAL_DIR / "X_matrix.csv")  # 重读（占内存）
    from crowding import factor_monthly_returns
    med = month_end_dates(X_long.index.get_level_values(0).unique())
    fr = factor_monthly_returns(X_long, med, close)
    cond = conditional_return_analysis(crowd_ts, fr)
    print(cond.to_string(index=False))

    print("[4] 生成图表...")
    plot_crowding_timeseries(crowd_ts)
    plot_conditional_returns(cond)

    section("完成")
    print(f"时序保存: {THIS_DIR}/crowding_time_series.csv")


if __name__ == "__main__":
    main()
