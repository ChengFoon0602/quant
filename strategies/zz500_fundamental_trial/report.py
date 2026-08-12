"""
report.py — 方向2 全链驱动（PIT 基本面 × 中证500）。

编排顺序（与价量链路 report.py 对齐）:
  提纯（月末截面）→ 月末 X/y → 月末 LGBM Purged CV → 月调仓回测
  → WF PIT-Select（年度重选池）→ 图表 + 指标打印

支持分阶段运行（数据未齐时便于逐步验证）:
    python report.py                 # 全链
    python report.py --stage purify  # 只跑提纯
    python report.py --stage train   # 只跑训练（复用已建矩阵）

用法:
    cd strategies/zz500_fundamental_trial && python report.py
"""

from __future__ import annotations

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    THIS_DIR, FEATURE_SEL_DIR, PROJECT_ROOT, FIGURES_DIR,
    INDEX, FWD_DAYS, COST_BPS, TOP_Q,
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FEATURE_SEL_DIR))

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from build_pit_matrix import load_pit_panel
from models.portfolio_backtest import performance_metrics


def section(title: str):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def load_pool_from_purify() -> list[str]:
    pur = pd.read_csv(THIS_DIR / "purify_results_monthly.csv")
    passed = pur[pur["pass"]]
    if passed.empty:
        print("[WARN] 无四维通过因子，用 |IC_IR| 前 10 候选诊断")
        return pur.sort_values("IC_IR", key=abs, ascending=False)["factor"].head(10).tolist()
    return passed.sort_values("IC_IR", key=abs, ascending=False)["factor"].head(10).tolist()


def stage_purify(close, volume, member):
    from purify import purify_and_select, plot_purify
    final_pool, purify_df, corr_mat, factor_tensor, keep_tensor = purify_and_select(
        close, volume, member)
    plot_purify(purify_df)
    n_pass = int(purify_df["pass"].sum())
    n_sign = int(purify_df["sign_ok"].sum())
    print(f"\n  [提纯] 通过 {n_pass}/{len(purify_df)} | 方向一致 {n_sign}/{len(purify_df)}")
    return final_pool, factor_tensor


def stage_matrix(close, volume, member, final_pool):
    from build_monthly_matrix import build_monthly_matrix
    from signals.fundamental.factors import compute_factor_tensor
    factor_tensor = compute_factor_tensor(close, final_pool)
    factor_tensor = {f: df.where(member) for f, df in factor_tensor.items()}
    X, y = build_monthly_matrix(close, volume, member, factor_tensor, final_pool)
    return X


def stage_train(close):
    from train_cv import load_monthly_matrix, train_cv
    X, y = load_monthly_matrix()
    pred_matrix, metrics, fi, auc_m, auc_s = train_cv(X, close)
    return pred_matrix, metrics, fi


def stage_backtest(close, member, pred_matrix):
    import backtest_monthly as bm
    from backtest_monthly import ffill_to_daily, yearly_metrics
    pred_daily = ffill_to_daily(pred_matrix, close)
    lo = bm.build_portfolio(pred_daily, close, long_only=True, cost=COST_BPS, hold_days=1)
    m_lo = performance_metrics(lo["port_ret"])
    yr = yearly_metrics(lo["port_ret"])
    print(f"\n  [回测] LO 年化 {m_lo['annual']:+.2%} 夏普 {m_lo['sharpe']:+.3f}")
    return lo, m_lo, yr


def plot_summary(m_lo, yr, metrics, m_ew, m_mkt, p_lo, p_ew):
    """图2: (a) LO 累计净值 vs 成员等权 (b) 逐年夏普 + (c) 月末 OOF 五分位。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    lo_nav = (1 + m_lo["port_ret"].fillna(0)).cumprod()
    mkt_nav = (1 + m_mkt["port_ret"].fillna(0)).cumprod()
    ax.plot(lo_nav.index, lo_nav.values, label="LO 月调仓", linewidth=1.2)
    ax.plot(mkt_nav.index, mkt_nav.values, label="成员等权", linewidth=1.0, linestyle="--")
    ax.set_title("LO 月调仓 vs 成员等权累计净值")
    ax.set_ylabel("净值"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    years = yr["year"].values
    ax.bar(years, yr["sharpe"].values, color="#4c72b0")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("LO 月调仓逐年夏普")
    ax.set_xlabel("year"); ax.set_ylabel("sharpe"); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    qc = metrics["quintile_cum"]
    if qc:
        qs = [qc[i] for i in range(5)]
        ax.bar(range(5), qs, color=["#c44e52", "#dd8452", "#4c72b0", "#4c72b0", "#2ca02c"])
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"Q{i}" for i in range(5)])
        ax.set_title("月末 OOF 五分位累计收益（未扣成本）")
        ax.set_ylabel("累计收益"); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = FIGURES_DIR / "02_backtest_monthly.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default=None,
                        help="可选: purify/matrix/train/backtest/wf，缺省跑全链")
    args = parser.parse_args()
    stage = args.stage

    print("=" * 72)
    print("方向2：PIT 基本面因子 × 中证500 — 全链报告")
    print("=" * 72)

    print("\n[0] 加载 PIT 面板...")
    close, volume, member = load_pit_panel(INDEX)
    print(f"  close {close.shape}")

    final_pool, factor_tensor = None, None
    if stage in (None, "purify"):
        final_pool, factor_tensor = stage_purify(close, volume, member)
        section("提纯完成")
    elif (THIS_DIR / "purify_results_monthly.csv").exists():
        final_pool = load_pool_from_purify()

    if stage in (None, "matrix"):
        if final_pool is None:
            final_pool = load_pool_from_purify()
        stage_matrix(close, volume, member, final_pool)

    if stage in (None, "train"):
        pred_matrix, metrics, fi = stage_train(close)

    if stage in (None, "backtest"):
        if "pred_matrix" not in locals():
            from train_cv import load_monthly_matrix, train_cv
            X, y = load_monthly_matrix()
            pred_matrix, metrics, fi, _, _ = train_cv(X, close)
        import backtest_monthly as bm
        lo, m_lo, yr = stage_backtest(close, member, pred_matrix)
        # 基准
        daily_ret = close.shift(-2) / close.shift(-1) - 1
        mask_al = member.reindex_like(daily_ret).fillna(False).astype(bool)
        market_ret = daily_ret.where(mask_al).mean(axis=1).dropna()
        m_mkt = performance_metrics(market_ret)
        # 等权基线
        pur = pd.read_csv(THIS_DIR / "purify_results_monthly.csv")
        passed = pur[pur["pass"]]
        if passed.empty:
            m_ew = None
        else:
            pool = passed.sort_values("IC_IR", key=abs, ascending=False)["factor"].head(10).tolist()
            from signals.fundamental.factors import compute_factor_tensor
            tensor = compute_factor_tensor(close, pool)
            tensor = {f: df.where(member) for f, df in tensor.items()}
            ew_m = bm.equal_weight_baseline(tensor, member, close)
            lo_ew = bm.build_portfolio(bm.ffill_to_daily(ew_m, close), close,
                                       long_only=True, cost=COST_BPS, hold_days=1)
            m_ew = performance_metrics(lo_ew["port_ret"])
        _, p_lo = bm.block_bootstrap_sharpe(lo["port_ret"])
        _, p_ew = bm.block_bootstrap_sharpe(lo_ew["port_ret"]) if m_ew else (None, np.nan)
        plot_summary({"port_ret": lo["port_ret"]}, yr, metrics,
                     m_ew or {"port_ret": lo["port_ret"].iloc[:0]},
                     {"port_ret": market_ret}, p_lo, p_ew)
        section("回测完成")
        print(f"  LO: SR {m_lo['sharpe']:+.3f} 年化 {m_lo['annual']:+.2%} bootstrap p={p_lo:.4f}")
        print(f"  成员等权: SR {m_mkt['sharpe']:+.3f} 年化 {m_mkt['annual']:+.2%}")
        if m_ew:
            print(f"  等权基线 LO: SR {m_ew['sharpe']:+.3f} 年化 {m_ew['annual']:+.2%}")

    if stage in (None, "wf"):
        section("Walk-Forward PIT Select（年度重选池）")
        import walk_forward_pit_select as wfp
        wfp.main()


if __name__ == "__main__":
    main()
