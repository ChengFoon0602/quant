"""
backtest_monthly.py — 月调仓组合回测（方向2 落地形态）。

月调仓语义（已论证）：
  季频因子在月末生成预测 → 月末持有到下一个月末 → 换手单次计提。
  build_portfolio 实现方式：pred 月末值【前向填充到日频】+ hold_days=1。
  （裸"月末有值 + hold=1"是错的：W_lag=shift(1) 会让持仓只在月末后 1 天存在，
  只吃到 1 天收益，与 21 日标签错配。前向填充后权重全月恒定、吃满整月。）

对照矩阵：
  ① LO-raw（ML 月调仓）vs LO-ew（等权基线，CLAUDE.md 铁律 4 必须先跑）
  ② cross_section.run_cross_section(rebalance="monthly") 交叉验证（差异 >10% 要查原因）
  ③ MA200 regime gate 对照（edge 的 regime 依赖性诚实呈现）
  ④ 逐年分解 + block bootstrap 显著性

用法:
    python strategies/zz500_fundamental_trial/backtest_monthly.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from config import (
    THIS_DIR, FEATURE_SEL_DIR, PROJECT_ROOT, FIGURES_DIR,
    INDEX, COST_BPS, TOP_Q, BOTTOM_Q, FWD_DAYS, DATE_START, DATE_END,
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FEATURE_SEL_DIR))

from build_pit_matrix import load_pit_panel
from models.portfolio_backtest import (
    build_portfolio, performance_metrics, block_bootstrap_sharpe,
)
from data.zz500_index import load_zz500_index

COST_BS = 0.00026   # cross_section 引擎口径（买入单边）
COST_SS = 0.00076   # 卖出单边


def section(title: str):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def load_pred_month_end():
    p = THIS_DIR / "oof_predictions_monthly.csv"
    if not p.exists():
        raise SystemExit("缺 oof_predictions_monthly.csv，先跑 train_cv.py")
    return pd.read_csv(p, index_col=0, parse_dates=True)


def ffill_to_daily(pred_month_end: pd.DataFrame, close_matrix: pd.DataFrame) -> pd.DataFrame:
    """月末预测 → 日频前向填充（月调仓语义核心，见模块 docstring）。"""
    td = close_matrix.index.intersection(pred_month_end.index)
    pred = pred_month_end.reindex(close_matrix.index).ffill()
    return pred.reindex(columns=close_matrix.columns)


def equal_weight_baseline(factor_tensor: dict[str, pd.DataFrame],
                          member_daily: pd.DataFrame,
                          close_matrix: pd.DataFrame) -> pd.DataFrame:
    """等权基线（CLAUDE.md 铁律 4）：最终池因子截面 zscore 等权均值 → 月末信号。"""
    frames = []
    for f, df in factor_tensor.items():
        z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
        frames.append(z)
    composite = pd.concat(frames).groupby(level=0).mean()
    composite = composite.where(member_daily)
    # 取月末值
    from purify import month_end_dates
    med = month_end_dates(composite.index.intersection(close_matrix.index))
    return composite.loc[med]


def yearly_metrics(ret: pd.Series) -> pd.DataFrame:
    rows = []
    for y, r in ret.dropna().groupby(ret.dropna().index.year):
        m = performance_metrics(r)
        rows.append({"year": y, "annual": m["annual"], "sharpe": m["sharpe"], "mdd": m["mdd"]})
    return pd.DataFrame(rows)


def run_gate(close_matrix, pred_daily):
    """MA200 regime gate 对照（复用 bear_short.build_gate V1，sh.000905）。"""
    try:
        from strategies.zz500_pit_trial.bear_short import build_gate
    except Exception:
        return None, None
    idx = load_zz500_index()
    if idx is None:
        return None, None
    idx_close = idx["close"].reindex(close_matrix.index).ffill()
    gate = build_gate(idx_close, variant="V1", confirm_days=1)
    pf_gate = build_portfolio(pred_daily, close_matrix, long_only=True,
                              cost=COST_BPS, hold_days=1, gate=gate)
    return pf_gate, gate


def main():
    print("=" * 72)
    print("方向2：月调仓组合回测")
    print("=" * 72)

    # 1. 加载
    print("\n[1] 加载数据...")
    close, volume, member = load_pit_panel(INDEX)
    pred_m = load_pred_month_end()
    td = close.index.intersection(pred_m.index)
    pred_daily = ffill_to_daily(pred_m, close)
    print(f"  预测月末 {pred_m.shape} | 日频前向填充 {pred_daily.shape}")

    # 2. LO / LS（主口径：hold=1 + 前向填充）
    print("\n[2] 月调仓组合（hold=1 + 前向填充, 双边 0.3%）...")
    lo = build_portfolio(pred_daily, close, long_only=True, cost=COST_BPS, hold_days=1)
    ls = build_portfolio(pred_daily, close, long_only=False, cost=COST_BPS, hold_days=1)
    m_lo, m_ls = performance_metrics(lo["port_ret"]), performance_metrics(ls["port_ret"])
    print(f"  LO: 年化 {m_lo['annual']:+.2%} 夏普 {m_lo['sharpe']:+.3f} 回撤 {m_lo['mdd']:+.2%}")
    print(f"  LS: 年化 {m_ls['annual']:+.2%} 夏普 {m_ls['sharpe']:+.3f} 回撤 {m_ls['mdd']:+.2%}")

    # 市场基准（成员等权）
    daily_ret = close.shift(-2) / close.shift(-1) - 1
    mask_al = member.reindex_like(daily_ret).fillna(False).astype(bool)
    market_ret = daily_ret.where(mask_al).mean(axis=1).dropna()
    m_mkt = performance_metrics(market_ret)
    print(f"  成员等权: 年化 {m_mkt['annual']:+.2%} 夏普 {m_mkt['sharpe']:+.3f}")

    # 指数基准
    idx = load_zz500_index()
    idx_ret = None
    if idx is not None:
        idx_close = idx["close"].reindex(close.index).ffill()
        idx_ret = idx_close.pct_change().fillna(0)
        m_idx = performance_metrics(idx_ret)
        print(f"  zz500 指数: 年化 {m_idx['annual']:+.2%} 夏普 {m_idx['sharpe']:+.3f}")

    # 3. 等权基线（铁律 4）
    print("\n[3] 等权基线（最终池截面 zscore 等权）...")
    pur = pd.read_csv(THIS_DIR / "purify_results_monthly.csv")
    passed = pur[pur["pass"]]
    if passed.empty:
        raise SystemExit("无通过因子，无法建等权基线")
    final_pool = passed.sort_values("IC_IR", key=abs, ascending=False)["factor"].head(10).tolist()
    from signals.fundamental.factors import compute_factor_tensor
    tensor = compute_factor_tensor(close, final_pool)
    tensor = {f: df.where(member) for f, df in tensor.items()}
    ew_m = equal_weight_baseline(tensor, member, close)
    ew_daily = ffill_to_daily(ew_m, close)
    lo_ew = build_portfolio(ew_daily, close, long_only=True, cost=COST_BPS, hold_days=1)
    m_ew = performance_metrics(lo_ew["port_ret"])
    print(f"  等权 LO: 年化 {m_ew['annual']:+.2%} 夏普 {m_ew['sharpe']:+.3f}")

    # 4. cross_section monthly 交叉验证
    print("\n[4] cross_section(rebalance=monthly) 交叉验证...")
    try:
        from backtest.cross_section import run_cross_section
        cs = run_cross_section(close, pred_m, top_pct=TOP_Q, bottom_pct=None,
                               rebalance="monthly", buy_cost=COST_BS, sell_cost=COST_SS,
                               universe=member)
        cs_lo = cs["long_equity"].pct_change().fillna(0)
        cs_equity = cs["equity"].pct_change().fillna(0)
        # cross_section 引擎的收益从调仓次日开始，与 build_portfolio 差 1 日标注
        # 对齐到 build_portfolio 的日期窗口再比较月累计
        m_cs = performance_metrics(cs_lo)
        print(f"  cross_section LO: 年化 {m_cs['annual']:+.2%} 夏普 {m_cs['sharpe']:+.3f}")
        diff = abs(m_lo["annual"] - m_cs["annual"]) / abs(m_cs["annual"]) if m_cs["annual"] else np.nan
        print(f"  年化差异: {diff:.1%}（>10% 需查原因）")
    except Exception as e:
        print(f"  cross_section 对照失败: {e}")

    # 5. MA200 regime gate 对照
    print("\n[5] MA200 regime gate 对照（edge 的 regime 依赖性）...")
    pf_gate, gate = run_gate(close, pred_daily)
    if pf_gate is not None:
        m_gate = performance_metrics(pf_gate["port_ret"])
        print(f"  LO+gate: 年化 {m_gate['annual']:+.2%} 夏普 {m_gate['sharpe']:+.3f} "
              f"vs LO-raw {m_lo['sharpe']:+.3f}")

    # 6. Bootstrap + 逐年分解
    print("\n[6] 显著性 + 逐年...")
    _, p_lo = block_bootstrap_sharpe(lo["port_ret"])
    _, p_ew = block_bootstrap_sharpe(lo_ew["port_ret"])
    print(f"  LO bootstrap SR>0 p={p_lo:.4f} | 等权基线 p={p_ew:.4f}")
    yr = yearly_metrics(lo["port_ret"])
    for _, r in yr.iterrows():
        print(f"    {int(r['year'])}: 年化 {r['annual']:+.2%} 夏普 {r['sharpe']:+.3f}")
    yr.to_csv(THIS_DIR / "backtest_monthly_yearly.csv", index=False)

    # 7. 保存摘要
    summary = {
        "lo_annual": m_lo["annual"], "lo_sharpe": m_lo["sharpe"], "lo_mdd": m_lo["mdd"],
        "ls_annual": m_ls["annual"], "ls_sharpe": m_ls["sharpe"],
        "ew_annual": m_ew["annual"], "ew_sharpe": m_ew["sharpe"],
        "market_annual": m_mkt["annual"], "market_sharpe": m_mkt["sharpe"],
        "boot_p_lo": p_lo, "boot_p_ew": p_ew,
    }
    if idx_ret is not None:
        summary["idx_annual"] = m_idx["annual"]; summary["idx_sharpe"] = m_idx["sharpe"]
    if pf_gate is not None:
        summary["gate_annual"] = m_gate["annual"]; summary["gate_sharpe"] = m_gate["sharpe"]
    pd.Series(summary).to_csv(THIS_DIR / "backtest_monthly_summary.csv")
    print("\n完成。")


if __name__ == "__main__":
    main()
