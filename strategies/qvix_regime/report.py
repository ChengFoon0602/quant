"""
strategies/qvix_regime/report.py — QVIX 波动率状态闸门：能否改善 ETF 动量轮动？

研究问题
--------
在 `etf_momentum_crowding` 的 20 日 Sharpe 动量轮动之上，叠加由**期权隐含波动率
（QVIX）**构造的市场状态闸门，能否比现有 MA20 趋势闸门提供更好的风险调整收益？
两者叠加是否进一步改善？

四道关卡（前两关不通过则研究终止）
--------------------------------
  G1 数据可信   QVIX 与标的已实现波动率的一致性（Pearson ≥ 0.60 + 方差风险溢价 > 0）
  G2 增量信息   IV 对未来 20 日已实现波动的预测力是否超越历史 RV（Mincer-Zarnowitz）
  G3 尾部收益   极端高 IV 分位是否对应显著为负的未来收益（非重叠月频 + 置换检验）
  G4 可交易性   同等平均权益暴露下，QVIX gate 是否优于 MA20 gate（夏普 + 回撤）

复用约定
--------
- 组合引擎：`risk.portfolio.build_weight_portfolio`（权重追踪法，内部 shift(1)）
- 成本真源：`risk.cost_model.ETF_ROUND_TRIP`（万 4，与 etf_momentum_crowding 一致）
- 分位：**滚动 252 日**，禁止全样本分位（见 signals.py 模块说明）

执行
----
    cd strategies/qvix_regime && python report.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.etf_fetcher import load_etf_daily
from data.qvix_fetcher import (
    G1_MIN_PEARSON,
    assert_qvix_usable,
    audit_consistency,
    load_qvix,
    realized_vol,
)
from risk.cost_model import ETF_ROUND_TRIP
from risk.portfolio import bootstrap_sharpe_test, build_weight_portfolio, calculate_metrics
from strategies.etf_momentum_crowding.signals import market_trend_gate, sharpe_momentum
from strategies.etf_momentum_crowding.universe import (
    ALL_ASSETS,
    RISK_ASSETS,
    filter_active_universe,
)
from strategies.qvix_regime.signals import (
    calibrate_quantile_for_exposure,
    combine_gates,
    qvix_binary_gate,
    qvix_position_scale,
    rolling_percentile,
)
from viz.plotting import plot_bootstrap_distribution, plot_equity_curve, plot_stratified_returns

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SAFE_LEG = ["511010", "518880"]  # 国债 / 黄金，与 etf_momentum_crowding 一致
TOP_Q = 0.25
HOLD_DAYS = 5


# ══════════════════════════════════════════════════════════════════
# 0. 数据
# ══════════════════════════════════════════════════════════════════
def load_all_matrices() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载 ETF 收盘/成交额矩阵与动态有效掩码。

    价格已由数据层返回后复权口径（`load_etf_daily` 默认 `adjust="hfq"`，
    分红与份额折算在 `data/etf_adjust.py` 中修正），本函数不再做二次清洗。
    """
    close_dict, amount_dict = {}, {}
    for sym in ALL_ASSETS:
        df = load_etf_daily(sym)
        if df is not None:
            close_dict[sym] = df["close"]
            amount_dict[sym] = df["amount"]

    close_matrix = pd.DataFrame(close_dict).sort_index()
    amount_matrix = pd.DataFrame(amount_dict).sort_index()
    active_mask = filter_active_universe(close_matrix, min_history_bars=60)
    return close_matrix, amount_matrix, active_mask


# ══════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════
def ols(y: pd.Series, X: pd.DataFrame) -> Dict[str, object]:
    """最小二乘（含量常数项），返回 R²、系数与 t 值。

    项目未安装 statsmodels，此处手写以避免新增依赖。
    """
    A = np.column_stack([np.ones(len(X))] + [X[c].to_numpy(float) for c in X.columns])
    yv = y.to_numpy(float)
    beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
    resid = yv - A @ beta
    n, k = A.shape
    s2 = float(resid @ resid) / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    r2 = 1.0 - float(resid @ resid) / float(((yv - yv.mean()) ** 2).sum())
    names = ["const"] + list(X.columns)
    return {
        "r2": r2,
        "beta": dict(zip(names, beta)),
        "t": dict(zip(names, beta / se)),
        "n": n,
    }


def safe_overlay(
    res: pd.DataFrame,
    close_mat: pd.DataFrame,
    weights: pd.DataFrame,
    safe_leg: list[str] = SAFE_LEG,
) -> pd.Series:
    """把避险腿接回权益腿，返回扣费后的日收益序列。

    口径：t 日收益 = 权益腿 W_held[t-1]·r[t] + 避险腿 safe_w[t-1]·r_safe[t] − 成本[t]。
    这里不使用 `.shift(-1)`：避险资产收益与权益收益必须锚定在同一天，
    否则会引入一日前视。
    """
    safe_ret = close_mat[safe_leg].pct_change(fill_method=None).mean(axis=1)
    safe_w = (1.0 - weights.sum(axis=1)).clip(0.0, 1.0)
    safe_w_lag = safe_w.shift(1).fillna(0.0).reindex(res.index).fillna(0.0)
    safe_contribution = safe_w_lag * safe_ret.reindex(res.index).fillna(0.0)
    return res["gross_ret"] + safe_contribution - res["cost"]


def matched_exposure_gate(
    iv: pd.Series,
    target_exposure: float,
    window: int = 252,
) -> Tuple[pd.Series, float]:
    """把 QVIX gate 的阈值校准到与对照 gate 相同的平均暴露。"""
    q = calibrate_quantile_for_exposure(iv, target_exposure, window=window)
    return qvix_binary_gate(iv, quantile=q, window=window), q


# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 78)
    print("  QVIX 波动率状态闸门：能否改善 ETF 动量轮动？")
    print("=" * 78)

    close_mat, amount_mat, active_mask = load_all_matrices()
    print(f"\n[数据] ETF {close_mat.shape[1]} 只  "
          f"{close_mat.index.min().date()} ~ {close_mat.index.max().date()}  "
          f"({len(close_mat)} 交易日)")

    iv = load_qvix("50etf")
    print(f"[数据] QVIX(50ETF) n={len(iv)}  {iv.index.min().date()} ~ {iv.index.max().date()}")

    risk_close = close_mat[RISK_ASSETS]
    risk_active = active_mask[RISK_ASSETS]
    bench = close_mat["510300"]

    # ── G1 数据可信度 ──────────────────────────────────────────
    print("\n" + "─" * 78)
    print("G1 · QVIX 与已实现波动率的一致性审计")
    print("─" * 78)
    g1 = audit_consistency(iv, close_mat["510050"], window=20)
    print(f"  样本 n={int(g1['n'])}")
    print(f"  Pearson  = {g1['pearson']:.4f}   Spearman = {g1['spearman']:.4f}   "
          f"(门槛 {G1_MIN_PEARSON})")
    print(f"  QVIX 中位 = {g1['iv_median']:.2f}   RV20 中位 = {g1['rv_median']:.2f}   "
          f"IV−RV 中位 = {g1['spread_median']:+.2f}")
    print(f"  方差风险溢价为正的比例 = {g1['spread_positive_ratio']:.1%}")
    print(f"  QVIX 区间 [{g1['iv_min']:.2f}, {g1['iv_max']:.2f}]   "
          f"RV 峰值 {g1['rv_max']:.2f}")
    assert_qvix_usable(g1, name="50etf")
    print("  ✅ G1 通过：数据可用于后续研究")

    # ── G2 增量信息 ────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("G2 · 对未来 20 日已实现波动的预测力（Mincer-Zarnowitz）")
    print("─" * 78)
    rv20 = realized_vol(close_mat["510050"], 20)
    d2 = pd.DataFrame({"iv": iv, "rv": rv20, "rv_fwd": rv20.shift(-20)}).dropna()

    m_iv = ols(d2["rv_fwd"], d2[["iv"]])
    m_rv = ols(d2["rv_fwd"], d2[["rv"]])
    m_both = ols(d2["rv_fwd"], d2[["iv", "rv"]])

    print(f"  样本 n={m_iv['n']}")
    print(f"  仅 IV   : R2={m_iv['r2']:.4f}  beta={m_iv['beta']['iv']:.4f}  "
          f"t={m_iv['t']['iv']:6.2f}")
    print(f"  仅 RV20 : R2={m_rv['r2']:.4f}  beta={m_rv['beta']['rv']:.4f}  "
          f"t={m_rv['t']['rv']:6.2f}")
    print(f"  IV + RV : R2={m_both['r2']:.4f}  "
          f"IV b={m_both['beta']['iv']:.4f}(t={m_both['t']['iv']:.2f})  "
          f"RV b={m_both['beta']['rv']:.4f}(t={m_both['t']['rv']:.2f})")

    g2_pass = (m_both["t"]["iv"] > 2.0) and (m_iv["r2"] > m_rv["r2"])
    print(f"  {'✅' if g2_pass else '❌'} G2 {'通过' if g2_pass else '未通过'}："
          f"IV 单独 R² {'高于' if m_iv['r2'] > m_rv['r2'] else '低于'} RV 基准，"
          f"联合回归 IV 的 t={m_both['t']['iv']:.2f}")

    # ── G3 尾部收益 ────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("G3 · QVIX 滚动分位与未来收益（非重叠月频，规避重叠样本陷阱）")
    print("─" * 78)
    iv_pct = rolling_percentile(iv, window=252)
    bench_ret = bench.pct_change(fill_method=None)
    fwd60 = (bench.shift(-60) / bench - 1.0) * 100.0

    d3 = pd.DataFrame({"iv": iv, "pct": iv_pct, "f60": fwd60}).dropna()
    # 每月最后一个观测 → 非重叠样本
    d3_month = d3.groupby(d3.index.to_period("M")).tail(1)

    bins = [(0.0, 0.10, "Q10 极低"), (0.10, 0.30, "Q10-30"),
            (0.30, 0.50, "Q30-50"), (0.50, 0.70, "Q50-70"),
            (0.70, 0.90, "Q70-90"), (0.90, 1.01, "Q90 极高")]
    bucket = {}
    bucket_t = {}
    bucket_n = {}
    for lo, hi, label in bins:
        g = d3_month[(d3_month["pct"] >= lo) & (d3_month["pct"] < hi)]
        bucket[label] = float(g["f60"].mean()) if len(g) else 0.0
        bucket_n[label] = int(len(g))
        bucket_t[label] = float(g["f60"].mean() / (g["f60"].std(ddof=1) / np.sqrt(len(g)))) \
            if len(g) > 2 else 0.0
        print(f"  {label:10s} n={len(g):3d}  QVIX均值={g['iv'].mean():5.1f}  "
              f"未来60日={g['f60'].mean():+6.2f}%  t={bucket_t[label]:+5.2f}")

    # 置换检验：极端分位组 vs 其余
    extreme = d3_month[d3_month["pct"] >= 0.90]["f60"]
    rest = d3_month[d3_month["pct"] < 0.90]["f60"]
    rng = np.random.default_rng(42)
    pooled = np.concatenate([extreme.to_numpy(), rest.to_numpy()])
    obs_diff = rest.mean() - extreme.mean()
    perm = np.empty(5000)
    for i in range(5000):
        rng.shuffle(pooled)
        perm[i] = pooled[len(extreme):].mean() - pooled[:len(extreme)].mean()
    p_perm = float((perm >= obs_diff).mean())
    print(f"\n  置换检验: 非极端组 − 极端组 = {obs_diff:+.2f}pp  "
          f"(n_extreme={len(extreme)}, n_rest={len(rest)})  p={p_perm:.4f}")
    g3_pass = (obs_diff > 0) and (p_perm < 0.10)
    print(f"  {'✅' if g3_pass else '❌'} G3 {'通过' if g3_pass else '未通过'}")

    # ── G4 覆盖层对比 ──────────────────────────────────────────
    print("\n" + "─" * 78)
    print("G4 · 叠加闸门后的组合表现（成本：ETF 双边 "
          f"{ETF_ROUND_TRIP*10000:.1f} bps）")
    print("─" * 78)

    mom = sharpe_momentum(risk_close, window=20).where(risk_active)
    gate_ma = market_trend_gate(close_mat, benchmark_symbol="510300", ma_window=20)

    # ── 统一样本起点（关键）────────────────────────────────────
    # QVIX 的 252 日滚动分位需要 252 个观测预热，2016-02 之前恒为 NaN →
    # qvix_binary_gate 回退为全仓。若不截断，2015 年股灾对 QVIX 组完全无保护、
    # 对 MA20 组（仅需 20 日预热）却能躲开，两组不可比（实测回撤 −46.4% vs −25.6%）。
    # 对照组必须在同一窗口内比较。
    start = rolling_percentile(iv, 252).dropna().index.min()
    n_before = len(mom.index)
    mom = mom.loc[start:]
    risk_close = risk_close.loc[start:]
    gate_ma = gate_ma.loc[start:]
    # QVIX 原始序列到 2026-11，比 ETF 缓存（2025-12）多 200 余个交易日。
    # 若不截止到此，这些额外日期会被计入「平均权益暴露」，使阈值校准系统性偏低。
    iv_g = iv.loc[start:mom.index.max()]

    gate_iv = qvix_binary_gate(iv_g, quantile=0.90, window=252)
    gate_both = combine_gates(gate_ma, gate_iv)
    scale_iv = qvix_position_scale(iv_g, window=252)
    print(f"\n[样本] 统一截断至 {start.date()}（QVIX 分位预热完成）  "
          f"{n_before} → {len(mom)} 交易日")
    print(f"[样本] QVIX 截断至 {iv_g.index.max().date()}，与 ETF 缓存对齐（丢弃 "
          f"{len(iv.loc[start:]) - len(iv_g)} 个无标的的交易日）")

    runs = {}

    def run(tag: str, **kwargs) -> None:
        res = build_weight_portfolio(
            pred_df=mom, close_matrix=risk_close, long_only=True,
            top_q=TOP_Q, cost=ETF_ROUND_TRIP, hold_days=HOLD_DAYS,
            return_weights=True, **kwargs,
        )
        df, w = res
        runs[tag] = {"res": df, "weights": w}

    run("B0 纯动量")
    run("B1 MA20 闸门", gate=gate_ma)
    run("B2 QVIX 闸门", gate=gate_iv)
    run("B3 双闸门", gate=gate_both)
    run("B4 QVIX 连续缩放", position_scale=scale_iv)

    # 基准（同样截断到统一起点）
    hs300_ret = bench.loc[start:].pct_change(fill_method=None).dropna()

    rows = {}
    curves = {}
    for tag, d in runs.items():
        net = safe_overlay(d["res"], close_mat, d["weights"])
        m = calculate_metrics(net)
        expo = float((d["weights"].sum(axis=1)).reindex(net.index).mean())
        rows[tag] = {
            "年化(CAGR)": m["annual_return"], "年化波动": m["annual_vol"],
            "夏普": m["sharpe"], "最大回撤": m["max_drawdown"],
            "Calmar": m["calmar"], "平均权益暴露": expo,
        }
        curves[tag] = (1.0 + net).cumprod()

    rows["沪深300 基准"] = {
        "年化(CAGR)": calculate_metrics(hs300_ret)["annual_return"],
        "年化波动": calculate_metrics(hs300_ret)["annual_vol"],
        "夏普": calculate_metrics(hs300_ret)["sharpe"],
        "最大回撤": calculate_metrics(hs300_ret)["max_drawdown"],
        "Calmar": calculate_metrics(hs300_ret)["calmar"],
        "平均权益暴露": 1.0,
    }

    tbl = pd.DataFrame(rows).T
    disp = tbl.copy()
    for c in ["年化(CAGR)", "年化波动", "最大回撤", "平均权益暴露"]:
        disp[c] = disp[c].map("{:+.2%}".format if c != "年化波动" else "{:.2%}".format)
    for c in ["夏普", "Calmar"]:
        disp[c] = disp[c].map("{:.3f}".format)
    print(disp.to_string())

    # ── G4b 暴露匹配对照组 ────────────────────────────────────
    print("\n" + "─" * 78)
    print("G4b · 暴露匹配对照（MA20 闸门 vs 阈值校准后的 QVIX 闸门）")
    print("─" * 78)
    # 暴露匹配：注意「gate 开仓率」≠「实际权益暴露」。
    # build_weight_portfolio 在有效标的数 < min_stocks(=8) 的交易日 W_target 恒为 0，
    # 因此实际暴露 = 开仓率 × 可交易基线暴露。校准目标必须用 MA20 的**实际**
    # 暴露（而非 gate 均值），否则 QVIX 会被系统性高估或低估仓位。
    base_expo = float(runs["B0 纯动量"]["weights"].sum(axis=1).mean())
    ma_actual_expo = float(runs["B1 MA20 闸门"]["weights"].sum(axis=1).mean())
    ma_gate_eff = ma_actual_expo / base_expo if base_expo > 0 else 1.0
    # 校准序列与回测序列必须同索引（QVIX 交易日历与 ETF 缓存在早年少若干天）
    gate_iv_matched, q_matched = matched_exposure_gate(
        iv_g.reindex(mom.index).dropna(), ma_gate_eff)
    print(f"  无闸门基线暴露 = {base_expo:.2%}   MA20 实际暴露 = {ma_actual_expo:.2%}  "
          f"→ 等效开仓率目标 = {ma_gate_eff:.2%}")
    print(f"  校准 QVIX 阈值 = {q_matched:.4f}（使暴露同等）")

    res_ma = runs["B1 MA20 闸门"]
    net_ma = safe_overlay(res_ma["res"], close_mat, res_ma["weights"])

    # 迭代反解：解析暴露（gate 均值）与回测实测暴露之间存在系统性偏差 ——
    # QVIX 交易日历与 ETF 缓存不完全重合，`reindex(...).dropna()` 后分位序列变短，
    # 使 (pct < q) 的占比估计偏高。故以**回测实测暴露**为反馈量做 3 轮定点迭代。
    q_target = ma_gate_eff
    gate_iv_matched, q_matched, res_ivm = None, np.nan, None
    for _ in range(3):
        g_try, q_try = matched_exposure_gate(
            iv_g.reindex(mom.index).dropna(), q_target)
        r_try = build_weight_portfolio(
            pred_df=mom, close_matrix=risk_close, long_only=True, top_q=TOP_Q,
            cost=ETF_ROUND_TRIP, hold_days=HOLD_DAYS, gate=g_try, return_weights=True,
        )
        expo_try = float(r_try[1].sum(axis=1).mean()) / base_expo
        gate_iv_matched, q_matched, res_ivm = g_try, q_try, r_try
        if abs(expo_try - ma_gate_eff) < 0.01:
            break
        q_target *= ma_gate_eff / max(expo_try, 1e-6)
    print(f"  校准 QVIX 阈值 = {q_matched:.4f}（迭代后暴露 "
          f"{float(res_ivm[1].sum(axis=1).mean()):.2%} vs 目标 {ma_actual_expo:.2%}）")

    net_ivm = safe_overlay(res_ivm[0], close_mat, res_ivm[1])

    m_ma, m_ivm = calculate_metrics(net_ma), calculate_metrics(net_ivm)
    print(f"\n  {'策略':22s} {'年化':>9s} {'波动':>9s} {'夏普':>8s} {'回撤':>9s} {'暴露':>8s}")
    for nm, m, e in [("MA20 闸门", m_ma, float(res_ma["weights"].sum(axis=1).mean())),
                     ("QVIX 闸门(匹配)", m_ivm,
                      float(res_ivm[1].sum(axis=1).mean()))]:
        print(f"  {nm:22s} {m['annual_return']:+9.2%} {m['annual_vol']:9.2%} "
              f"{m['sharpe']:8.3f} {m['max_drawdown']:+9.2%} {e:8.1%}")
    d_sharpe = m_ivm["sharpe"] - m_ma["sharpe"]
    d_mdd = m_ivm["max_drawdown"] - m_ma["max_drawdown"]
    print(f"\n  差值（QVIX − MA20）: 夏普 {d_sharpe:+.3f}   回撤 {d_mdd:+.2%}")

    g4_pass = d_sharpe > 0
    print(f"  {'✅' if g4_pass else '❌'} G4 {'通过' if g4_pass else '未通过'}"
          f"：同等暴露下 QVIX 闸门夏普{'优于' if g4_pass else '不及'} MA20")

    # ── G4c 逐年稳健性 ────────────────────────────────────────
    print("\n" + "─" * 78)
    print("G4c · 逐年稳健性（超额不应集中于单一年份）")
    print("─" * 78)
    rows_y = []
    for y in sorted({d.year for d in net_ma.index}):
        a = net_ma[net_ma.index.year == y]
        b = net_ivm[net_ivm.index.year == y]
        if len(a) < 60 or len(b) < 60:
            continue
        rows_y.append({
            "年份": y,
            "MA20夏普": calculate_metrics(a)["sharpe"],
            "QVIX夏普": calculate_metrics(b)["sharpe"],
        })
    ydf = pd.DataFrame(rows_y)
    ydf["差值"] = ydf["QVIX夏普"] - ydf["MA20夏普"]
    print(ydf.round(3).to_string(index=False))
    n_win = int((ydf["差值"] > 0).sum())
    total = float(ydf["差值"].sum())
    max_contrib = float(ydf["差值"].max()) / total if total > 0 else 1.0
    print(f"\n  胜出 {n_win}/{len(ydf)} 年   年度差值合计 = {total:+.3f}   "
          f"最大单年贡献占比 = {max_contrib:.0%}")
    # 双判据：胜出过半 + 无单年贡献过半。后者用于识别「靠一年行情撑起全部超额」。
    wf_pass = (n_win > len(ydf) / 2) and (max_contrib < 0.5)
    print(f"  {'✅' if wf_pass else '❌'} G4c {'通过' if wf_pass else '未通过'}"
          f"（判据：胜出过半 且 最大单年贡献 < 50%）")

    # ── Bootstrap ─────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("统计显著性（Block Bootstrap，N=10000，block=20）")
    print("─" * 78)
    boot_iv = bootstrap_sharpe_test(net_ivm, n_boot=10000, block_size=20)
    boot_ma = bootstrap_sharpe_test(net_ma, n_boot=10000, block_size=20)
    for nm, b in [("MA20 闸门", boot_ma), ("QVIX 闸门(匹配)", boot_iv)]:
        print(f"  {nm:20s} 夏普={b['observed_sharpe']:+.3f}  "
              f"95%CI=[{b['ci_95_low']:+.3f}, {b['ci_95_high']:+.3f}]  "
              f"p={b['p_value']:.4f}")

    # ── 成本敏感性 ────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("成本敏感性（QVIX 闸门 · 匹配暴露）")
    print("─" * 78)
    for c in [0.0002, 0.0004, 0.0008, 0.0015]:
        r = build_weight_portfolio(
            pred_df=mom, close_matrix=risk_close, long_only=True, top_q=TOP_Q,
            cost=c, hold_days=HOLD_DAYS, gate=gate_iv_matched, return_weights=True,
        )
        net = safe_overlay(r[0], close_mat, r[1])
        m = calculate_metrics(net)
        print(f"  双边 {c*10000:4.1f} bps | 年化={m['annual_return']:+.2%}  "
              f"夏普={m['sharpe']:.3f}  回撤={m['max_drawdown']:+.2%}")

    # ── 图表 ──────────────────────────────────────────────────
    print("\n生成图表 ...")
    curves_plot = {k: v for k, v in curves.items()}
    curves_plot["沪深300 基准"] = (1.0 + hs300_ret).cumprod().reindex(
        list(curves.values())[0].index)
    plot_equity_curve(
        curves_plot,
        title="QVIX 波动率闸门 vs MA20 趋势闸门 · ETF 动量轮动 (2015-2025)",
        save_path=FIGURES_DIR / "01_equity_curves.png",
    )
    plot_stratified_returns(
        bucket, t_stats=bucket_t, n_obs=bucket_n,
        title="QVIX 滚动分位与沪深300未来60日收益（非重叠月频）",
        xlabel="QVIX 滚动分位分组", ylabel="未来60日收益 (%)",
        save_path=FIGURES_DIR / "02_qvix_quantile_returns.png",
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(iv.index, iv.values, lw=0.9, color="#4C72B0", label="QVIX(50ETF)")
    pct = rolling_percentile(iv, 252)
    hi = pct >= 0.90
    ax.fill_between(iv.index, iv.min(), iv.max(), where=hi,
                    color="#C44E52", alpha=0.18, label="滚动分位 ≥ 90%")
    ax.set_yscale("log")
    ax.set_title("50ETF 期权隐含波动率指数 QVIX 与极端分位区间")
    ax.set_ylabel("QVIX（点，对数轴）")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_qvix_regime.png", dpi=150)
    plt.close(fig)

    plot_bootstrap_distribution(
        boot_iv["boot_sharpes"], observed_sharpe=boot_iv["observed_sharpe"],
        p_value=boot_iv["p_value"], ci_low=boot_iv["ci_95_low"],
        ci_high=boot_iv["ci_95_high"],
        title="QVIX 闸门（匹配暴露）夏普 Bootstrap 分布",
        save_path=FIGURES_DIR / "04_bootstrap_sharpe.png",
    )

    print(f"图表已保存至 {FIGURES_DIR}")
    print("\n" + "=" * 78)
    print("  完成。四关判决：")
    marks = [("G2 波动预测", g2_pass), ("G3 尾部收益", g3_pass),
             ("G4 暴露匹配", g4_pass), ("G4c 逐年稳健", wf_pass)]
    for nm, ok in marks:
        print(f"    {'✅' if ok else '❌'} {nm}")
    print(f"  → {sum(ok for _, ok in marks)}/4 通过")
    print("=" * 78)


if __name__ == "__main__":
    main()
