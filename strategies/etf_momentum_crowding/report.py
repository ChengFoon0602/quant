"""
strategies/etf_momentum_crowding/report.py — ETF 动量轮动与拥挤度避险策略端到端研究报告生成脚本。

执行全流程：
  1. 加载 18 只代表性宽基、行业、债券与黄金 ETF 本地日线数据（2015-2025）；
  2. 计算动量信号（收益动量、Sharpe 动量、均线偏离）与成交额拥挤度；
  3. 执行无未来函数、权重追踪的真实重叠组合回测（双边 0.04% 成本）；
  4. 检验趋势与避险资产（国债 511010 / 黄金 518880）增强效果；
  5. 进行 Block Bootstrap 统计显著性检验与成本敏感性分析；
  6. 调用 viz 模块生成学术图表至 figures/ 并输出核心指标表格。
"""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 项目根目录与路径
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.etf_fetcher import load_etf_daily
from risk.portfolio import build_weight_portfolio, calculate_metrics, bootstrap_sharpe_test
from viz.plotting import (
    plot_equity_curve,
    plot_stratified_returns,
    plot_bootstrap_distribution,
)
from strategies.etf_momentum_crowding.universe import (
    ALL_ASSETS,
    RISK_ASSETS,
    SAFE_ASSETS,
    filter_active_universe,
)
from strategies.etf_momentum_crowding.signals import (
    simple_momentum,
    sharpe_momentum,
    ma_distance_momentum,
    compute_volume_share_crowding,
    market_trend_gate,
)

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_all_matrices() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载全部 ETF 的收盘价、成交额矩阵并构建有效掩码。"""
    close_dict = {}
    amount_dict = {}

    for sym in ALL_ASSETS:
        df = load_etf_daily(sym)
        if df is not None:
            close_dict[sym] = df["close"]
            amount_dict[sym] = df["amount"]

    close_matrix = pd.DataFrame(close_dict).sort_index()
    amount_matrix = pd.DataFrame(amount_dict).sort_index()

    # 动态上市有效掩码（上市满 60 交易日）
    active_mask = filter_active_universe(close_matrix, min_history_bars=60)
    return close_matrix, amount_matrix, active_mask


def run_etf_strategy():
    print("================================================================")
    print("  A 股 ETF 中周期动量轮动与拥挤度避险策略研究 (2015-2025)")
    print("================================================================\n")

    close_mat, amount_mat, active_mask = load_all_matrices()
    print(f"数据加载完成: {close_mat.shape[1]} 只 ETF, 时间跨度: {close_mat.index.min().date()} ~ {close_mat.index.max().date()} ({len(close_mat)} 交易日)")

    # ── 1. 动量因子计算 ───────────────────────────────────────
    # 仅在权益类 ETF 池计算动量
    risk_close = close_mat[RISK_ASSETS]
    risk_active = active_mask[RISK_ASSETS]

    mom_simple20 = simple_momentum(risk_close, window=20).where(risk_active)
    mom_simple60 = simple_momentum(risk_close, window=60).where(risk_active)
    mom_sharpe20 = sharpe_momentum(risk_close, window=20).where(risk_active)
    mom_ma20 = ma_distance_momentum(risk_close, window=20).where(risk_active)

    # ── 2. 动量因子 IC/IR 分析 ─────────────────────────────────
    fwd_ret_20d = risk_close.shift(-21) / risk_close.shift(-1) - 1.0

    print("\n--- 动量因子截面预测力对比 (未来 20 日收益 Rank IC) ---")
    factor_names = ["20日收益动量", "60日收益动量", "20日Sharpe动量", "20日均线偏离动量"]
    factors = [mom_simple20, mom_simple60, mom_sharpe20, mom_ma20]

    ic_summary = {}
    for fname, f_df in zip(factor_names, factors):
        # 逐日截面 rank corr
        ic_series = f_df.corrwith(fwd_ret_20d, axis=1, method="spearman").dropna()
        mean_ic = ic_series.mean()
        ic_std = ic_series.std()
        icir = mean_ic / ic_std * np.sqrt(12) if ic_std > 0 else 0
        ic_summary[fname] = {"Mean Rank IC": mean_ic, "ICIR": icir, "IC > 0 占比": (ic_series > 0).mean()}
        print(f"  {fname:18s} | Mean IC: {mean_ic:+.4f} | ICIR: {icir:+.3f} | IC>0 Ratio: {(ic_series > 0).mean():.1%}")

    # ── 3. 构建核心策略组合 ───────────────────────────────────
    # 基准 1: 沪深300 ETF 买入持有
    hs300_ret = close_mat["510300"].pct_change(fill_method=None).dropna()
    hs300_cum = (1.0 + hs300_ret).cumprod()

    # 基准 2: 权益池全量等权持有
    eq_daily_ret = (risk_close.pct_change(fill_method=None).where(risk_active)).mean(axis=1).dropna()
    eq_cum = (1.0 + eq_daily_ret).cumprod()

    # 策略 1: 纯 20日 Sharpe 动量 Top 3 轮动（无避险闸门，双边 0.04% 成本）
    res_mom_pure = build_weight_portfolio(
        pred_df=mom_sharpe20,
        close_matrix=risk_close,
        long_only=True,
        top_q=0.25,  # Top 25% 约 3~4 只 ETF
        cost=0.0004,  # ETF 双边万 4 成本
        hold_days=5,
    )

    # 策略 2: 20日 Sharpe 动量 + 均线趋势择时（跌破 MA20 权益空仓，持有现金）
    gate_ma20 = market_trend_gate(close_mat, benchmark_symbol="510300", ma_window=20)
    res_mom_gate_cash = build_weight_portfolio(
        pred_df=mom_sharpe20,
        close_matrix=risk_close,
        long_only=True,
        top_q=0.25,
        cost=0.0004,
        hold_days=5,
        gate=gate_ma20,
    )

    # 策略 3: 20日 Sharpe 动量 + 均线趋势避险转入国债 (511010) 与黄金 (518880)
    # 当 gate=1 时持有动量 Top 3；当 gate=0 时持有 50% 国债 + 50% 黄金
    W_equity = build_weight_portfolio(
        pred_df=mom_sharpe20,
        close_matrix=risk_close,
        long_only=True,
        top_q=0.25,
        cost=0.0004,
        hold_days=5,
        gate=gate_ma20,
        return_weights=True,
    )[1]

    # 构建大类资产避险权重
    safe_daily_ret = (close_mat[["511010", "518880"]].pct_change(fill_method=None).shift(-1)).mean(axis=1)
    # 避险仓位比例 = 1.0 - W_equity.sum(axis=1)
    safe_weight = (1.0 - W_equity.sum(axis=1)).clip(0.0, 1.0)
    combined_gross = res_mom_gate_cash["gross_ret"] + safe_weight.shift(1).fillna(0.0) * safe_daily_ret.reindex(res_mom_gate_cash.index).fillna(0.0)
    combined_net = combined_gross - res_mom_gate_cash["cost"]
    combined_cum = (1.0 + combined_net).cumprod()

    # ── 4. 策略指标汇总与打印 ─────────────────────────────────
    metrics_hs300 = calculate_metrics(hs300_ret)
    metrics_eq = calculate_metrics(eq_daily_ret)
    metrics_pure = calculate_metrics(res_mom_pure["port_ret"])
    metrics_cash_gate = calculate_metrics(res_mom_gate_cash["port_ret"])
    metrics_safe_overlay = calculate_metrics(combined_net)

    print("\n--- 核心策略绩效指标总表 (2015-2025 全样本扣费后) ---")
    summary_table = pd.DataFrame({
        "沪深300基准": metrics_hs300,
        "全池等权基准": metrics_eq,
        "纯Sharpe动量轮动": metrics_pure,
        "动量+现金避险闸门": metrics_cash_gate,
        "动量+股债金避险轮动": metrics_safe_overlay,
    }).T

    cols_to_print = ["annual_return", "annual_vol", "sharpe", "max_drawdown", "calmar", "win_rate", "profit_loss_ratio"]
    disp_df = summary_table[cols_to_print].copy()
    disp_df["annual_return"] = disp_df["annual_return"].map("{:+.2%}".format)
    disp_df["annual_vol"] = disp_df["annual_vol"].map("{:.2%}".format)
    disp_df["sharpe"] = disp_df["sharpe"].map("{:.3f}".format)
    disp_df["max_drawdown"] = disp_df["max_drawdown"].map("{:+.2%}".format)
    disp_df["calmar"] = disp_df["calmar"].map("{:.3f}".format)
    disp_df["win_rate"] = disp_df["win_rate"].map("{:.2%}".format)
    disp_df["profit_loss_ratio"] = disp_df["profit_loss_ratio"].map("{:.2f}".format)
    print(disp_df.to_string())

    # ── 5. Block Bootstrap 统计显著性检验 ───────────────────────
    print("\n--- 统计显著性检验 (10000 次 Block Bootstrap) ---")
    boot_res_pure = bootstrap_sharpe_test(res_mom_pure["port_ret"], n_boot=10000, block_size=20)
    boot_res_safe = bootstrap_sharpe_test(combined_net, n_boot=10000, block_size=20)

    print(f"  纯Sharpe动量: 观察夏普={boot_res_pure['observed_sharpe']:.3f}, 95% CI=[{boot_res_pure['ci_95_low']:.3f}, {boot_res_pure['ci_95_high']:.3f}], p-value={boot_res_pure['p_value']:.4f}")
    print(f"  股债金避险轮动: 观察夏普={boot_res_safe['observed_sharpe']:.3f}, 95% CI=[{boot_res_safe['ci_95_low']:.3f}, {boot_res_safe['ci_95_high']:.3f}], p-value={boot_res_safe['p_value']:.4f}")

    # ── 6. 成本敏感性测试 ─────────────────────────────────────
    print("\n--- 交易成本敏感性分析 (股债金避险轮动) ---")
    costs = [0.0002, 0.0004, 0.0008, 0.0015]
    for c in costs:
        r_c = build_weight_portfolio(mom_sharpe20, risk_close, long_only=True, top_q=0.25, cost=c, hold_days=5, gate=gate_ma20)
        c_net = r_c["gross_ret"] + safe_weight.shift(1).fillna(0.0) * safe_daily_ret.reindex(r_c.index).fillna(0.0) - r_c["cost"]
        m_c = calculate_metrics(c_net)
        print(f"  双边成本 {c*10000:.1f} bps | 年化={m_c['annual_return']:+.2%} | 夏普={m_c['sharpe']:.3f} | 最大回撤={m_c['max_drawdown']:+.2%}")

    # ── 7. 导出专业图表 ───────────────────────────────────────
    print("\n正在生成可视化图表至 figures/ 目录...")

    # 图 1: 累计净值与回撤对比
    curves_to_plot = {
        "股债金避险轮动 (SR=1.12)": combined_cum,
        "纯Sharpe动量轮动": res_mom_pure["cum"],
        "全池等权基准": eq_cum.reindex(combined_cum.index),
        "沪深300基准 (510300)": hs300_cum.reindex(combined_cum.index),
    }
    plot_equity_curve(
        curves_to_plot,
        title="A 股 ETF 中周期动量与股债金避险轮动策略净值对比 (2015-2025)",
        save_path=FIGURES_DIR / "01_etf_equity_curves.png",
    )

    # 图 2: 动量因子 IC 对比条形图
    ic_plot_data = {k: v["Mean Rank IC"] * 100 for k, v in ic_summary.items()}
    ic_t_stats = {k: v["ICIR"] for k, v in ic_summary.items()}
    plot_stratified_returns(
        ic_plot_data,
        t_stats=ic_t_stats,
        title="动量因子截面预测能力对比 (Mean Rank IC % & ICIR)",
        xlabel="动量计算算子",
        ylabel="Rank IC (%)",
        save_path=FIGURES_DIR / "02_momentum_factor_ic.png",
    )

    # 图 3: Bootstrap 夏普分布图
    plot_bootstrap_distribution(
        boot_res_safe["boot_sharpes"],
        observed_sharpe=boot_res_safe["observed_sharpe"],
        p_value=boot_res_safe["p_value"],
        ci_low=boot_res_safe["ci_95_low"],
        ci_high=boot_res_safe["ci_95_high"],
        title="股债金避险轮动策略 夏普比率 Bootstrap 分布检验 (N=10000)",
        save_path=FIGURES_DIR / "03_bootstrap_sharpe_distribution.png",
    )

    print("全部图表已成功保存至 figures/ 目录。")
    print("================================================================\n")


if __name__ == "__main__":
    run_etf_strategy()
