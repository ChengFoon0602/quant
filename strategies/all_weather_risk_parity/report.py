"""
strategies/all_weather_risk_parity/report.py — 全天候多资产风险平价策略端到端研究报告生成脚本。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.etf_fetcher import load_etf_daily
from risk.portfolio import calculate_metrics, bootstrap_sharpe_test
from risk.crowding import month_end_dates
from viz.plotting import (
    plot_equity_curve,
    plot_bootstrap_distribution,
)
from strategies.all_weather_risk_parity.universe import (
    ALL_WEATHER_UNIVERSE,
    CORE_ASSETS,
    filter_active_multi_assets,
)
from strategies.all_weather_risk_parity.optimizer import (
    solve_equal_risk_contribution,
    optimize_inverse_volatility,
    optimize_60_40_blend,
    determine_macro_risk_budgets,
    calculate_risk_contributions,
)

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_all_weather_matrices() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载全部全天候资产的收盘价矩阵。"""
    close_dict = {}
    for sym in CORE_ASSETS:
        df = load_etf_daily(sym)
        if df is not None:
            close_dict[sym] = df["close"]

    close_matrix = pd.DataFrame(close_dict).sort_index()
    active_mask = filter_active_multi_assets(close_matrix, min_history_bars=60)
    return close_matrix, active_mask


def run_all_weather_backtest(
    lookback_days: int = 120,
    cost_bps: float = 0.0004,
):
    print("================================================================")
    print("  全天候风险平价 (All-Weather Risk Parity) 策略实证研究 (2015-2025)")
    print("================================================================\n")

    close_matrix, active_mask = load_all_weather_matrices()
    symbols = list(close_matrix.columns)
    print(f"资产池: {[ALL_WEATHER_UNIVERSE[s]['name'] for s in symbols]}")
    print(f"数据区间: {close_matrix.index.min().date()} ~ {close_matrix.index.max().date()} ({len(close_matrix)} 交易日)\n")

    # 日收益率序列：t+1 日收盘相对 t 日收盘收益，对齐到 t 日（代表 t 日持有到 t+1 日的收益）
    daily_rets = close_matrix.pct_change(fill_method=None).shift(-1)

    # 月末调仓日期
    med = month_end_dates(close_matrix.index)
    med = [d for d in med if d in close_matrix.index and close_matrix.index.get_loc(d) >= lookback_days]

    # 初始化每日持仓权重 DataFrame
    W_eq = pd.DataFrame(0.0, index=close_matrix.index, columns=symbols)
    W_6040 = pd.DataFrame(0.0, index=close_matrix.index, columns=symbols)
    W_inv_vol = pd.DataFrame(0.0, index=close_matrix.index, columns=symbols)
    W_rp = pd.DataFrame(0.0, index=close_matrix.index, columns=symbols)
    W_macro_rp = pd.DataFrame(0.0, index=close_matrix.index, columns=symbols)

    # 记录风险贡献
    rc_rp_records = {}

    for i, d in enumerate(med):
        loc = close_matrix.index.get_loc(d)
        valid_mask = active_mask.loc[d]
        active_syms = [s for s in symbols if valid_mask.get(s, False)]
        if len(active_syms) < 2:
            continue

        hist_close = close_matrix.iloc[loc - lookback_days : loc + 1][active_syms]
        hist_rets = hist_close.pct_change(fill_method=None).dropna()
        if len(hist_rets) < 20:
            continue

        cov = hist_rets.cov().values * 252.0  # 年化协方差

        # 1. 等权 (1/N)
        w_eq = np.full(len(active_syms), 1.0 / len(active_syms))

        # 2. 传统 60/40
        w_6040 = optimize_60_40_blend(active_syms)

        # 3. 波动率倒数
        w_inv_vol = optimize_inverse_volatility(cov)

        # 4. 标准风险平价 (Equal Risk Contribution)
        w_rp = solve_equal_risk_contribution(cov)

        # 5. 宏观自适应风险预算风险平价
        b_macro = determine_macro_risk_budgets(hist_close, active_syms)
        w_macro_rp = solve_equal_risk_contribution(cov, risk_budgets=b_macro)

        # 确定本期持有区间 [d, next_d]
        next_d = med[i + 1] if i + 1 < len(med) else close_matrix.index[-1]
        hold_dates = close_matrix.loc[d:next_d].index

        # 填入当月每一天的实际持仓权重
        for sym_idx, sym in enumerate(active_syms):
            W_eq.loc[hold_dates, sym] = w_eq[sym_idx]
            W_6040.loc[hold_dates, sym] = w_6040[sym_idx]
            W_inv_vol.loc[hold_dates, sym] = w_inv_vol[sym_idx]
            W_rp.loc[hold_dates, sym] = w_rp[sym_idx]
            W_macro_rp.loc[hold_dates, sym] = w_macro_rp[sym_idx]

        # 记录 RP 的风险贡献
        rc_ratios = calculate_risk_contributions(w_rp, cov)
        rc_rp_records[d] = dict(zip(active_syms, rc_ratios))

    start_date = med[0]
    valid_dates = [d for d in close_matrix.index if d >= start_date and d in daily_rets.index]

    # 回测计算函数
    def backtest_weights(W_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        W_sub = W_df.loc[valid_dates]
        r_sub = daily_rets.loc[valid_dates]
        gross = (W_sub * r_sub).sum(axis=1)
        turnover = (W_sub - W_sub.shift(1).fillna(0.0)).abs().sum(axis=1)
        cost = turnover * (cost_bps / 2.0)
        net = (gross - cost).dropna()
        cum = (1.0 + net).cumprod()
        return net, cum

    ret_eq, cum_eq = backtest_weights(W_eq)
    ret_6040, cum_6040 = backtest_weights(W_6040)
    ret_inv_vol, cum_inv_vol = backtest_weights(W_inv_vol)
    ret_rp, cum_rp = backtest_weights(W_rp)
    ret_macro_rp, cum_macro_rp = backtest_weights(W_macro_rp)

    # 沪深300 基准
    hs300_ret = daily_rets.loc[valid_dates, "510300"].dropna()
    hs300_cum = (1.0 + hs300_ret).cumprod()

    # ── 绩效指标汇总 ───────────────────────────────────────────
    m_hs300 = calculate_metrics(hs300_ret)
    m_eq = calculate_metrics(ret_eq)
    m_6040 = calculate_metrics(ret_6040)
    m_inv_vol = calculate_metrics(ret_inv_vol)
    m_rp = calculate_metrics(ret_rp)
    m_macro_rp = calculate_metrics(ret_macro_rp)

    summary_df = pd.DataFrame({
        "沪深300基准 (510300)": m_hs300,
        "全资产等权 (1/N)": m_eq,
        "经典股债 60/40": m_6040,
        "波动率倒数加权": m_inv_vol,
        "标准风险平价 (ERC)": m_rp,
        "宏观自适应风险平价": m_macro_rp,
    }).T

    cols_print = ["annual_return", "annual_vol", "sharpe", "max_drawdown", "calmar", "win_rate", "profit_loss_ratio"]
    disp_df = summary_df[cols_print].copy()
    disp_df["annual_return"] = disp_df["annual_return"].map("{:+.2%}".format)
    disp_df["annual_vol"] = disp_df["annual_vol"].map("{:.2%}".format)
    disp_df["sharpe"] = disp_df["sharpe"].map("{:.3f}".format)
    disp_df["max_drawdown"] = disp_df["max_drawdown"].map("{:+.2%}".format)
    disp_df["calmar"] = disp_df["calmar"].map("{:.3f}".format)
    disp_df["win_rate"] = disp_df["win_rate"].map("{:.2%}".format)
    disp_df["profit_loss_ratio"] = disp_df["profit_loss_ratio"].map("{:.2f}".format)

    print("--- 策略全周期绩效指标总表 (2015-2025，扣除双边 4 bps 摩擦) ---")
    print(disp_df.to_string())
    print("\n" + "=" * 64)

    # ── Bootstrap 检验 ─────────────────────────────────────────
    boot_rp = bootstrap_sharpe_test(ret_rp, n_boot=10000, block_size=20)
    boot_macro = bootstrap_sharpe_test(ret_macro_rp, n_boot=10000, block_size=20)

    print("\n--- 统计显著性检验 (10000 次 Block Bootstrap) ---")
    print(f"  标准风险平价 (ERC): 观察夏普={boot_rp['observed_sharpe']:.3f}, 95% CI=[{boot_rp['ci_95_low']:.3f}, {boot_rp['ci_95_high']:.3f}], p-value={boot_rp['p_value']:.4f}")
    print(f"  宏观自适应风险平价: 观察夏普={boot_macro['observed_sharpe']:.3f}, 95% CI=[{boot_macro['ci_95_low']:.3f}, {boot_macro['ci_95_high']:.3f}], p-value={boot_macro['p_value']:.4f}")

    # ── 成本敏感性测试 ─────────────────────────────────────
    print("\n--- 交易成本敏感性分析 (宏观自适应风险平价) ---")
    for c in [0.0002, 0.0004, 0.0008, 0.0015]:
        r_c, _ = backtest_weights(W_macro_rp)
        m_c = calculate_metrics(r_c)
        print(f"  双边成本 {c*10000:.1f} bps | 年化={m_c['annual_return']:+.2%} | 夏普={m_c['sharpe']:.3f} | 最大回撤={m_c['max_drawdown']:+.2%}")

    # ── 生成学术图表 ───────────────────────────────────────────
    print("\n正在生成可视化图表至 figures/ 目录...")

    # 图 1: 累计净值与回撤对比
    curves = {
        f"宏观自适应风险平价 (SR={m_macro_rp['sharpe']:.2f})": cum_macro_rp,
        f"标准风险平价 ERC (SR={m_rp['sharpe']:.2f})": cum_rp,
        f"经典股债 60/40 (SR={m_6040['sharpe']:.2f})": cum_6040,
        f"全资产等权 (1/N) (SR={m_eq['sharpe']:.2f})": cum_eq,
        f"沪深300基准 (510300) (SR={m_hs300['sharpe']:.2f})": hs300_cum,
    }
    plot_equity_curve(
        curves,
        title="全天候风险平价 (Risk Parity) vs 传统资产配置策略净值对比 (2015-2025)",
        save_path=FIGURES_DIR / "01_risk_parity_equity_curves.png",
    )

    # 图 2: 资产持仓权重演变 (堆叠面积图)
    fig_w, ax_w = plt.subplots(figsize=(10, 5))
    W_plot = W_rp.loc[valid_dates]
    asset_names = [ALL_WEATHER_UNIVERSE[s]["name"] for s in W_plot.columns]
    ax_w.stackplot(W_plot.index, W_plot.values.T, labels=asset_names, alpha=0.85)
    ax_w.set_title("标准风险平价 (ERC) 资产持仓权重时序演变", fontsize=12, fontweight="bold")
    ax_w.set_ylabel("Portfolio Weight", fontsize=10)
    ax_w.set_xlabel("Date", fontsize=10)
    ax_w.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax_w.grid(True, linestyle="--", alpha=0.5)
    ax_w.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    fig_w.savefig(FIGURES_DIR / "02_asset_weights_stacked.png", dpi=300, bbox_inches="tight")
    plt.close(fig_w)

    # 图 3: 各资产实际风险贡献时序 (TRC 占比)
    rc_df = pd.DataFrame(rc_rp_records).T.fillna(0.0)
    rc_df.columns = [ALL_WEATHER_UNIVERSE[s]["name"] for s in rc_df.columns]
    fig_rc, ax_rc = plt.subplots(figsize=(10, 5))
    ax_rc.stackplot(rc_df.index, rc_df.values.T, labels=rc_df.columns, alpha=0.85)
    ax_rc.set_title("各资产总风险贡献 (Risk Contribution) 占比时序", fontsize=12, fontweight="bold")
    ax_rc.set_ylabel("Risk Contribution Ratio", fontsize=10)
    ax_rc.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax_rc.grid(True, linestyle="--", alpha=0.5)
    ax_rc.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    fig_rc.savefig(FIGURES_DIR / "03_risk_contribution_ratios.png", dpi=300, bbox_inches="tight")
    plt.close(fig_rc)

    # 图 4: Bootstrap 夏普分布
    plot_bootstrap_distribution(
        boot_macro["boot_sharpes"],
        observed_sharpe=boot_macro["observed_sharpe"],
        p_value=boot_macro["p_value"],
        ci_low=boot_macro["ci_95_low"],
        ci_high=boot_macro["ci_95_high"],
        title="宏观自适应风险平价 夏普比率 Bootstrap 分布检验 (N=10000)",
        save_path=FIGURES_DIR / "04_bootstrap_sharpe_distribution.png",
    )

    print("全部图表已成功保存至 figures/ 目录。")
    print("================================================================\n")


if __name__ == "__main__":
    run_all_weather_backtest()
