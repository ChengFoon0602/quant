"""
enhance.py — 中证500多头增强落地 + 容量检验（P1 主入口）。

目标: 验证路线② 残余价值最高的"多头指数增强"方向是否走得通。
用消除 selection bias 的逐年预测（WF 年度重选因子池）做中性化 LO 增强，
对比中证500指数基准算超额/TE/IR，再用成交额加权检验容量上限。

数据来源（全部已存在或由本流程生成）:
  - oof_predictions_pit_select.csv   消除 selection bias 的逐年预测
  - load_pit_panel('zz500')          close_matrix / member_daily
  - data.zz500_index                 中证500指数行情（sh.000905）
  - data.industry                    行业分类（当前归属快照，非 PIT，报告标注）
  - data 缓存 amount 字段            成交额矩阵

主流程:
  1. 加载全部数据
  2. pred_neutral = 截面中性化(行业哑变量 + log成交额 OLS 残差)
  3. 对比: LO-raw vs LO-neutral vs zz500 指数基准 → 超额/TE/IR
  4. 容量检验: AUM 网格冲击成本（k=0.3/1.0 两档）→ 夏普衰减
  5. 诊断: 行业暴露对比 + Rank IC 变化
  6. 存 CSV + 图表

用法:
    cd strategies/zz500_pit_trial && python enhance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "strategies" / "feature_selection"))

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from models.portfolio_backtest import (
    build_portfolio,
    performance_metrics,
    block_bootstrap_sharpe,
)
from strategies.feature_selection.build_pit_matrix import load_pit_panel
from strategies.zz500_pit_trial.neutralize import load_amount_matrix, neutralize_cross_section
from strategies.zz500_pit_trial.capacity import run_capacity_sweep
from data.zz500_index import load_zz500_index
from data.industry import load_industry

COST_BPS = 0.003
HOLD = 5
TOP_Q = 0.20
AUM_GRID = [0.5e8, 2e8, 5e8, 20e8, 50e8]  # 0.5亿 ~ 50亿

FIGURES_DIR = THIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ── 数据加载 ──────────────────────────────────────────────

def load_pred_pit_select() -> pd.DataFrame:
    """读消除 selection bias 的逐年预测 (date × symbol)。"""
    p = THIS_DIR / "oof_predictions_pit_select.csv"
    if not p.exists():
        raise FileNotFoundError(f"缺 {p}，先跑 report.py 的 walk_forward_pit_select 生成")
    return pd.read_csv(p, index_col=0, parse_dates=True)


def index_daily_ret(index_close: pd.DataFrame, common_dates) -> pd.Series:
    """指数基准收益，与 build_portfolio 同口径: r[t] = close(t+2)/close(t+1)-1。"""
    r = index_close["close"].shift(-2) / index_close["close"].shift(-1) - 1
    return r.reindex(common_dates)


def member_equal_weight_ret(close_matrix, member_daily, common_dates) -> pd.Series:
    """成员等权基准（第二基准），同口径。"""
    daily_ret = close_matrix.shift(-2) / close_matrix.shift(-1) - 1
    mask = member_daily.reindex(index=daily_ret.index, columns=daily_ret.columns).fillna(False)
    mret = daily_ret.where(mask).mean(axis=1)
    return mret.reindex(common_dates)


def excess_metrics(port_ret: pd.Series, bench_ret: pd.Series) -> tuple[dict, pd.Series]:
    """超额收益指标: 年化超额、信息比率(IR)、跟踪误差(TE)。"""
    common = port_ret.dropna().index.intersection(bench_ret.dropna().index)
    ex = port_ret.loc[common] - bench_ret.loc[common]
    m = performance_metrics(ex)
    te = ex.std() * np.sqrt(252)
    return {"ex_annual": m["annual"], "ir": m["sharpe"], "te": te, "n": len(ex)}, ex


def rank_ic_daily(pred_matrix: pd.DataFrame, close_matrix: pd.DataFrame) -> pd.Series:
    """每日 Spearman Rank IC: pred[t] vs fwd_ret[t]（同口径 close(t+2)/close(t+1)-1）。"""
    fwd = close_matrix.shift(-2) / close_matrix.shift(-1) - 1
    dates, ics = [], []
    for d in pred_matrix.index:
        if d not in fwd.index:
            continue
        pv, fv = pred_matrix.loc[d], fwd.loc[d]
        mask = pv.notna() & fv.notna()
        if mask.sum() < 30:
            continue
        dates.append(d)
        ics.append(pv[mask].corr(fv[mask], method="spearman"))
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="rank_ic")


def industry_exposure(pred_matrix, member_daily, industry_series) -> tuple[float, float]:
    """每日 top20% 行业权重 vs 成员行业权重 的平均绝对偏离（active industry weight）。

    返回 (平均偏离, 样本天数)。偏离 = Σ|w_top - w_mem| / 2。中性化应显著降低该值。
    """
    sym_ind = industry_series.astype(str)
    industries = sorted(industry_series.dropna().unique())
    devs = []
    for d in pred_matrix.index:
        pv = pred_matrix.loc[d]
        valid = pv[pv.notna() & pv.index.isin(sym_ind.index)]
        if len(valid) < 30:
            continue
        top = valid[valid >= valid.quantile(1 - TOP_Q)].index
        if len(top) < 10:
            continue
        w_top = top.map(sym_ind).value_counts(normalize=True).reindex(industries).fillna(0)
        if d in member_daily.index:
            mem = member_daily.loc[d]
            mem_syms = mem[mem].index
            w_mem = mem_syms.map(sym_ind).value_counts(normalize=True).reindex(industries).fillna(0)
        else:
            w_mem = pd.Series(1 / len(industries), index=industries)
        devs.append((w_top - w_mem).abs().sum() / 2)
    if not devs:
        return float("nan"), 0
    return float(np.mean(devs)), len(devs)


# ── 图表 ──────────────────────────────────────────────────

def plot_nav(results: dict, idx_ret: pd.Series, bench_name: str, bench_cum: pd.Series):
    """图1: 中性化前后 LO 累计净值 vs 指数基准 + 超额累计。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name, df in results.items():
        ax.plot(df["cum"], label=name, linewidth=1.2)
    ax.plot(bench_cum, label=bench_name, color="black", linewidth=1.0, linestyle="--")
    ax.axhline(1.0, color="gray", linewidth=0.5)
    ax.set_title("累计净值（扣 0.3% 双边成本）")
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, df in results.items():
        ex = df["port_ret"] - idx_ret.reindex(df["port_ret"].index)
        ex_cum = (1 + ex.fillna(0)).cumprod()
        ax.plot(ex_cum, label=f"{name} 超额", linewidth=1.2)
    ax.axhline(1.0, color="gray", linewidth=0.5)
    ax.set_title("累计超额收益 vs 中证500指数")
    ax.set_ylabel("超额净值")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "enhance_nav.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {path}")


def plot_capacity(cap_dfs: dict):
    """图2: 容量-夏普衰减曲线（k=0.3 / 1.0 两档）。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, cap_df in cap_dfs.items():
        ax.plot(cap_df["aum_yi"], cap_df["sharpe"], marker="o", label=f"k={k}")
    ax.axhline(0.5, color="red", linewidth=1.0, linestyle="--")
    ax.text(0.98, 0.5, "SR=0.5 容量下限", transform=ax.get_yaxis_transform(), color="red", ha="right")
    ax.set_xlabel("AUM (亿元)")
    ax.set_ylabel("LO 夏普（含冲击成本）")
    ax.set_title("中证500多头增强 容量-夏普衰减")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "enhance_capacity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {path}")


# ── 主流程 ────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("P1: 中证500多头增强落地 + 容量检验")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    pred = load_pred_pit_select()
    close_matrix, volume_matrix, member_daily = load_pit_panel("zz500")
    idx = load_zz500_index()
    if idx is None:
        raise FileNotFoundError("缺中证500指数缓存，先跑 python data/zz500_index.py")
    ind = load_industry()
    if ind is None:
        raise FileNotFoundError("缺行业缓存，先跑 python data/industry.py")

    common_cols = pred.columns.intersection(close_matrix.columns)
    common_dates = pred.index.intersection(close_matrix.index)
    pred = pred.loc[common_dates, common_cols]
    close = close_matrix.loc[common_dates, common_cols]
    member = member_daily.reindex(index=common_dates, columns=common_cols).fillna(False)
    print(f"  预测 {pred.shape} | 收盘 {close.shape} | 指数 {len(idx)} | 行业 {ind.notna().sum()}")

    # 2. 中性化
    print("\n[2] 截面中性化（行业哑变量 + log成交额）...")
    amount = load_amount_matrix(close)
    print(f"  amount 覆盖: {amount.notna().sum().sum():,} / {amount.size:,}")
    pred_neutral, diag = neutralize_cross_section(pred, ind, amount)
    corr = diag.dropna()
    print(f"  残差与原始预测相关性: 均值={corr.mean():+.3f} "
          f"（低=中性化洗掉较多信号；高=信号不在行业/市值上）")

    # 3. 组合对比
    print("\n[3] LO 组合对比（hold=5, 双边 0.3% 成本）...")
    lo_raw = build_portfolio(pred, close, long_only=True, cost=COST_BPS, hold_days=HOLD)
    lo_neu = build_portfolio(pred_neutral, close, long_only=True, cost=COST_BPS, hold_days=HOLD)
    m_raw, m_neu = performance_metrics(lo_raw["port_ret"]), performance_metrics(lo_neu["port_ret"])

    idx_ret = index_daily_ret(idx, common_dates)
    mem_ret = member_equal_weight_ret(close_matrix, member_daily, common_dates)

    ex_raw, ex_raw_s = excess_metrics(lo_raw["port_ret"], idx_ret)
    ex_neu, ex_neu_s = excess_metrics(lo_neu["port_ret"], idx_ret)
    m_idx = performance_metrics(idx_ret.dropna())
    m_mem = performance_metrics(mem_ret.dropna())

    _, p_raw = block_bootstrap_sharpe(lo_raw["port_ret"])
    _, p_neu = block_bootstrap_sharpe(lo_neu["port_ret"])

    print(f"  {'组合':<18} {'年化':>8} {'夏普':>7} {'回撤':>8} {'超额年化':>9} {'TE':>7} {'IR':>6} {'p':>6}")
    print(f"  {'LO-raw':<18} {m_raw['annual']:>+8.1%} {m_raw['sharpe']:>7.2f} {m_raw['mdd']:>8.1%} "
          f"{ex_raw['ex_annual']:>+9.1%} {ex_raw['te']:>7.2%} {ex_raw['ir']:>6.2f} {p_raw:>6.3f}")
    print(f"  {'LO-neutral':<18} {m_neu['annual']:>+8.1%} {m_neu['sharpe']:>7.2f} {m_neu['mdd']:>8.1%} "
          f"{ex_neu['ex_annual']:>+9.1%} {ex_neu['te']:>7.2%} {ex_neu['ir']:>6.2f} {p_neu:>6.3f}")
    print(f"  {'zz500指数':<18} {m_idx['annual']:>+8.1%} {m_idx['sharpe']:>7.2f} {m_idx['mdd']:>8.1%}")
    print(f"  {'成员等权':<18} {m_mem['annual']:>+8.1%} {m_mem['sharpe']:>7.2f} {m_mem['mdd']:>8.1%}")

    # 4. 容量检验
    print("\n[4] 容量检验（冲击成本 k=0.3/1.0，AUM 0.5~50亿）...")
    cap_dfs = {}
    for k in [0.3, 1.0]:
        cap_df, _ = run_capacity_sweep(pred_neutral, close, amount,
                                       AUM_GRID, k=k, hold_days=HOLD, cost=COST_BPS)
        cap_dfs[k] = cap_df
        for _, r in cap_df.iterrows():
            print(f"  k={k}  AUM={r['aum_yi']:>5.1f}亿  夏普={r['sharpe']:+.3f}  "
                  f"年化={r['annual']:+.2%}  平均冲击={r['avg_impact_bps']:.1f}bps")
    cap_all = pd.concat([df.assign(k=k) for k, df in cap_dfs.items()])
    cap_all.to_csv(THIS_DIR / "capacity_summary.csv", index=False)

    # 5. 诊断
    print("\n[5] 诊断...")
    ric_raw = rank_ic_daily(pred, close)
    ric_neu = rank_ic_daily(pred_neutral, close)
    print(f"  Rank IC: 原始={ric_raw.mean():+.4f} 中性化后={ric_neu.mean():+.4f}")
    dev_raw, n_raw = industry_exposure(pred, member, ind)
    dev_neu, n_neu = industry_exposure(pred_neutral, member, ind)
    print(f"  行业暴露偏离(active weight): 原始={dev_raw:.3f} 中性化后={dev_neu:.3f} (n={n_neu})")

    # 6. 保存
    summary = pd.DataFrame([
        {"portfolio": "LO-raw", "annual": m_raw["annual"], "sharpe": m_raw["sharpe"],
         "mdd": m_raw["mdd"], "ex_annual": ex_raw["ex_annual"], "te": ex_raw["te"],
         "ir": ex_raw["ir"], "boot_p": p_raw},
        {"portfolio": "LO-neutral", "annual": m_neu["annual"], "sharpe": m_neu["sharpe"],
         "mdd": m_neu["mdd"], "ex_annual": ex_neu["ex_annual"], "te": ex_neu["te"],
         "ir": ex_neu["ir"], "boot_p": p_neu},
    ])
    summary.to_csv(THIS_DIR / "enhance_summary.csv", index=False)

    # 图
    idx_cum = (1 + idx_ret.reindex(lo_neu.index).fillna(0)).cumprod()
    results = {"LO-raw": lo_raw, "LO-neutral": lo_neu}
    plot_nav(results, idx_ret, "zz500指数", idx_cum)
    plot_capacity(cap_dfs)

    # 结论
    print("\n" + "=" * 70)
    print("P1 结论")
    print(f"  中性化杀死 alpha? IR: {ex_raw['ir']:.2f} → {ex_neu['ir']:.2f}")
    print(f"  容量上限 (SR<0.5): 见 capacity_summary.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
