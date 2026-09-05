"""
bear_short.py — 2018 式熊市做空端闭环（P2 主入口）。

目标: 验证研究弧线最后存活假设「regime 过滤 + 只做空」是否成立。
用已消除 selection bias 的逐年预测（WF PIT-Select）做纯空头腿，
叠加先验 regime 闸门（指数跌破 MA200）与流动性代理（成交额前 50%），
检验在真实融券成本（年化 8-10%）下是否仍为正期望。

这是研究弧线收尾，结论允许任何方向——诚实呈现比证明更重要。

数据来源（全部已存在）:
  - oof_predictions_pit_select.csv   消除 selection bias 的逐年预测
  - load_pit_panel('zz500')          close_matrix / member_daily
  - data.zz500_index                 sh.000905 价格指数（非全收益，regime 基准）
  - data 缓存 amount 字段            成交额矩阵（流动性代理）

检验矩阵:
  ① gated vs 永续（gate=None）基线对照
  ② regime 段内累计收益 + 指数涨跌 + gate 覆盖天数
  ③ 非 regime 时段：gate-off 日 gated 收益 ≈ 0 vs 永续大额亏损
  ④ 剔除 2018 段稳健性
  ⑤ Block bootstrap（block=20, n=10000）三档融券成本下 SR>0 的 p 值
  ⑥ 选股 vs 纯择时分解（固定任意 20% 篮子做空 = 无选股控制）

假设与局限（必须随报告披露）:
  - 可做空标的 = 成交额前 liquid_top 的 PIT 成员（流动性代理，非真实融券标的清单；
    baostock 无融券接口，akshare 仅当前快照）。代理未覆盖单票券源可得性/转融通限制。
  - 融券成本年化 borrow_rate∈{0,0.08,0.10}，卖出所得现金不计无风险利息（无 rebate）。
  - 指数为价格指数（忽略分红），做空方需补偿分红 → 做空成本被略低估（对 P2 有利方向）。
  - 不建模还券期限、集中度、强制平仓。

用法:
    cd strategies/zz500_pit_trial && python bear_short.py
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
from strategies.zz500_pit_trial.neutralize import load_amount_matrix
from data.zz500_index import load_zz500_index

COST_BPS = 0.00102     # 双边合计铁律 0.1%（2026-09 收口）
HOLD = 10               # 与 WF PIT-Select（SR +2.33 口径）一致
TOP_Q = 0.20            # 空头腿：截面 bottom 20%
MA_WIN = 200            # regime 主口径：指数 < MA200
BORROW_RATES = [0.0, 0.08, 0.10]   # 融券年化费率敏感性
N_BOOT = 10000

FIGURES_DIR = THIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# 命名日历窗口（检验矩阵②的对照，先验选定）
NAMED_WINDOWS = {
    "2015股灾": ("2015-06-15", "2016-02-29"),
    "2018熊市": ("2018-01-01", "2018-12-31"),
    "2022熊市": ("2022-01-01", "2022-12-31"),
    "2024小盘崩": ("2023-12-01", "2024-02-29"),
    "2025-04关税": ("2025-04-01", "2025-04-30"),
}


# ── 数据加载 ──────────────────────────────────────────────

def load_pred_pit_select() -> pd.DataFrame:
    """读消除 selection bias 的逐年预测 (date × symbol)。"""
    p = THIS_DIR / "oof_predictions_pit_select.csv"
    if not p.exists():
        raise FileNotFoundError(f"缺 {p}，先跑 report.py 的 walk_forward_pit_select 生成")
    return pd.read_csv(p, index_col=0, parse_dates=True)


def load_common_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """加载并对齐 (pred, close, member, idx_close)。"""
    pred = load_pred_pit_select()
    close_matrix, _, member_daily = load_pit_panel("zz500")
    idx = load_zz500_index()
    if idx is None:
        raise FileNotFoundError("缺中证500指数缓存，先跑 python data/zz500_index.py")

    common_cols = pred.columns.intersection(close_matrix.columns)
    common_dates = pred.index.intersection(close_matrix.index)
    pred = pred.loc[common_dates, common_cols]
    close = close_matrix.loc[common_dates, common_cols]
    member = member_daily.reindex(index=common_dates, columns=common_cols).fillna(False)
    idx_close = idx["close"].reindex(common_dates).ffill()
    return pred, close, member, idx_close


# ── Regime 过滤器（先验定义，禁止用 2018 拟合）────────────

def build_gate(idx_close: pd.Series, variant: str = "V1", confirm_days: int = 1) -> pd.Series:
    """构造逐日 regime 闸门（1=开空，0=关）。

    全部基于 sh.000905 价格指数，向后滚动，无未来函数。
    V1 主口径: close < MA200（行业标准）
    V2 纯动量: ret60 < 0
    V3 波动确认: V1 且 idx_vol20 > idx_vol60ma
    V4 深度趋势: V1 且 ret60 < -5%
    V5 慢闸门:  V1 + 入场连续 confirm_days 天确认
    """
    close = idx_close
    ma = close.rolling(MA_WIN, min_periods=MA_WIN).mean()
    v1 = close < ma
    ret60 = close / close.shift(60) - 1
    v2 = ret60 < 0
    idx_ret = close.pct_change()
    vol20 = idx_ret.rolling(20, min_periods=20).std()
    vol60ma = vol20.rolling(60, min_periods=60).mean()
    v3 = v1 & (vol20 > vol60ma)
    v4 = v1 & (ret60 < -0.05)
    v5 = v1.rolling(confirm_days, min_periods=confirm_days).min().fillna(0).astype(bool)

    base = {"V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5}[variant]
    if variant != "V5" and confirm_days > 1:
        base = base.rolling(confirm_days, min_periods=confirm_days).min().fillna(0).astype(bool)
    return base.fillna(False).astype(float)


def gate_segments(gate: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """gate==1 的连续段（合并 ≤3 个交易日的缺口），返回 (start, end) 含端点。"""
    g = gate.fillna(0).astype(int)
    idx = g.index
    on = (g == 1)
    runs = []
    i, n = 0, len(idx)
    while i < n:
        if on.iloc[i]:
            j = i
            while j + 1 < n and on.iloc[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    merged = []
    for (i0, j0) in runs:
        if merged and (i0 - merged[-1][1] - 1) <= 3:
            merged[-1] = (merged[-1][0], j0)
        else:
            merged.append((i0, j0))
    return [(idx[i0], idx[j0]) for (i0, j0) in merged]


# ── 流动性代理（可做空标的近似）───────────────────────────

def build_liquid_pred(pred: pd.DataFrame, amount_matrix: pd.DataFrame,
                      liquid_top: float | None) -> pd.DataFrame:
    """逐日只保留成交额前 liquid_top 的成员（PIT 成员 ∩ amount 有效），其余置 NaN。

    前置掩码后 build_portfolio 的 bottom-quantile 天然只在流动性子集内选。
    liquid_top=None 时不过滤。停牌/缺 amount 的成员自动剔除（不可做空）。
    """
    out = pred.copy()
    if liquid_top is None or liquid_top >= 1.0:
        return out
    amt = amount_matrix.reindex(index=pred.index, columns=pred.columns)
    mask = pred.notna() & amt.notna()
    thr = amt.where(mask).quantile(1 - liquid_top, axis=1)   # 逐日截面分位
    keep = amt.ge(thr, axis=0) & mask
    out[~keep] = np.nan
    return out


# ── 融券成本 ──────────────────────────────────────────────

def apply_borrow_cost(port_ret: pd.Series, w_held: pd.DataFrame, borrow_rate: float) -> pd.Series:
    """融券成本: 每日按空头暴露 W_held.abs().sum() × 年化费率/252 扣。"""
    exposure = w_held.abs().sum(axis=1).reindex(port_ret.index).fillna(0.0)
    borrow_cost = exposure * (borrow_rate / 252)
    return port_ret - borrow_cost


# ── 工具 ──────────────────────────────────────────────────

def yearly_metrics(ret: pd.Series) -> pd.DataFrame:
    rows = []
    for y, r in ret.dropna().groupby(ret.dropna().index.year):
        m = performance_metrics(r)
        rows.append({"year": y, "annual": m["annual"], "sharpe": m["sharpe"],
                     "mdd": m["mdd"], "n": m["n"]})
    return pd.DataFrame(rows)


def cum_in_window(port_ret: pd.Series, start: str, end: str) -> float:
    seg = port_ret.loc[start:end]
    if seg.dropna().empty:
        return float("nan")
    return (1 + seg.fillna(0)).prod() - 1


def index_change_in_window(idx_close: pd.Series, start: str, end: str) -> float:
    seg = idx_close.loc[start:end].dropna()
    if len(seg) == 0:
        return float("nan")
    return seg.iloc[-1] / seg.iloc[0] - 1


def short_pf(pred: pd.DataFrame, close: pd.DataFrame, gate: pd.Series | None,
             position_scale: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """跑空头腿组合，返回 (df, W_held)。gate pre-multiply；position_scale post-multiply。"""
    df, W = build_portfolio(pred, close, short_only=True, hold_days=HOLD,
                            gate=gate, position_scale=position_scale, return_weights=True)
    return df, W


# ── 图表 ──────────────────────────────────────────────────

def plot_gate_and_nav(idx_close, gate, nav_gated, nav_perm):
    """图1: (a) 指数 vs MA200 + gate 阴影 (b) gated vs 永续空头净值(0/8/10% 融券)。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(idx_close.index, idx_close.values, label="zz500 价格指数", linewidth=1.0, color="black")
    ax.plot(idx_close.index, idx_close.rolling(MA_WIN).mean().values, label=f"MA{MA_WIN}",
            linewidth=1.0, color="orange")
    for s, e in gate_segments(gate):
        ax.axvspan(s, e, color="red", alpha=0.12)
    ax.set_title(f"中证500 指数 vs MA{MA_WIN}（红色阴影 = 做空闸门开启）")
    ax.set_xlabel("date")
    ax.set_ylabel("index")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(nav_gated.index, nav_gated["cum"], label="gated short (borrow 0%)", linewidth=1.2)
    ax.plot(nav_perm.index, nav_perm["cum"], label="永续空头 (borrow 0%)", linewidth=1.2,
            linestyle="--", color="gray")
    ax.axhline(1.0, color="gray", linewidth=0.5)
    ax.set_title("gated vs 永续空头累计净值（扣 0.3% 双边成本）")
    ax.set_ylabel("净值")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "bear_gate_and_nav.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {path}")


def plot_yearly(yearly_df, gated_yearly, perm_yearly):
    """图2: (a) 逐年收益柱状 gated vs 永续 (b) 融券成本敏感性。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    years = yearly_df["year"].values
    w = 0.38
    ax.bar(years - w / 2, gated_yearly["annual"].values, width=w, label="gated short")
    ax.bar(years + w / 2, perm_yearly["annual"].values, width=w, label="永续空头", alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("逐年收益：gated vs 永续空头")
    ax.set_xlabel("year")
    ax.set_ylabel("annual return")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for rate in BORROW_RATES:
        ax.plot(yearly_df["year"], yearly_df[f"sharpe_b{int(rate * 100)}"],
                marker="o", label=f"borrow {rate:.0%}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("gated short 逐年夏普 × 融券成本")
    ax.set_xlabel("year")
    ax.set_ylabel("sharpe")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "bear_yearly.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {path}")


def plot_segments(seg_df):
    """图3: 段级累计收益柱状（2018 / 2024-05~09 高亮）。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#c44e52" if s in ("2018-03-13", "2024-05-22") else "#4c72b0"
              for s in seg_df["start"].astype(str)]
    ax.bar(range(len(seg_df)), seg_df["gated_cum"].values, color=colors)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xticks(range(len(seg_df)))
    ax.set_xticklabels([f"{s:%y-%m-%d}~{e:%y-%m-%d}" for s, e in zip(seg_df["start"], seg_df["end"])],
                       rotation=45, ha="right", fontsize=7)
    ax.set_title("Regime 段内累计收益（红色 = 2018 熊市段 / 2024-05~09 反弹段）")
    ax.set_ylabel("segment cumulative return")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = FIGURES_DIR / "bear_segments.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {path}")


# ── 主流程 ────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("P2: 2018 式熊市做空端闭环")
    print("=" * 72)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    pred, close, member, idx_close = load_common_panel()
    amount = load_amount_matrix(close)
    print(f"  预测 {pred.shape} | 收盘 {close.shape} | 指数 {len(idx_close)} 天 | "
          f"amount 覆盖 {amount.notna().sum().sum():,}")

    # 2. Regime 闸门（V1 主口径）
    print("\n[2] Regime 闸门（指数 < MA200）...")
    gate = build_gate(idx_close, variant="V1", confirm_days=1)
    segs = gate_segments(gate)
    on_pct = gate.mean()
    print(f"  V1 开启 {on_pct:.1%} 交易日, {len(segs)} 个段")
    for s, e in segs:
        print(f"    {s.date()} ~ {e.date()}  ({np.busday_count(s.date(), e.date()) + 1} 交易日)")

    # 3. 流动性代理（主口径 0.50）
    print("\n[3] 流动性代理（成交额前 50%）...")
    pred_liq = build_liquid_pred(pred, amount, liquid_top=0.50)
    valid = pred_liq.notna().sum(axis=1)
    print(f"  逐日有效股票数: 中位 {valid.median():.0f} / 原中位 {pred.notna().sum(axis=1).median():.0f}")

    # 4. 组合（全部 short_only, hold=10, 0.3% 双边）
    print("\n[4] 组合...")
    # 永续基线（无闸门）
    df_perm, W_perm = short_pf(pred_liq, close, gate=None)
    # gated 主口径（pre-multiply）
    df_gated, W_gated = short_pf(pred_liq, close, gate=gate)
    # gated 无流动性（流动性敏感性）
    df_gated_noliq, _ = short_pf(pred, close, gate=gate)
    # gated post-multiply（convention 敏感性）
    df_post, _ = short_pf(pred_liq, close, gate=None, position_scale=gate)

    m_perm, m_gated = performance_metrics(df_perm["port_ret"]), performance_metrics(df_gated["port_ret"])
    m_gated_noliq = performance_metrics(df_gated_noliq["port_ret"])
    m_post = performance_metrics(df_post["port_ret"])
    print(f"  永续空头  (borrow 0%): 年化 {m_perm['annual']:+.1%}  夏普 {m_perm['sharpe']:+.2f}")
    print(f"  gated      (borrow 0%): 年化 {m_gated['annual']:+.1%}  夏普 {m_gated['sharpe']:+.2f}")
    print(f"  gated 无流动性        : 年化 {m_gated_noliq['annual']:+.1%}  夏普 {m_gated_noliq['sharpe']:+.2f}")
    print(f"  gated post-multiply   : 年化 {m_post['annual']:+.1%}  夏普 {m_post['sharpe']:+.2f}")

    # 5. 融券成本三档
    print("\n[5] 融券成本敏感性...")
    ret_by_rate = {}
    for rate in BORROW_RATES:
        r = apply_borrow_cost(df_gated["port_ret"], W_gated, rate)
        ret_by_rate[rate] = r
        m = performance_metrics(r)
        print(f"  borrow {rate:.0%}: 年化 {m['annual']:+.1%}  夏普 {m['sharpe']:+.2f}  "
              f"回撤 {m['mdd']:.1%}")

    # 6. 检验矩阵
    print("\n[6] 检验矩阵...")

    # ① gated vs 永续
    print("  ① gated vs 永续对照（borrow 0%）:")
    gy = yearly_metrics(df_gated["port_ret"])
    py = yearly_metrics(df_perm["port_ret"])
    for _, row in gy.iterrows():
        p = py[py["year"] == row["year"]]
        print(f"    {int(row['year'])}: gated {row['sharpe']:+.2f} ({row['annual']:+.1%}) | "
              f"永续 {p['sharpe'].iloc[0]:+.2f} ({p['annual'].iloc[0]:+.1%})")

    # ② 段级收益
    print("  ② regime 段级收益（gated, borrow 0%）:")
    seg_rows = []
    for s, e in segs:
        seg_rows.append({
            "start": s, "end": e,
            "gated_cum": cum_in_window(df_gated["port_ret"], str(s.date()), str(e.date())),
            "index_chg": index_change_in_window(idx_close, str(s.date()), str(e.date())),
        })
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(THIS_DIR / "bear_short_segments.csv", index=False)
    for _, r in seg_df.iterrows():
        print(f"    {r['start'].date()} ~ {r['end'].date()}: 段内 {r['gated_cum']:+.1%} | "
              f"指数 {r['index_chg']:+.1%}")

    # 命名窗口覆盖
    print("  ②b 命名日历窗口 gate 覆盖:")
    win_rows = []
    for name, (s, e) in NAMED_WINDOWS.items():
        wgate = gate.loc[s:e]
        cov = wgate.sum()
        win_rows.append({
            "window": name, "start": s, "end": e,
            "gate_on_days": int(cov),
            "total_days": int(len(wgate)),
            "gated_cum": cum_in_window(df_gated["port_ret"], s, e),
            "index_chg": index_change_in_window(idx_close, s, e),
        })
        print(f"    {name:<10} 覆盖 {cov:.0f}/{len(wgate)} 天  gated {cum_in_window(df_gated['port_ret'], s, e):+.1%}")

    # ③ gate-off 时段
    off_mask = (gate == 0).reindex(df_gated["port_ret"].index).fillna(True)
    gated_off = df_gated["port_ret"][off_mask]
    perm_off = df_perm["port_ret"][off_mask]
    print(f"  ③ gate-off 时段: gated 累计 {gated_off.sum():+.2%} | 永续累计 {perm_off.sum():+.2%}")

    # ④ 剔除 2018 段稳健性（borrow 8%）
    ex18 = ret_by_rate[0.08].dropna()
    ex18 = ex18[~ex18.index.isin(df_gated["port_ret"].loc["2018-01-01":"2018-12-31"].index)]
    m_ex18 = performance_metrics(ex18)
    print(f"  ④ 剔除 2018 (borrow 8%): 年化 {m_ex18['annual']:+.1%}  夏普 {m_ex18['sharpe']:+.2f}")

    # ⑤ Block bootstrap 三档
    print("  ⑤ Block bootstrap (block=20, n=10000):")
    boot_rows = []
    for rate in BORROW_RATES:
        _, p = block_bootstrap_sharpe(ret_by_rate[rate])
        boot_rows.append({"borrow_rate": rate, "boot_p": p})
        print(f"    borrow {rate:.0%}: SR>0 p = {p:.4f}")

    # ⑥ 选股 vs 纯择时（固定任意 20% 篮子做空 = 无选股控制）
    rank = np.arange(len(pred.columns)) / max(len(pred.columns) - 1, 1)
    pred_flat = pd.DataFrame(np.tile(rank, (len(pred), 1)), index=pred.index, columns=pred.columns)
    pred_flat = pred_flat.where(pred_liq.notna(), np.nan)   # 同 tradable 集合
    df_flat, _ = short_pf(pred_flat, close, gate=gate)
    m_flat = performance_metrics(df_flat["port_ret"])
    print(f"  ⑥ 选股 vs 纯择时 (borrow 0%): 选股 {m_gated['annual']:+.1%} (SR {m_gated['sharpe']:+.2f}) | "
          f"纯择时 {m_flat['annual']:+.1%} (SR {m_flat['sharpe']:+.2f}) | "
          f"差值(选股alpha) {m_gated['annual'] - m_flat['annual']:+.1%}")

    # 7. 保存
    print("\n[7] 保存...")
    summary_rows = []
    for rate in BORROW_RATES:
        r = ret_by_rate[rate]
        m = performance_metrics(r)
        _, p = block_bootstrap_sharpe(r)
        summary_rows.append({
            "gate": "V1", "borrow_rate": rate, "annual": m["annual"], "sharpe": m["sharpe"],
            "mdd": m["mdd"], "boot_p": p,
        })
    summary_rows += [
        {"gate": "永续", "borrow_rate": 0.0, "annual": m_perm["annual"], "sharpe": m_perm["sharpe"],
         "mdd": m_perm["mdd"], "boot_p": float("nan")},
        {"gate": "V1_noliq", "borrow_rate": 0.0, "annual": m_gated_noliq["annual"],
         "sharpe": m_gated_noliq["sharpe"], "mdd": m_gated_noliq["mdd"], "boot_p": float("nan")},
        {"gate": "V1_post", "borrow_rate": 0.0, "annual": m_post["annual"],
         "sharpe": m_post["sharpe"], "mdd": m_post["mdd"], "boot_p": float("nan")},
        {"gate": "V1_no_select", "borrow_rate": 0.0, "annual": m_flat["annual"],
         "sharpe": m_flat["sharpe"], "mdd": m_flat["mdd"], "boot_p": float("nan")},
    ]
    pd.DataFrame(summary_rows).to_csv(THIS_DIR / "bear_short_summary.csv", index=False)

    # 逐年（各融券成本档）
    yearly_out = gy.set_index("year")
    for rate in BORROW_RATES:
        ym = yearly_metrics(ret_by_rate[rate]).set_index("year")
        yearly_out[f"sharpe_b{int(rate * 100)}"] = ym["sharpe"]
        yearly_out[f"annual_b{int(rate * 100)}"] = ym["annual"]
    yearly_out = yearly_out.reset_index()
    yearly_out.to_csv(THIS_DIR / "bear_short_yearly.csv", index=False)

    # 逐日 gate（审计用）
    gate.to_csv(THIS_DIR / "bear_short_gate.csv", header=["gate"])

    # 8. 图表
    print("\n[8] 图表...")
    nav_gated = df_gated[["cum"]].copy()
    nav_perm = df_perm[["cum"]].copy()
    plot_gate_and_nav(idx_close, gate, nav_gated, nav_perm)
    plot_yearly(yearly_out, gy, py)
    plot_segments(seg_df)

    # 9. 结论
    print("\n" + "=" * 72)
    print("P2 结论")
    m0 = performance_metrics(ret_by_rate[0.0])
    m8 = performance_metrics(ret_by_rate[0.08])
    print(f"  永续空头 → gated (borrow 0%): SR {m_perm['sharpe']:+.2f} → {m0['sharpe']:+.2f}")
    print(f"  融券成本: borrow 0% SR {m0['sharpe']:+.2f} → 8% {m8['sharpe']:+.2f}")
    print(f"  2018 段贡献见 segments CSV；块自举 p 见 summary CSV")
    print("=" * 72)


if __name__ == "__main__":
    main()
