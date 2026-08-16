"""
report.py — 方向C 延伸④ 跨指数验证：沪深300 因子拥挤度（vs 中证500）。

复用 zz500_crowding_trial 的整套拥挤度代码（crowding / fundamental_crowding /
event_study / tradability），在本目录 config.py（INDEX="hs300"）下驱动。
对 zz500 与 hs300 各跑一遍同一套分析，回答三个问题：
  1. 事件数 / 低拥挤月数是否随指数翻倍 → bootstrap 是否更有机会过 0.05
  2. 「量价延续 vs 基本面反转」双面体在沪深300 是否成立
  3. 低拥挤择时可交易性在沪深300 更强还是更弱

方法论纪律（与 zz500 报告一致）：
  - 月末采样（复用方向2 month_end_dates），季频因子日频前向填充后的 IC 自相关
    已在方向2 用 test_monthly_sampling.py 实证；这里直接月末采样
  - 全部用截至 t 的数据（C1 用 t 月末截面，C2-C4 用 t 及之前历史窗口）
  - PIT 成员掩码：量价/基本面因子都 mask 到 load_pit_panel 的成员矩阵
  - 领先-滞后：拥挤度 t → 收益 t+1..t+h，无前瞻

数据（零新拉取）：
  - 量价池：16 alpha 因子在 hs300 PIT 面板上算（X_matrix_hs300.csv 缓存）；
    zz500 直接读 zz500_pit_trial/X_matrix.csv
  - 基本面池：方向2 缓存（cache_fundamental + cache_valuation）。⚠️ 缓存为
    zz500 域（1625 只），hs300 成员中无 mega-cap（600519/601398/601318 等
    全不在），月末可用股票数中位数 ~138/300（~46%），报告如实披露。

用法:
    cd strategies/hs300_crowding_trial && python report.py
"""

from __future__ import annotations

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = __import__("pathlib").Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent.parent
ZZ500_DIR = PROJECT_ROOT / "strategies" / "zz500_crowding_trial"
FEATURE_SEL_DIR = PROJECT_ROOT / "strategies" / "feature_selection"
PIT_TRIAL_DIR = PROJECT_ROOT / "strategies" / "zz500_pit_trial"

# 本目录 config.py 必须在 sys.path 最前（共享模块 `from config import ...`
# 因此拿到 INDEX="hs300" 的配置）。注意 insert 顺序：最后 insert 的在最前。
sys.path.insert(0, str(FEATURE_SEL_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ZZ500_DIR))
sys.path.insert(0, str(THIS_DIR))  # 最后插入 → sys.path[0]

from config import (  # noqa: E402
    FIGURES_DIR, CONDITION_LOOKBACK, FACTOR_COLS, MARKET_EVENTS,
    DATE_START, DATE_END, INDEX,
)

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from build_pit_matrix import load_pit_panel, build_market_features_pit  # noqa: E402
from crowding import (  # noqa: E402
    compute_all, factor_monthly_returns, month_end_dates,
    load_factor_matrix, wide_to_long, align_direction,
)
from fundamental_crowding import FUNDAMENTAL_COLS, DIRECTION_MAP  # noqa: E402
from event_study import composite_crowding, extreme_events, event_windows, summarize  # noqa: E402
from tradability import run_tradability  # noqa: E402

IDX_LABEL = {"hs300": "沪深300", "zz500": "中证500"}
IDX_MEMBERS = {"hs300": 300, "zz500": 500}


def section(title: str):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def conditional_return_analysis(crowd_ts: pd.DataFrame, factor_ret: pd.DataFrame,
                                lookback: int = CONDITION_LOOKBACK) -> pd.DataFrame:
    """拥挤度分桶 → 未来 lookback 月平均因子收益（与 zz500 report.py 同口径）。

    综合拥挤度（C1-C4 标准化均值）在 t 分桶，收益 = t+1..t+lookback 的
    factor_ret 均值（领先-滞后，无前瞻）。输出每桶的均值/t 值/样本数。
    """
    z = crowd_ts.sub(crowd_ts.mean()).div(crowd_ts.std())
    composite = z.mean(axis=1)
    ret_roll = factor_ret.mean(axis=1)
    fwd = ret_roll.shift(-1).rolling(lookback).mean().shift(-(lookback - 1))[::-1]

    common = composite.index.intersection(fwd.dropna().index)
    comp = composite.loc[common]
    fw = fwd.loc[common]

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


def build_fundamental_long(close: pd.DataFrame, member: pd.DataFrame) -> pd.DataFrame:
    """基本面 20 因子 → 方向翻转 → 长表（PIT 成员掩码），复用已加载面板。

    与 fundamental_crowding.load_fundamental_long 同逻辑，但传入已加载的
    close/member，避免在本进程重复 load_pit_panel（每次 ~60s）。
    """
    from signals.fundamental.factors import compute_factor_tensor
    tensor = compute_factor_tensor(close, fields=FUNDAMENTAL_COLS)
    tensor = {f: df.where(member) for f, df in tensor.items()}
    tensor = align_direction(tensor, DIRECTION_MAP)
    return wide_to_long(tensor)


def build_alpha_matrix(close: pd.DataFrame, volume: pd.DataFrame,
                       member: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """hs300 量价 16 因子长表（PIT 成员掩码），缓存 X_matrix_hs300.csv。

    与 build_pit_matrix.build_and_save 同构造，但因子池用 FACTOR_COLS（与
    zz500 完全一致，跨指数可比）。缓存存在时直接复用。
    """
    path = THIS_DIR / "X_matrix_hs300.csv"
    if path.exists() and not force:
        print(f"  复用缓存 {path.name}")
        return load_factor_matrix(path)

    from signals.alpha191.calculator import compute_factor_matrix
    print(f"  计算 {len(FACTOR_COLS)} 个 alpha 因子 × {close.shape[1]} 只股票...")
    _, factor_tensor = compute_factor_matrix(
        list(close.columns), FACTOR_COLS, start=DATE_START, end=DATE_END, verbose=True,
    )
    frames = [factor_tensor[fid].stack().rename(fid) for fid in FACTOR_COLS]
    X_factor = pd.concat(frames, axis=1)

    mkt_feat = build_market_features_pit(close, volume, member)
    mkt_long = mkt_feat.loc[X_factor.index.get_level_values(0)]
    mkt_long.index = X_factor.index
    X = pd.concat([X_factor, mkt_long], axis=1)

    member_long = member.stack()
    keep = member_long.reindex(X.index).fillna(False).astype(bool)
    X = X[keep.values]
    X.index.names = ["date", "symbol"]
    X.to_csv(path)
    print(f"  保存 {path} | 形状 {X.shape} | 股票数 {X.index.get_level_values(1).nunique()}")
    return X


def compute_index(idx: str) -> dict:
    """对一个指数跑完整套拥挤度分析，返回全部结果 dict。"""
    print(f"\n{'─' * 72}\n  计算 {IDX_LABEL[idx]}（{idx}）\n{'─' * 72}")
    close, volume, member = load_pit_panel(idx)
    turnover = build_market_features_pit(close, volume, member)["market_turnover_20d"]
    print(f"  面板 {close.shape} | 月末成员均数 {member.sum(axis=1).mean():.0f}")

    # ── 量价池 ──
    if idx == "zz500":
        X_long = load_factor_matrix(PIT_TRIAL_DIR / "X_matrix.csv")
    else:
        X_long = build_alpha_matrix(close, volume, member)
    med_v = month_end_dates(X_long.index.get_level_values(0).unique())
    crowd_v = compute_all(X_long, close, turnover)
    fr_v = factor_monthly_returns(X_long, med_v, close)
    cond_v = conditional_return_analysis(crowd_v, fr_v)
    crowd_v.to_csv(THIS_DIR / f"crowding_time_series_{idx}.csv")
    print(f"  量价: 时序 {crowd_v.shape} | 因子月均多空 {fr_v.mean(axis=1).mean():+.4f}")
    print("    " + cond_v.to_string(index=False).replace("\n", "\n    "))

    # ── 基本面池 ──
    X_fund = build_fundamental_long(close, member)
    med_f = month_end_dates(X_fund.index.get_level_values(0).unique())
    crowd_f = compute_all(X_fund, close, turnover, factor_cols=FUNDAMENTAL_COLS)
    fr_f = factor_monthly_returns(X_fund, med_f, close, factor_cols=FUNDAMENTAL_COLS)
    cond_f = conditional_return_analysis(crowd_f, fr_f)
    crowd_f.to_csv(THIS_DIR / f"fundamental_crowding_time_series_{idx}.csv")
    n_cov = X_fund.notna().any(axis=1).groupby(level=0).sum().mean()  # 月末至少一个因子可用的股票数
    print(f"  基本面: 时序 {crowd_f.shape} | 月末可用股票均数 {n_cov:.0f}")
    print("    " + cond_f.to_string(index=False).replace("\n", "\n    "))

    # ── 事件研究（>90 分位，相邻 6 月合并）──
    comp_v = composite_crowding(crowd_v)
    comp_f = composite_crowding(crowd_f)
    events_v = extreme_events(comp_v)
    events_f = extreme_events(comp_f)
    ev_sum_v = summarize(event_windows(events_v, fr_v), fr_v.mean(axis=1))
    ev_sum_f = summarize(event_windows(events_f, fr_f), fr_f.mean(axis=1))
    print(f"  事件数: 量价 {len(events_v)} ({[e.date().isoformat() for e in events_v]})")
    print(f"          基本面 {len(events_f)} ({[e.date().isoformat() for e in events_f]})")
    ev_sum_v.to_csv(THIS_DIR / f"event_study_{idx}_量价.csv", index=False)
    ev_sum_f.to_csv(THIS_DIR / f"event_study_{idx}_基本面.csv", index=False)
    print("  量价事件研究:\n    " + ev_sum_v.to_string(index=False).replace("\n", "\n    "))
    print("  基本面事件研究:\n    " + ev_sum_f.to_string(index=False).replace("\n", "\n    "))

    # ── 基本面低拥挤可交易性 ──
    trad = run_tradability(X_fund=X_fund, close_f=close, crowd_f=crowd_f,
                           fr_f=fr_f, index_name=idx)
    print(f"  可交易性: 策略 SR={trad['strategy']['sharpe']:+.3f} "
          f"(月频口径 {trad['strategy']['sr_12']:+.3f}) p={trad['strategy']['bootstrap_p']:.4f} "
          f"| 基准 SR={trad['baseline']['sharpe']:+.3f} p={trad['baseline']['bootstrap_p']:.4f}")
    print(f"    持仓 {trad['capacity']['n_hold_months']} 月 / {trad['capacity']['n_months']} "
          f"| 切换 {trad['capacity']['n_switches']} 次 | 成本敏感性 "
          + " ".join(f"{c*100:.1f}%→{v:+.3f}" for c, v in trad["cost_sensitivity"].items()))

    return {
        "idx": idx, "crowd_v": crowd_v, "fr_v": fr_v, "cond_v": cond_v,
        "crowd_f": crowd_f, "fr_f": fr_f, "cond_f": cond_f,
        "comp_v": comp_v, "comp_f": comp_f,
        "events_v": events_v, "events_f": events_f,
        "ev_sum_v": ev_sum_v, "ev_sum_f": ev_sum_f,
        "trad": trad,
    }


# ── 图表 ─────────────────────────────────────────────────────

def plot_hs300_timeseries(r: dict):
    """图1: hs300 量价 C1-C4 拥挤度时序 + 综合拥挤度。"""
    crowd = r["crowd_v"]
    z = crowd.sub(crowd.mean()).div(crowd.std())
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    ax = axes[0]
    for col in crowd.columns:
        ax.plot(z.index, z[col].values, label=col, linewidth=1.2)
    for ev_date, ev_name in MARKET_EVENTS.items():
        ax.axvline(pd.Timestamp(ev_date), color="red", linestyle="--", linewidth=0.8)
    ax.set_title(f"{IDX_LABEL[r['idx']]} 量价因子拥挤度时序（C1-C4 标准化）")
    ax.set_xlabel("date"); ax.set_ylabel("z-score")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    ax = axes[1]
    comp = z.mean(axis=1)
    ax.plot(comp.index, comp.values, color="#c44e52", linewidth=1.5, label="综合拥挤度")
    for ev_date, _ in MARKET_EVENTS.items():
        ax.axvline(pd.Timestamp(ev_date), color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(f"{IDX_LABEL[r['idx']]} 综合拥挤度 + 市场大事件")
    ax.set_xlabel("date"); ax.set_ylabel("z-score")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "01_hs300_crowding_timeseries.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"图表保存: {path}")


def plot_hs300_conditional(r: dict):
    """图2: hs300 量价 vs 基本面 条件收益柱状图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (name, cond) in zip(axes, [("量价 16 因子", r["cond_v"]), ("基本面 20 因子", r["cond_f"])]):
        vals = cond["mean"] * 100
        colors = ["#c44e52" if b == "Q4_high" else "#4c72b0" for b in cond["bucket"]]
        ax.bar(cond["bucket"], vals, color=colors)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(f"{IDX_LABEL[r['idx']]} {name} 条件收益")
        ax.set_xlabel("拥挤度分桶（Q1=低 → Q4=高）"); ax.set_ylabel("未来月均收益 (%)")
        ax.grid(True, alpha=0.3, axis="y")
        for i, row in cond.iterrows():
            ax.text(i, row["mean"] * 100 + 0.02, f"t={row['t']:.2f} n={int(row['n'])}",
                    ha="center", fontsize=8)
    plt.tight_layout()
    path = FIGURES_DIR / "02_hs300_conditional_returns.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"图表保存: {path}")


def plot_cross_index_composite(rv: dict, rh: dict):
    """图3: 两指数综合拥挤度时序对比（量价 / 基本面 分面板）。"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for ax, (key, label) in zip(axes, [("comp_v", "量价 16 因子"), ("comp_f", "基本面 20 因子")]):
        for r, color, name in [(rv, "#c44e52", "中证500"), (rh, "#4c72b0", "沪深300")]:
            comp = r[key]
            z = (comp - comp.mean()) / comp.std()
            ax.plot(z.index, z.values, label=f"{name}", color=color, linewidth=1.2)
        for ev_date, _ in MARKET_EVENTS.items():
            ax.axvline(pd.Timestamp(ev_date), color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("综合拥挤度 z-score")
        ax.set_title(f"{label}：中证500 vs 沪深300 综合拥挤度（各指数自身标准化）")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("date")
    plt.tight_layout()
    path = FIGURES_DIR / "03_cross_index_composite.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"图表保存: {path}")


def plot_cross_index_conditional(rv: dict, rh: dict):
    """图4（核心）: 两指数 量价/基本面 条件收益分组柱状图。"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    buckets = ["Q1_low", "Q2", "Q3", "Q4_high"]
    for ax, (key, label) in zip(axes, [("cond_v", "量价 16 因子"), ("cond_f", "基本面 20 因子")]):
        x = np.arange(4)
        w = 0.35
        for i, (r, color, name) in enumerate([(rv, "#c44e52", "中证500"), (rh, "#4c72b0", "沪深300")]):
            cond = r[key].set_index("bucket").reindex(buckets)
            vals = cond["mean"].values * 100
            ax.bar(x + (i - 0.5) * w, vals, w, label=name, color=color)
            for j, (t, n) in enumerate(zip(cond["t"].values, cond["n"].values)):
                if not np.isnan(t):
                    ax.text(x[j] + (i - 0.5) * w, vals[j] + 0.02,
                            f"t={t:.1f}", ha="center", fontsize=7)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(["Q1低", "Q2", "Q3", "Q4高"])
        ax.set_title(f"{label}：未来 12 月因子收益按拥挤度分桶")
        ax.set_ylabel("未来月均收益 (%)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = FIGURES_DIR / "04_cross_index_conditional.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"图表保存: {path}")


def plot_cross_index_tradability(rv: dict, rh: dict):
    """图5: 两指数 基本面低拥挤策略 vs 无条件基准 累计净值。"""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for r, color, style, name in [
        (rv, "#c44e52", "-", "中证500 低拥挤策略"),
        (rh, "#4c72b0", "-", "沪深300 低拥挤策略"),
    ]:
        nav = (1 + r["trad"]["strategy_ret"].fillna(0)).cumprod()
        ax.plot(nav.index, nav.values, color=color, linewidth=1.4, label=name)
    for r, color, style, name in [
        (rv, "#c44e52", "--", "中证500 无条件基准"),
        (rh, "#4c72b0", "--", "沪深300 无条件基准"),
    ]:
        nav = (1 + r["trad"]["baseline_ret"].fillna(0)).cumprod()
        ax.plot(nav.index, nav.values, color=color, linewidth=1.1, linestyle=style, alpha=0.7, label=name)
    ax.set_title("基本面低拥挤择时 vs 无条件因子多空（累计净值，0.3% 双边成本）")
    ax.set_xlabel("date"); ax.set_ylabel("累计净值")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "05_cross_index_tradability.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"图表保存: {path}")


def plot_hs300_event_study(r: dict):
    """图6: hs300 事件研究（极端拥挤后 3/6/12 月 vs 常态）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (name, ev) in zip(axes, [("量价池", r["ev_sum_v"]), ("基本面池", r["ev_sum_f"])]):
        x = np.arange(len(ev))
        w = 0.35
        ax.bar(x - w / 2, ev["event_mean_cum"], w, label="极端拥挤后", color="#c44e52")
        ax.bar(x + w / 2, ev["normal_mean_cum"], w, label="常态对照", color="#4c72b0")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels([f"{h}月" for h in ev["horizon_months"]])
        ax.set_title(f"{IDX_LABEL[r['idx']]} {name} 事件研究（>90 分位）")
        ax.set_ylabel("累计因子收益")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
        for i, row in ev.iterrows():
            ax.text(i - w / 2, row["event_mean_cum"] + 0.005, f"n={int(row['n_events'])}",
                    ha="center", fontsize=7)
            ax.text(i + w / 2, row["normal_mean_cum"] + 0.005, f"p={row['p_worse_than_normal']:.2f}",
                    ha="center", fontsize=7)
    plt.tight_layout()
    path = FIGURES_DIR / "06_hs300_event_study.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"图表保存: {path}")


# ── 主流程 ───────────────────────────────────────────────────

def print_comparison(rv: dict, rh: dict):
    """打印跨指数对比总表（供 report.md 引用）。"""
    section("跨指数对比：中证500 vs 沪深300")
    print("\n[1] 事件数 & 低拥挤月数（bootstrap 机会）")
    for name, r in [("中证500", rv), ("沪深300", rh)]:
        trad = r["trad"]
        print(f"  {name}: 极端事件 量价{len(r['events_v'])}/基本面{len(r['events_f'])} 个 | "
              f"低拥挤月 {trad['capacity']['n_hold_months']}/{trad['capacity']['n_months']} | "
              f"bootstrap p={trad['strategy']['bootstrap_p']:.4f}")

    print("\n[2] 双面体：量价 Q4 vs 基本面 Q1 条件收益")
    for name, r in [("中证500", rv), ("沪深300", rh)]:
        v_q4 = r["cond_v"].set_index("bucket").loc["Q4_high"]
        f_q1 = r["cond_f"].set_index("bucket").loc["Q1_low"]
        f_q4 = r["cond_f"].set_index("bucket").loc["Q4_high"]
        print(f"  {name}: 量价 Q4 {v_q4['mean']:+.4f} (t={v_q4['t']:.2f}) | "
              f"基本面 Q1 {f_q1['mean']:+.4f} (t={f_q1['t']:.2f}) | "
              f"基本面 Q4 {f_q4['mean']:+.4f} (t={f_q4['t']:.2f})")

    print("\n[3] 基本面低拥挤可交易性")
    rows = []
    for name, r in [("中证500", rv), ("沪深300", rh)]:
        s, b = r["trad"]["strategy"], r["trad"]["baseline"]
        cap = r["trad"]["capacity"]
        rows.append({
            "指数": name, "策略SR": s["sharpe"], "策略SR月频": s["sr_12"],
            "策略年化": s["annual"], "策略p": s["bootstrap_p"],
            "基准SR": b["sharpe"], "持仓月": cap["n_hold_months"],
            "切换次数": cap["n_switches"], "容量": cap["nominal_capacity_mid"],
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n[4] 成本敏感性（成本 → 策略 SR）")
    for name, r in [("中证500", rv), ("沪深300", rh)]:
        sens = r["trad"]["cost_sensitivity"]
        print(f"  {name}: " + " ".join(f"{c*100:.1f}%→{v:+.3f}" for c, v in sens.items()))


def main():
    section(f"方向C 延伸④：跨指数验证（{IDX_LABEL[INDEX]}）")
    print(f"[*] 复用 zz500_crowding_trial 代码，对 {INDEX} 与 zz500 各跑全套。")

    rv = compute_index("zz500")
    rh = compute_index("hs300")

    section("生成图表")
    plot_hs300_timeseries(rh)
    plot_hs300_conditional(rh)
    plot_cross_index_composite(rv, rh)
    plot_cross_index_conditional(rv, rh)
    plot_cross_index_tradability(rv, rh)
    plot_hs300_event_study(rh)

    print_comparison(rv, rh)
    section("完成")
    print("图表: 06 张 → figures/ | 量价因子矩阵缓存: X_matrix_hs300.csv")


if __name__ == "__main__":
    main()
