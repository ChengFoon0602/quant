"""
pit_universe_comparison.py — 幸存者偏差修复前后对比图。

前提：
  - 新 PIT 结果已生成（oof_predictions.csv / walk_forward_yearly.csv）
  - 旧 universe 备份存在（*_pre_pit.csv）
  - 旧净值曲线用 oof_predictions_pre_pit.csv 的股票列即时重建
    （不能再用 sorted(cache)[:300]——缓存已扩到 1298 只，切片不可复现）

输出：
  figures/pit_universe_overview.png    — PIT universe 结构 + 信号衰减
  figures/pit_vs_old_comparison.png    — 新旧净值曲线 + 指标对比

用法: python models/pit_universe_comparison.py
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import load_daily
from data.index_membership import load_membership
from models.portfolio_backtest import build_portfolio, performance_metrics, COST_BPS

MODEL_DIR = Path(__file__).parent
FIGURES_DIR = MODEL_DIR / "figures"

# 旧 universe 的信号指标（来自 report.md，训练输出已被覆盖）
OLD_METRICS = {"cv_auc": 0.5520, "rank_ic": 0.0926, "ic_ir": 0.5246}
NEW_METRICS = {"cv_auc": 0.5347, "rank_ic": 0.0574, "ic_ir": 0.3122}


def load_close_for(symbols) -> pd.DataFrame:
    close_data = {}
    for sym in symbols:
        df = load_daily(sym)
        if df is not None and len(df) >= 100:
            s = df.loc[(df.index >= "2010-01-01") & (df.index <= "2025-12-31"), "close"]
            if len(s) >= 100:
                close_data[sym] = s
    return pd.DataFrame(close_data).sort_index()


def main():
    membership = load_membership()

    # ── 图 1: PIT universe 结构 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) 幸存者覆盖率：历史成员被"最新名单"覆盖的比例
    final_members = set(membership.columns[membership.iloc[-1]])
    coverage = membership.apply(
        lambda row: len(set(membership.columns[row]) & final_members) / row.sum(), axis=1
    )
    ax = axes[0, 0]
    ax.plot(coverage.index, coverage * 100, color="#d62728", linewidth=1.5)
    ax.axhline(100, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("覆盖率 (%)")
    ax.set_title("用最新成分股回看历史能覆盖多少当时成员\n（2010年仅约1/3——这就是幸存者偏差的规模）")
    ax.grid(True, alpha=0.3)

    # (b) 月度成员调整数
    changes = (membership.astype(int).diff().abs().sum(axis=1) / 2).iloc[1:]
    ax = axes[0, 1]
    ax.bar(changes.index, changes.values, width=20, color="#1f77b4")
    ax.set_ylabel("调整只数 (单边)")
    ax.set_title("CSI 300 月度成员调整数（定调 6/12 月 + 临时调整）")
    ax.grid(True, alpha=0.3)

    # (c) 新 universe 沪/深构成 vs 旧 universe（纯深市）
    sh_count = membership[[c for c in membership.columns if c.startswith("6")]].sum(axis=1)
    sz_count = membership.sum(axis=1) - sh_count
    ax = axes[1, 0]
    ax.stackplot(membership.index, sh_count, sz_count,
                 labels=["沪市 (6xxxxx)", "深市 (0/3xxxxx)"],
                 colors=["#ff7f0e", "#2ca02c"], alpha=0.8)
    ax.axhline(0, color="red", linewidth=2)
    ax.annotate("旧事故 universe: 沪市 = 0 只（298 只纯深市）", xy=(0.03, 0.05),
                xycoords="axes fraction", color="red", fontsize=10)
    ax.set_ylabel("成员数")
    ax.set_title("PIT universe 沪/深构成（旧 universe 完全没有沪市股票）")
    ax.legend(loc="center left")
    ax.grid(True, alpha=0.3)

    # (d) 信号指标对比
    ax = axes[1, 1]
    labels = ["CV AUC - 0.5", "OOF Rank IC", "IC_IR"]
    old_vals = [OLD_METRICS["cv_auc"] - 0.5, OLD_METRICS["rank_ic"], OLD_METRICS["ic_ir"]]
    new_vals = [NEW_METRICS["cv_auc"] - 0.5, NEW_METRICS["rank_ic"], NEW_METRICS["ic_ir"]]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, old_vals, w, label="旧 universe (298 深市)", color="#9467bd")
    b2 = ax.bar(x + w / 2, new_vals, w, label="PIT CSI 300", color="#17becf")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("信号强度全面衰减：大盘股截面可预测性更弱")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p1 = FIGURES_DIR / "pit_universe_overview.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"图 1: {p1}")

    # ── 净值重建 ──
    pred_new = pd.read_csv(MODEL_DIR / "oof_predictions.csv", index_col=0, parse_dates=True)
    pred_new.columns = [str(c).zfill(6) for c in pred_new.columns]
    pred_old = pd.read_csv(MODEL_DIR / "oof_predictions_pre_pit.csv", index_col=0, parse_dates=True)
    pred_old.columns = [str(c).zfill(6) for c in pred_old.columns]

    close_new = load_close_for(pred_new.columns)
    close_old = load_close_for(pred_old.columns)

    print("重建组合净值 (hold_days=5, 双边 0.3%)...")
    # PIT 市场基准用成员掩码（与 portfolio_backtest 一致）；旧 universe 面板=其池子本身，无需掩码
    member_daily_close = None
    curves = {}
    for tag, pred, close in [("PIT", pred_new, close_new), ("旧", pred_old, close_old)]:
        curves[f"{tag} LS"] = build_portfolio(pred, close, long_only=False, cost=COST_BPS)
        curves[f"{tag} LO"] = build_portfolio(pred, close, long_only=True, cost=COST_BPS)
        daily_ret = close.shift(-2) / close.shift(-1) - 1
        if tag == "PIT":
            from data.index_membership import expand_to_daily
            member_daily_close = expand_to_daily(membership, close.index).reindex(
                columns=close.columns, fill_value=False)
            daily_ret = daily_ret.where(member_daily_close)
        mkt = daily_ret.mean(axis=1).dropna()
        curves[f"{tag} 市场"] = pd.DataFrame({"port_ret": mkt, "cum": (1 + mkt).cumprod()})

    for name, df in curves.items():
        m = performance_metrics(df["port_ret"])
        print(f"  {name:10s} 年化={m['annual']:+.2%}  夏普={m['sharpe']:+.3f}  回撤={m['mdd']:.2%}")

    # ── 图 2: 新旧对比 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    styles = {
        "PIT LS": ("#d62728", "-"), "PIT LO": ("#ff7f0e", "-"), "PIT 市场": ("gray", "-"),
        "旧 LS": ("#d62728", "--"), "旧 LO": ("#ff7f0e", "--"), "旧 市场": ("gray", "--"),
    }
    # (a) 旧 universe 净值（虚线）
    ax = axes[0, 0]
    for name in ["旧 LS", "旧 LO", "旧 市场"]:
        c, ls = styles[name]
        ax.plot(curves[name].index, curves[name]["cum"], color=c, linestyle=ls, label=name, linewidth=1.2)
    ax.set_yscale("log")
    ax.axhline(1, color="black", linewidth=0.5)
    ax.set_title("旧事故 universe（298 深市混合池）：LS 夏普 2.13")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # (b) PIT universe 净值（实线）
    ax = axes[0, 1]
    for name in ["PIT LS", "PIT LO", "PIT 市场"]:
        c, ls = styles[name]
        ax.plot(curves[name].index, curves[name]["cum"], color=c, linestyle=ls, label=name, linewidth=1.2)
    ax.set_yscale("log")
    ax.axhline(1, color="black", linewidth=0.5)
    ax.set_title("PIT CSI 300（790 只历史成员）：LS 夏普 0.08 — alpha 消失")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # (c) 指标对比条形
    ax = axes[1, 0]
    wf_new = pd.read_csv(MODEL_DIR / "walk_forward_summary.csv")
    wf_old = pd.read_csv(MODEL_DIR / "walk_forward_summary_pre_pit.csv")
    sr_old_wf = float(wf_old.loc[wf_old["method"] == "Walk-Forward", "sharpe"].iloc[0])
    sr_new_wf = float(wf_new.loc[wf_new["method"] == "Walk-Forward", "sharpe"].iloc[0])
    groups = ["LS (hold=5)", "LO (hold=5)", "WF (hold=10)"]
    sr_old = [performance_metrics(curves["旧 LS"]["port_ret"])["sharpe"],
              performance_metrics(curves["旧 LO"]["port_ret"])["sharpe"], sr_old_wf]
    sr_new = [performance_metrics(curves["PIT LS"]["port_ret"])["sharpe"],
              performance_metrics(curves["PIT LO"]["port_ret"])["sharpe"], sr_new_wf]
    x = np.arange(len(groups))
    w = 0.35
    b1 = ax.bar(x - w / 2, sr_old, w, label="旧 universe", color="#9467bd")
    b2 = ax.bar(x + w / 2, sr_new, w, label="PIT CSI 300", color="#17becf")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Sharpe Ratio")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("夏普对比：universe 修正后 alpha 大幅衰减")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # (d) WF 逐年夏普对比
    ax = axes[1, 1]
    y_new = pd.read_csv(MODEL_DIR / "walk_forward_yearly.csv")
    y_old = pd.read_csv(MODEL_DIR / "walk_forward_yearly_pre_pit.csv")
    years = y_new["年份"].astype(int)
    x = np.arange(len(years))
    ax.bar(x - w / 2, y_old["夏普"], w, label="旧 universe", color="#9467bd")
    ax.bar(x + w / 2, y_new["夏普"], w, label="PIT CSI 300", color="#17becf")
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Walk-Forward 逐年夏普对比 (hold=10)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p2 = FIGURES_DIR / "pit_vs_old_comparison.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"图 2: {p2}")


if __name__ == "__main__":
    main()
