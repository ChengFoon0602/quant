"""
models/walk_forward.py — 滚动前向回测，真正样本外检验。

与 Purged CV OOF 的关键区别：
  - CV 的"未来信息泄露"已通过 purge 防止，但所有 fold 的验证集
    加起来覆盖了全历史。训练集虽然是"fold 前的数据"，
    但每个 fold 之后的数据点从未被任何 fold 训练过。
  - 滚动回测更保守：每年用截至当年的数据训练，预测下一年，
    完全不接触未来。
  - 两者都是样本外，但滚动回测匹配真实生产环境。

设计：
  - 年度再训练：每年用 expanding window 训练一次
  - 初始训练期：2010-01-01 ~ 2014-12-31（至少 5 年）
  - 第一个测试年：2015
  - 最后一个测试年：2025
  - 参数：复用 CV 最佳迭代次数（79 轮），不做 early stopping
  - 组合：hold_days=10, 双边 0.3% 成本, 动态仓位

用法:
    cd D:/桌面文件/quant
    python models/walk_forward.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import load_daily
from models.labels import align_X_y, build_labels, build_sample_weights
from models.lgbm_trainer import LGBM_PARAMS, FWD_DAYS, FEATURE_DIR
from models.portfolio_backtest import build_portfolio, performance_metrics

MODEL_DIR = Path(__file__).parent
FIGURES_DIR = MODEL_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
NUM_BOOST = 79  # CV 平均最佳迭代
COST = 0.003
HOLD_DAYS = 10
TOP_Q, BOTTOM_Q = 0.2, 0.2


def load_features_and_close(feature_dir: Path = FEATURE_DIR):
    x_path = feature_dir / "X_matrix.csv"
    X_raw = pd.read_csv(x_path, dtype=str)
    X_raw["date"] = pd.to_datetime(X_raw["date"])
    stock_col = X_raw.columns[1]
    X_long = X_raw.set_index(["date", stock_col]).astype(float)

    # close_matrix 只取 X_matrix 内实际出现的股票（禁止 sorted(cache)[:300] 切片）
    symbols = sorted(X_long.index.get_level_values(1).unique())
    close_data = {}
    for sym in symbols:
        df = load_daily(sym)
        if df is not None and len(df) >= 100:
            s = df.loc[(df.index >= "2010-01-01") & (df.index <= "2025-12-31"), "close"]
            if len(s) >= 100:
                close_data[sym] = s
    close_matrix = pd.DataFrame(close_data).sort_index()

    common_stocks = X_long.index.get_level_values(1).unique().intersection(close_matrix.columns)
    X_long = X_long.loc[(slice(None), common_stocks), :]
    close_matrix = close_matrix[common_stocks]
    return X_long, close_matrix


def main():
    print("=" * 70)
    print("滚动前向回测（Walk-Forward Backtest）")
    print("=" * 70)

    X_long, close_matrix = load_features_and_close()
    all_features = [c for c in X_long.columns]
    # PIT 掩码 = X_matrix 中实际出现的样本，分位数只在当日成员内算
    universe = pd.Series(True, index=X_long.index).unstack(fill_value=False)
    labels_full = build_labels(close_matrix, fwd_days=FWD_DAYS, top_q=TOP_Q, bottom_q=BOTTOM_Q,
                               universe=universe)
    aligned_full = align_X_y(X_long, labels_full).sort_index(level=0)
    valid_full = aligned_full.dropna(subset=["label"])
    print(f"全量样本: {len(valid_full):,}")

    # ── 年度窗口 ──
    test_years = list(range(2015, 2026))  # 2015-2025
    print(f"\n测试年份: {test_years[0]}-{test_years[-1]} ({len(test_years)} 年)")
    print(f"每窗口训练: expanding（使用截至前一年全部数据）")
    print(f"参数: num_boost_round={NUM_BOOST}, hold_days={HOLD_DAYS}, cost={COST}")

    wf_predictions = {}  # year -> (pred_matrix, port_df)

    for test_year in test_years:
        train_end = f"{test_year - 1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year}-12-31"

        # 训练集
        train_dates = valid_full.index.get_level_values(0)
        train_mask = (train_dates >= "2010-01-01") & (train_dates <= train_end)
        train_df = valid_full[train_mask]

        # 验证集（用于标签/组合构建的 close_matrix slice）
        test_mask_full = (valid_full.index.get_level_values(0) >= test_start) & (
            valid_full.index.get_level_values(0) <= test_end
        )
        test_df = valid_full[test_mask_full]

        if len(train_df) < 1000 or len(test_df) < 100:
            print(f"  {test_year}: 训练={len(train_df):,} 测试={len(test_df):,} — 样本不足，跳过")
            continue

        # 训练模型
        X_tr = train_df[all_features]
        y_tr = train_df["label"]
        w_tr = build_sample_weights(y_tr, "balanced")
        train_data = lgb.Dataset(X_tr.values, label=y_tr.values, weight=w_tr)
        model = lgb.train({**LGBM_PARAMS, "verbose": -1}, train_data, num_boost_round=NUM_BOOST)

        # 预测测试年份
        X_te = test_df[all_features]
        test_pred = model.predict(X_te.values)
        pred_series = pd.Series(test_pred, index=X_te.index, name="pred")

        # 转为宽格式矩阵
        pred_matrix = pred_series.unstack(level=1)
        test_dates = pred_matrix.index.intersection(close_matrix.index)
        test_stocks = pred_matrix.columns.intersection(close_matrix.columns)
        pred_matrix = pred_matrix.loc[test_dates, test_stocks]

        # 构建组合
        cm_test = close_matrix.loc[test_dates, test_stocks]
        if len(test_dates) > HOLD_DAYS + 2:
            port_df = build_portfolio(pred_matrix, cm_test, long_only=False, cost=COST, hold_days=HOLD_DAYS)
            m = performance_metrics(port_df["port_ret"])
            print(f"  {test_year}: 训练={len(train_df):>7,}  测试={len(test_df):>7,}  "
                  f"组合天数={len(port_df):>4}  年化={m['annual']:>+8.2%}  夏普={m['sharpe']:>+7.3f}")
            wf_predictions[test_year] = (pred_matrix, port_df)
        else:
            print(f"  {test_year}: 训练={len(train_df):>7,}  测试={len(test_df):>7,}  "
                  f"测试日不足，跳过组合构建")

    if not wf_predictions:
        print("无有效测试窗口")
        return

    # ── 汇总 ──
    all_ports = pd.concat([v[1]["port_ret"] for v in wf_predictions.values()])
    all_ports = all_ports.sort_index()
    m_wf = performance_metrics(all_ports)

    # 对比：Purged CV OOF 组合
    print("\n读取 Purged CV OOF 预测进行对比...")
    from models.portfolio_backtest import load_data as pb_load
    pred_lgb_cv, _, cm, mkt_ret, _, _ = pb_load()
    port_cv = build_portfolio(pred_lgb_cv, cm, long_only=False, cost=COST, hold_days=HOLD_DAYS)
    m_cv = performance_metrics(port_cv["port_ret"])

    # 市场基准：直接复用 load_data 的成员掩码等权日收益
    # （等权恒定权重无重叠 tranche；此前 rolling(HOLD_DAYS).mean() 平滑残留已移除）
    m_mkt = performance_metrics(mkt_ret)

    print(f"\n{'方法':<25} {'年化':>9} {'夏普':>7} {'最大回撤':>9}")
    print("-" * 56)
    print(f"{'滚动回测 (Walk-Forward)':<25} {m_wf['annual']:>+8.2%} {m_wf['sharpe']:>7.3f} {m_wf['mdd']:>+8.2%}")
    print(f"{'Purged CV OOF':<25} {m_cv['annual']:>+8.2%} {m_cv['sharpe']:>7.3f} {m_cv['mdd']:>+8.2%}")
    print(f"{'市场等权基准':<25} {m_mkt['annual']:>+8.2%} {m_mkt['sharpe']:>7.3f} {m_mkt['mdd']:>+8.2%}")

    # ── 逐年绩效 ──
    yearly = []
    for year, (_, port_df) in sorted(wf_predictions.items()):
        m = performance_metrics(port_df["port_ret"])
        yearly.append({"年份": year, "年化": m["annual"], "夏普": m["sharpe"], "最大回撤": m["mdd"], "天数": m["n"]})
    yearly_df = pd.DataFrame(yearly)
    print(f"\n逐年绩效:")
    for _, r in yearly_df.iterrows():
        print(f"  {r['年份']}: 年化={r['年化']:>+8.2%}  夏普={r['夏普']:>+7.3f}  回撤={r['最大回撤']:>+7.2%}")
    yearly_df.to_csv(MODEL_DIR / "walk_forward_yearly.csv", index=False)

    # ── 图表 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 累计净值：WF vs CV vs 市场
    ax = axes[0, 0]
    wf_cum = (1 + all_ports).cumprod()
    cv_cum = port_cv["cum"]
    mkt_cum = (1 + mkt_ret.reindex(wf_cum.index).fillna(0)).cumprod()
    ax.plot(wf_cum.index, wf_cum.values, linewidth=1.5, color="#2ca02c", label="滚动回测 WF")
    ax.plot(cv_cum.index, cv_cum.values, linewidth=1.0, color="#1f77b4", alpha=0.7, label="Purged CV OOF")
    ax.plot(mkt_cum.index, mkt_cum.values, linewidth=0.8, color="gray", alpha=0.5, label="市场等权")
    ax.axhline(1, color="black", linewidth=0.5)
    ax.set_title("累计净值：Walk-Forward vs Purged CV vs 市场")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. 逐年夏普柱状图
    ax = axes[0, 1]
    colors = ["#2ca02c" if s >= 0 else "#d62728" for s in yearly_df["夏普"]]
    bars = ax.bar(range(len(yearly_df)), yearly_df["夏普"].values, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1.40, color="green", linestyle="--", alpha=0.5, label="SR=1.40")
    ax.set_xticks(range(len(yearly_df)))
    ax.set_xticklabels(yearly_df["年份"].values, fontsize=8)
    ax.set_title("逐年夏普比率（Walk-Forward）")
    ax.set_ylabel("夏普比率")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. 逐年年化收益
    ax = axes[1, 0]
    colors_ret = ["#2ca02c" if r >= 0 else "#d62728" for r in yearly_df["年化"]]
    ax.bar(range(len(yearly_df)), [r * 100 for r in yearly_df["年化"].values], color=colors_ret, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(yearly_df)))
    ax.set_xticklabels(yearly_df["年份"].values, fontsize=8)
    ax.set_title("逐年年化收益（Walk-Forward）")
    ax.set_ylabel("年化收益 (%)")
    ax.grid(True, alpha=0.3)

    # 4. 逐年 vs 累积
    ax = axes[1, 1]
    ax.axis("off")
    rows = [["方法", "年化", "夏普", "最大回撤", "样本数"]]
    rows.append(["Walk-Forward", f"{m_wf['annual']:.2%}", f"{m_wf['sharpe']:.2f}",
                 f"{m_wf['mdd']:.2%}", f"{m_wf['n']}"])
    rows.append(["Purged CV OOF", f"{m_cv['annual']:.2%}", f"{m_cv['sharpe']:.2f}",
                 f"{m_cv['mdd']:.2%}", f"{m_cv['n']}"])
    rows.append(["市场等权", f"{m_mkt['annual']:.2%}", f"{m_mkt['sharpe']:.2f}",
                 f"{m_mkt['mdd']:.2%}", f"{m_mkt['n']}"])
    tbl = ax.table(cellText=rows, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    ax.set_title("汇总对比")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "walk_forward_summary.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\n图表保存: {fig_path}")

    # 汇总 CSV
    pd.DataFrame([{
        "method": "Walk-Forward", "annual": m_wf["annual"], "sharpe": m_wf["sharpe"],
        "mdd": m_wf["mdd"], "n_days": m_wf["n"],
    }, {
        "method": "Purged CV OOF", "annual": m_cv["annual"], "sharpe": m_cv["sharpe"],
        "mdd": m_cv["mdd"], "n_days": m_cv["n"],
    }, {
        "method": "市场等权", "annual": m_mkt["annual"], "sharpe": m_mkt["sharpe"],
        "mdd": m_mkt["mdd"], "n_days": m_mkt["n"],
    }]).to_csv(MODEL_DIR / "walk_forward_summary.csv", index=False)
    print("结果保存: models/walk_forward_summary.csv")

    print("\n滚动回测完成。")


if __name__ == "__main__":
    main()
