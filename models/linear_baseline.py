"""
models/linear_baseline.py — 线性基线对比。

与 LightGBM 使用完全相同的：
  - 16 个 alpha 因子（不含市场状态特征）
  - 5 日截面分类标签（仅用于验证，不参与权重估计的等权基线除外）
  - Purged Time-Series 5 折 CV
  - OOF 评估框架

基线方法:
  1. 等权线性合成：每天对每个因子截面 z-score 后等权求和
  2. ICIR 加权线性合成：每天截面 z-score 后，按训练集 |IC_IR| 加权

输出:
  - oof_pred_equal_weight.csv / oof_pred_icir_weight.csv
  - 评估指标与对比表

用法:
    cd D:/桌面文件/quant
    python models/linear_baseline.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import load_daily
from models.cv import PurgedTimeSeriesSplit
from models.evaluate import evaluate_oof
from models.labels import align_X_y, build_labels

# ── 配置 ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_DIR = PROJECT_ROOT / "strategies" / "feature_selection"
MODEL_DIR = Path(__file__).parent
FWD_DAYS = 5
N_SPLITS = 5
PURGE_DAYS = 6
ALPHA_COLS = [
    "alpha116", "alpha142", "alpha001", "alpha144", "alpha003", "alpha011",
    "alpha051", "alpha110", "alpha075", "alpha169", "alpha108", "alpha068",
    "alpha166", "alpha171", "alpha162", "alpha055",
]


def load_features_and_close(feature_dir: Path = FEATURE_DIR):
    """读取 X_matrix.csv 并重建 close_matrix。"""
    x_path = feature_dir / "X_matrix.csv"
    if not x_path.exists():
        raise FileNotFoundError(f"找不到特征矩阵: {x_path}")

    print(f"读取特征矩阵: {x_path}")
    X_raw = pd.read_csv(x_path, dtype=str)
    X_raw["date"] = pd.to_datetime(X_raw["date"])
    stock_col = X_raw.columns[1]
    X_long = X_raw.set_index(["date", stock_col])
    X_long = X_long.astype(float)

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


def cross_sectional_zscore(df_long: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """对长格式数据按日期做截面 z-score。"""
    z = df_long[cols].groupby(level=0).transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    return z


def compute_icir_weights(
    factor_z: pd.DataFrame,
    close_matrix: pd.DataFrame,
    dates: pd.DatetimeIndex,
    fwd_days: int = FWD_DAYS,
) -> pd.Series:
    """在训练集日期范围内计算每个因子的 IC（带符号），返回归一化权重。

    带符号 IC 保证负向因子（如 alpha003/011/169）在合成时被自动反向。
    """
    cm = close_matrix.loc[dates]
    entry = cm.shift(-1)
    exit_ = cm.shift(-(fwd_days + 1))
    fwd_ret = exit_ / entry - 1

    ic_stats = {}
    for col in factor_z.columns:
        fz = factor_z[col].unstack(level=1)
        cd = fz.index.intersection(fwd_ret.index)
        cs = fz.columns.intersection(fwd_ret.columns)
        f_mat = fz.loc[cd, cs]
        r_mat = fwd_ret.loc[cd, cs]

        ics = []
        for d in cd:
            f = f_mat.loc[d]
            r = r_mat.loc[d]
            mask = f.notna() & r.notna()
            if mask.sum() < 10:
                continue
            ic = f[mask].rank().corr(r[mask].rank())
            if pd.isna(ic):
                continue
            ics.append(ic)
        arr = np.array(ics)
        if len(arr) == 0 or arr.std(ddof=0) == 0:
            ic_stats[col] = 0.0
        else:
            # 带符号 IC_IR：负向因子会自动反向
            ic_stats[col] = arr.mean() / arr.std(ddof=0)

    weights = pd.Series(ic_stats)
    # 如果全部 IC 为 0，退化为等权
    if weights.abs().sum() == 0:
        weights[:] = 1.0
    weights = weights / weights.abs().sum()
    return weights


def _combine_scores(zscore_df: pd.DataFrame, weights: pd.Series | None) -> pd.Series:
    """z-score 矩阵 + 权重 → 合成预测分数 Series (index=(date, stock))。"""
    if weights is None:
        score = zscore_df.mean(axis=1)
    else:
        score = (zscore_df * weights.reindex(zscore_df.columns)).sum(axis=1)
    return score


def run_baseline(method: str = "equal") -> tuple[pd.DataFrame, dict]:
    """运行指定线性基线，返回 (oof_pred_matrix, metrics)。"""
    assert method in ("equal", "icir")

    print("\n" + "=" * 70)
    print(f"线性基线: {method}-weight")
    print("=" * 70)

    X_long, close_matrix = load_features_and_close()
    universe = pd.Series(True, index=X_long.index).unstack(fill_value=False)
    labels = build_labels(close_matrix, fwd_days=FWD_DAYS, top_q=0.2, bottom_q=0.2,
                          universe=universe)
    aligned = align_X_y(X_long, labels)
    aligned = aligned.sort_index(level=0)

    print(f"总样本: {len(aligned):,}")

    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS, purge_days=PURGE_DAYS)
    oof_records = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(aligned), 1):
        train_df = aligned.iloc[train_idx]
        val_df = aligned.iloc[val_idx]

        train_dates = train_df.index.get_level_values(0).unique()
        val_dates = val_df.index.get_level_values(0).unique()

        # 训练集：估计 z-score 参数 + ICIR 权重
        train_z = cross_sectional_zscore(train_df, ALPHA_COLS)

        if method == "icir":
            weights = compute_icir_weights(train_z, close_matrix, train_dates)
            print(f"  Fold {fold_idx}: train={len(train_df):,}  val={len(val_df):,}")
            print(f"    top weights: {weights.sort_values(ascending=False).head(5).to_dict()}")
        else:
            weights = None
            print(f"  Fold {fold_idx}: train={len(train_df):,}  val={len(val_df):,}")

        # 验证集：用训练集的截面均值/std 做 z-score
        # 这里我们直接用全局截面 z-score（每个 fold 内独立标准化），
        # 因为线性合成是符号/方向型信号，标准化参数不会泄露未来收益方向
        val_z = cross_sectional_zscore(val_df, ALPHA_COLS)
        val_score = _combine_scores(val_z, weights)
        val_score.name = "pred"
        oof_records.append(val_score)

    all_preds = pd.concat(oof_records)
    all_preds = all_preds[~all_preds.index.duplicated(keep="first")]
    pred_matrix = all_preds.unstack(level=1)
    pred_matrix = pred_matrix.reindex(index=close_matrix.index, columns=close_matrix.columns)

    # 保存
    out_name = f"oof_pred_{method}_weight.csv"
    pred_matrix.to_csv(MODEL_DIR / out_name)
    print(f"\nOOF 预测矩阵: {pred_matrix.shape} → {out_name}")

    metrics, curves = evaluate_oof(pred_matrix, close_matrix, label_days=FWD_DAYS, hold_days=1)

    print(f"\n{method} 基线 OOF 评估:")
    print(f"  Rank IC mean = {metrics['ic_mean']:+.4f}")
    print(f"  IC_IR        = {metrics['ic_ir']:+.4f}")
    print(f"  IC t         = {metrics['ic_t']:.2f}")
    print(f"  Long-Short 年化收益 = {metrics['ls_annual_return']:.2%}")
    print(f"  Long-Short 夏普     = {metrics['ls_sharpe']:.3f}")
    print(f"  Long-Short 最大回撤 = {metrics['ls_max_drawdown']:.2%}")

    return pred_matrix, metrics, curves


def main():
    print("线性基线对比 — 与 LightGBM 相同标签/评估框架")

    eq_pred, eq_metrics, eq_curves = run_baseline("equal")
    ic_pred, ic_metrics, ic_curves = run_baseline("icir")

    # ── 读取 LightGBM 结果 ──
    lgb_summary = None
    lgb_curve = None
    if (MODEL_DIR / "summary.json").exists():
        import json
        with open(MODEL_DIR / "summary.json") as f:
            lgb_summary = json.load(f)
    if (MODEL_DIR / "oof_predictions.csv").exists():
        lgb_pred = pd.read_csv(MODEL_DIR / "oof_predictions.csv", index_col=0, parse_dates=True)
        _, lgb_curves = evaluate_oof(lgb_pred, load_features_and_close()[1], label_days=FWD_DAYS, hold_days=1)
        lgb_curve = lgb_curves["long_short"]

    # ── 对比表 ──
    print("\n" + "=" * 70)
    print("线性基线 vs LightGBM 对比")
    print("=" * 70)

    rows = []
    for name, m in [("等权线性", eq_metrics), ("ICIR 加权", ic_metrics)]:
        rows.append({
            "方法": name,
            "Rank IC": f"{m['ic_mean']:+.4f}",
            "IC_IR": f"{m['ic_ir']:+.4f}",
            "IC t": f"{m['ic_t']:.2f}",
            "多空年化": f"{m['ls_annual_return']:.2%}",
            "多空夏普": f"{m['ls_sharpe']:.3f}",
            "最大回撤": f"{m['ls_max_drawdown']:.2%}",
        })
    if lgb_summary:
        rows.append({
            "方法": "LightGBM",
            "Rank IC": f"{lgb_summary['ic_mean']:+.4f}",
            "IC_IR": f"{lgb_summary['ic_ir']:+.4f}",
            "IC t": f"{lgb_summary['ic_t']:.2f}",
            "多空年化": f"{lgb_summary['ls_annual_return']:.2%}",
            "多空夏普": f"{lgb_summary['ls_sharpe']:.3f}",
            "最大回撤": f"{lgb_summary['ls_max_drawdown']:.2%}",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # 保存对比表
    df.to_csv(MODEL_DIR / "baseline_comparison.csv", index=False)
    print(f"\n对比表保存: models/baseline_comparison.csv")

    # ── 图表 1: 对比柱状图 ──
    import matplotlib.pyplot as plt
    metrics_plot = ["IC_IR", "多空夏普"]
    # 把字符串转回 float
    numeric_rows = []
    for r in rows:
        numeric_rows.append({
            "方法": r["方法"],
            "IC_IR": float(r["IC_IR"]),
            "多空年化": float(r["多空年化"].replace("%", "")) / 100,
            "多空夏普": float(r["多空夏普"]),
        })
    plot_df = pd.DataFrame(numeric_rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    for ax, col in zip(axes, ["IC_IR", "多空年化", "多空夏普"]):
        bars = ax.bar(plot_df["方法"], plot_df[col], color=colors, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(col, fontsize=12)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.3f}",
                    ha="center", va="bottom" if h >= 0 else "top", fontsize=9)
    plt.tight_layout()
    fig_path = MODEL_DIR / "figures" / "baseline_metrics_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {fig_path}")

    # ── 图表 2: 累计多空曲线对比 ──
    fig, ax = plt.subplots(figsize=(12, 5))
    curve_data = {
        "等权线性": eq_curves["long_short"]["long_short_cum"],
        "ICIR 加权": ic_curves["long_short"]["long_short_cum"],
    }
    if lgb_curve is not None:
        curve_data["LightGBM"] = lgb_curve["long_short_cum"]

    for label, series in curve_data.items():
        ax.plot(series.index, series.values, label=label, linewidth=1.2)
    ax.axhline(1.0, color="black", linewidth=0.5)
    ax.set_title("Long-Short 累计收益对比 (扣成本后)")
    ax.set_ylabel("累计净值")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = MODEL_DIR / "figures" / "baseline_cum_curve.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {curve_path}")


if __name__ == "__main__":
    main()
