"""
models/lgbm_trainer.py — LightGBM 非线性多因子合成。

策略:
  - 标签: 未来 5 日截面收益率前 20% = 1, 后 20% = 0, 中间 60% = NaN 不参与训练。
  - 特征: 由 strategies/feature_selection/select_features.py 生成的 X_matrix.csv
    (16 个 alpha 因子 + 2 个市场状态特征)。
  - CV: Purged Time-Series Split (5 折, purge 6 天)。
  - 目标函数: binary (logloss)。
  - 早停: 验证集 AUC。
  - 复杂度约束: max_depth=5, num_leaves=31, min_child_samples=50。

输出:
  - OOF 预测分数 oof_predictions.csv (date × stocks)
  - 特征重要性 feature_importance.csv
  - 每折模型 models/lgbm_fold_*.txt
  - 评估指标与曲线到 models/report.md 和 figures/

用法:
    cd D:/桌面文件/quant
    python models/lgbm_trainer.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import load_daily
from models.cv import PurgedTimeSeriesSplit
from models.evaluate import evaluate_oof
from models.labels import align_X_y, build_labels, build_sample_weights, get_valid_samples

# ── 路径 ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_DIR = PROJECT_ROOT / "strategies" / "feature_selection"
MODEL_DIR = Path(__file__).parent
FIGURES_DIR = MODEL_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

N_SPLITS = 5
PURGE_DAYS = 6       # fwd_days=5 → 前瞻期 6 个交易日
FWD_DAYS = 5         # 预测未来 5 日收益
TOP_Q = 0.20
BOTTOM_Q = 0.20

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,            # bagging_fraction
    "colsample_bytree": 0.8,     # feature_fraction
    "subsample_freq": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_estimators": 1000,
    "verbosity": -1,
}

EARLY_STOPPING_ROUNDS = 50


def load_features_and_close(
    feature_dir: Path = FEATURE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 X_matrix.csv 并重建 close_matrix（用于标签和评估）。"""
    x_path = feature_dir / "X_matrix.csv"
    if not x_path.exists():
        raise FileNotFoundError(f"找不到特征矩阵: {x_path}。请先运行 select_features.py")

    print(f"读取特征矩阵: {x_path}")
    # pd.read_csv(index_col=[0,1]) 会把 unnamed 第二列解析为整数，
    # 这里显式按字符串读入再设置 MultiIndex，保持 symbol 格式。
    X_raw = pd.read_csv(x_path, dtype=str)
    X_raw["date"] = pd.to_datetime(X_raw["date"])
    stock_col = X_raw.columns[1]  # 通常是 'symbol' 或 'Unnamed: 1'
    X_long = X_raw.set_index(["date", stock_col])
    X_long = X_long.astype(float)
    # X_long index: (date, stock), columns: features
    feature_cols = [c for c in X_long.columns if c not in ("market_vol_20d", "market_turnover_20d")]

    # close_matrix 只取 X_matrix 内实际出现的股票——
    # 禁止 sorted(cache)[:300] 切片：缓存扩容后该切片会漂移（曾取出 298 只纯深市股票）
    symbols = sorted(X_long.index.get_level_values(1).unique())
    close_data = {}
    for sym in symbols:
        df = load_daily(sym)
        if df is not None and len(df) >= 100:
            s = df.loc[(df.index >= "2010-01-01") & (df.index <= "2025-12-31"), "close"]
            if len(s) >= 100:
                close_data[sym] = s
    close_matrix = pd.DataFrame(close_data).sort_index()

    # 对齐：X_long 的股票集合必须 <= close_matrix
    common_stocks = X_long.index.get_level_values(1).unique().intersection(close_matrix.columns)
    X_long = X_long.loc[(slice(None), common_stocks), :]
    close_matrix = close_matrix[common_stocks]

    return X_long, close_matrix


def _train_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    fold_idx: int,
) -> tuple[lgb.Booster, float]:
    """训练单个 fold，返回 (booster, best_auc)。"""
    X_tr, y_tr = get_valid_samples(train_df[feature_cols], train_df["label"])
    X_val, y_val = get_valid_samples(val_df[feature_cols], val_df["label"])

    print(f"  Fold {fold_idx}: train={len(X_tr):,}  val={len(X_val):,}  "
          f"pos_train={(y_tr==1).mean():.2%}  pos_val={(y_val==1).mean():.2%}")

    w_tr = build_sample_weights(y_tr, "balanced")

    train_data = lgb.Dataset(X_tr.values, label=y_tr.values, weight=w_tr)
    val_data = lgb.Dataset(X_val.values, label=y_val.values, reference=train_data)

    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    best_auc = model.best_score["val"]["auc"]
    model_path = MODEL_DIR / f"lgbm_fold_{fold_idx}.txt"
    model.save_model(str(model_path))
    print(f"    best val AUC={best_auc:.4f}  model saved: {model_path}")
    return model, best_auc


def _oof_predictions_to_matrix(
    oof_records: list[pd.DataFrame],
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """把每个 fold 的 val 预测拼回 (date × stock) 矩阵。"""
    all_preds = pd.concat(oof_records, axis=0)
    # 去重：某 (date, stock) 可能只出现在一个 fold 的 val 中
    all_preds = all_preds[~all_preds.index.duplicated(keep="first")]
    pred_matrix = all_preds.unstack(level=1)
    pred_matrix = pred_matrix.reindex(index=labels.index, columns=labels.columns)
    return pred_matrix


def main():
    print("=" * 70)
    print("LightGBM 非线性多因子合成")
    print("=" * 70)

    # ── 加载特征与收盘价 ──
    X_long, close_matrix = load_features_and_close()
    feature_cols = [c for c in X_long.columns if c not in ("market_vol_20d", "market_turnover_20d")]
    market_cols = [c for c in X_long.columns if c in ("market_vol_20d", "market_turnover_20d")]
    all_feature_cols = feature_cols + market_cols
    print(f"特征维度: {len(feature_cols)} 个 alpha 因子 + {len(market_cols)} 个市场状态特征")
    print(f"股票数: {close_matrix.shape[1]}  交易日: {close_matrix.shape[0]}")

    # ── 构建 5 日截面分类标签 ──
    # PIT 掩码 = X_matrix 中实际出现的 (date, symbol) 样本——
    # 截面分位数只在当日指数成员内计算
    universe = pd.Series(True, index=X_long.index).unstack(fill_value=False)
    print(f"\n构建标签: 未来 {FWD_DAYS} 日截面收益 前 {TOP_Q:.0%}=1 后 {BOTTOM_Q:.0%}=0")
    labels = build_labels(close_matrix, fwd_days=FWD_DAYS, top_q=TOP_Q, bottom_q=BOTTOM_Q,
                          universe=universe)
    aligned = align_X_y(X_long, labels)
    print(f"总样本: {len(aligned):,}  有效标签: {aligned['label'].notna().sum():,}  "
          f"占比 {aligned['label'].notna().mean():.2%}")

    # 样本按日期排序（CV 要求）
    aligned = aligned.sort_index(level=0)
    dates = aligned.index.get_level_values(0).unique().sort_values()
    print(f"时间跨度: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")

    # ── Purged Time-Series CV ──
    print(f"\n开始 Purged CV: {N_SPLITS} folds, purge_days={PURGE_DAYS}")
    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS, purge_days=PURGE_DAYS)

    oof_records = []
    fold_aucs = []
    feature_importance = pd.DataFrame(index=all_feature_cols)

    # 注意：PurgedTimeSeriesSplit.split 接受整数索引
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(aligned), 1):
        train_df = aligned.iloc[train_idx]
        val_df = aligned.iloc[val_idx]

        model, auc = _train_one_fold(train_df, val_df, all_feature_cols, fold_idx)
        fold_aucs.append(auc)

        # OOF 预测
        X_val, _ = get_valid_samples(val_df[all_feature_cols], val_df["label"])
        val_pred = model.predict(X_val.values, num_iteration=model.best_iteration)
        pred_series = pd.Series(val_pred, index=X_val.index, name="pred")
        oof_records.append(pred_series)

        # 特征重要性
        imp = model.feature_importance(importance_type="gain")
        feature_importance[f"fold_{fold_idx}"] = imp

    print(f"\n{'='*70}")
    print(f"CV AUC: mean={np.mean(fold_aucs):.4f}  std={np.std(fold_aucs):.4f}")
    print(f"        {', '.join(f'{a:.4f}' for a in fold_aucs)}")
    print(f"{'='*70}")

    # ── OOF 矩阵 & 评估 ──
    pred_matrix = _oof_predictions_to_matrix(oof_records, labels)
    pred_matrix.to_csv(MODEL_DIR / "oof_predictions.csv")
    print(f"\nOOF 预测矩阵: {pred_matrix.shape} → oof_predictions.csv")

    metrics, curves = evaluate_oof(pred_matrix, close_matrix, label_days=FWD_DAYS, hold_days=1)

    print("\nOOF 评估:")
    print(f"  Rank IC mean = {metrics['ic_mean']:+.4f}")
    print(f"  Rank IC std  = {metrics['ic_std']:.4f}")
    print(f"  IC_IR        = {metrics['ic_ir']:+.4f}")
    print(f"  IC t         = {metrics['ic_t']:+.2f}  (n={metrics['ic_days']})")
    print(f"  Long-Short 年化收益 = {metrics['ls_annual_return']:.2%}")
    print(f"  Long-Short 夏普     = {metrics['ls_sharpe']:.3f}")
    print(f"  Long-Short 最大回撤 = {metrics['ls_max_drawdown']:.2%}")

    # ── 特征重要性 ──
    feature_importance["mean"] = feature_importance.mean(axis=1)
    feature_importance["std"] = feature_importance.std(axis=1)
    feature_importance = feature_importance.sort_values("mean", ascending=False)
    feature_importance.to_csv(MODEL_DIR / "feature_importance.csv")

    print("\n特征重要性 (gain):")
    for feat, row in feature_importance.iterrows():
        print(f"  {feat:20s} mean={row['mean']:10.1f}  std={row['std']:8.1f}")

    # ── 图表 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. OOF Rank IC 累积
    ax = axes[0, 0]
    rank_ic = curves["rank_ic"]
    cum_ic = rank_ic.cumsum()
    ax.plot(cum_ic.index, cum_ic.values, linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Cumulative Rank IC (OOF)")
    ax.set_ylabel("Cumulative IC")
    ax.grid(True, alpha=0.3)

    # 2. 五分位收益曲线
    ax = axes[0, 1]
    quintile_ret = curves["quintile_ret"]
    for col in quintile_ret.columns:
        cum = (1 + quintile_ret[col].fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=col, linewidth=1.2)
    ax.set_title("Quintile Cumulative Returns (OOF)")
    ax.set_ylabel("Cumulative Return")
    ax.legend(title="Quintile", loc="upper left")
    ax.grid(True, alpha=0.3)

    # 3. 多空累计曲线
    ax = axes[1, 0]
    ls = curves["long_short"]
    ax.plot(ls.index, ls["long_short_cum"], color="darkgreen", linewidth=1.5)
    ax.fill_between(ls.index, 1.0, ls["long_short_cum"], alpha=0.2, color="green")
    ax.set_title("Long-Short Cumulative Return (Top - Bottom, After Cost)")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)

    # 4. 特征重要性柱状图
    ax = axes[1, 1]
    top_feats = feature_importance.head(20)
    y_pos = np.arange(len(top_feats))
    ax.barh(y_pos, top_feats["mean"].values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feats.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Feature Importance (gain, top 20)")
    ax.set_xlabel("Mean gain")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "lgbm_oof_summary.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\n图表保存: {fig_path}")

    # ── 保存指标摘要 ──
    summary = {
        "cv_auc_mean": float(np.mean(fold_aucs)),
        "cv_auc_std": float(np.std(fold_aucs)),
        "cv_aucs": [float(a) for a in fold_aucs],
        "n_features": len(all_feature_cols),
        "n_samples": int(len(aligned)),
        "fwd_days": FWD_DAYS,
        "purge_days": PURGE_DAYS,
        **{k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()},
    }
    pd.Series(summary).to_json(MODEL_DIR / "summary.json", force_ascii=False, indent=2)

    print("\n训练完成。")
    return summary, curves, feature_importance


if __name__ == "__main__":
    main()
