"""
models/train_final_model.py — 训练最终全量 LightGBM 模型。

逻辑:
  1. 复用 Purged CV 确定的超参数（models/lgbm_trainer.py 中的 LGBM_PARAMS）。
  2. 读取 5 个 fold 模型的 best_iteration，取平均作为最终模型的 boost_round。
  3. 在全部历史数据（X_matrix + labels）上训练一个单一模型。
  4. 保存为 models/lgbm_final_model.txt，用于未来预测 / 滚动回测。

用法:
    cd D:/桌面文件/quant
    python models/train_final_model.py
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

from data.fetcher import cache_summary, load_daily
from models.labels import align_X_y, build_labels, build_sample_weights
from models.lgbm_trainer import LGBM_PARAMS, FWD_DAYS, FEATURE_DIR

MODEL_DIR = Path(__file__).parent


def load_features_and_close(feature_dir: Path = FEATURE_DIR, n_stocks: int = 300):
    """读取 X_matrix.csv 并重建 close_matrix。"""
    x_path = feature_dir / "X_matrix.csv"
    if not x_path.exists():
        raise FileNotFoundError(f"找不到特征矩阵: {x_path}")

    print(f"读取特征矩阵: {x_path}")
    X_raw = pd.read_csv(x_path, dtype=str)
    X_raw["date"] = pd.to_datetime(X_raw["date"])
    stock_col = X_raw.columns[1]
    X_long = X_raw.set_index(["date", stock_col]).astype(float)

    cache = cache_summary()
    symbols = sorted(cache["symbol"].tolist())[:n_stocks]
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


def get_cv_best_iterations(model_dir: Path = MODEL_DIR) -> list[int]:
    """读取 5 个 fold 模型的实际树数量作为 best_iteration 代理。"""
    iters = []
    for i in range(1, 6):
        path = model_dir / f"lgbm_fold_{i}.txt"
        if not path.exists():
            raise FileNotFoundError(f"找不到 fold 模型: {path}。请先运行 lgbm_trainer.py")
        model = lgb.Booster(model_file=str(path))
        n_trees = model.num_trees()
        iters.append(n_trees)
        print(f"  fold {i}: num_trees = {n_trees}")
    return iters


def main():
    print("=" * 70)
    print("训练最终全量 LightGBM 模型")
    print("=" * 70)

    # ── 加载数据 ──
    X_long, close_matrix = load_features_and_close()
    all_feature_cols = [c for c in X_long.columns]
    print(f"\n特征数: {len(all_feature_cols)}")
    print(f"股票数: {close_matrix.shape[1]}  交易日: {close_matrix.shape[0]}")

    labels = build_labels(close_matrix, fwd_days=FWD_DAYS, top_q=0.2, bottom_q=0.2)
    aligned = align_X_y(X_long, labels)
    aligned = aligned.sort_index(level=0)
    print(f"总样本: {len(aligned):,}  有效标签: {aligned['label'].notna().sum():,}")

    # ── 确定 boost rounds ──
    print("\n从 CV fold 模型确定最佳迭代次数...")
    cv_iters = get_cv_best_iterations()
    avg_iter = int(np.mean(cv_iters))
    std_iter = np.std(cv_iters)
    print(f"  CV best_iterations: mean={avg_iter}  std={std_iter:.1f}  min={min(cv_iters)}  max={max(cv_iters)}")

    # ── 全量训练 ──
    print(f"\n在全部历史数据上训练最终模型 (num_boost_round={avg_iter})...")
    valid = aligned.dropna(subset=["label"])
    X_train = valid[all_feature_cols]
    y_train = valid["label"]

    w_train = build_sample_weights(y_train, "balanced")
    train_data = lgb.Dataset(X_train.values, label=y_train.values, weight=w_train)

    final_model = lgb.train(
        {**LGBM_PARAMS, "verbose": -1},
        train_data,
        num_boost_round=avg_iter,
    )

    final_path = MODEL_DIR / "lgbm_final_model.txt"
    final_model.save_model(str(final_path))
    print(f"\n最终模型保存: {final_path}")

    # ── 特征重要性 ──
    importance = pd.DataFrame({
        "feature": all_feature_cols,
        "gain": final_model.feature_importance(importance_type="gain"),
        "split": final_model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(MODEL_DIR / "final_model_feature_importance.csv", index=False)

    print("\n最终模型特征重要性 (gain):")
    for _, row in importance.iterrows():
        print(f"  {row['feature']:20s} gain={row['gain']:10.1f}  split={row['split']:6d}")

    # ── 可视化 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    top = importance.head(20)
    ax.barh(range(len(top)), top["gain"].values, color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_title("Final Model Feature Importance (gain)")
    ax.set_xlabel("Gain")

    ax = axes[1]
    ax.bar(range(1, 6), cv_iters, color="coral", edgecolor="white")
    ax.axhline(avg_iter, color="darkred", linestyle="--", label=f"mean={avg_iter}")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("CV Fold")
    ax.set_ylabel("Best Iteration")
    ax.set_title("CV Best Iterations (drive final num_boost_round)")
    ax.legend()

    plt.tight_layout()
    fig_path = MODEL_DIR / "figures" / "final_model_summary.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\n图表保存: {fig_path}")

    print("\n最终模型训练完成。")


if __name__ == "__main__":
    main()
