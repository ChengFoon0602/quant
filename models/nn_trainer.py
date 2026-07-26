"""
models/nn_trainer.py — MLP 深度学习对照。

与 LightGBM 相同的:
  - 20/60/20 截面分类标签
  - Purged Time-Series 5 折 CV + purge 6 天
  - OOF 评估 + 组合回测框架

架构: 3 层 MLP (128→64→32), BatchNorm, Dropout, ReLU, BCE loss
优化: AdamW, ReduceLROnPlateau, early stopping on val AUC

用法:
    cd D:/桌面文件/quant
    python models/nn_trainer.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import cache_summary, load_daily
from models.cv import PurgedTimeSeriesSplit
from models.labels import align_X_y, build_labels
from models.lgbm_trainer import FEATURE_DIR, FWD_DAYS
from models.portfolio_backtest import build_portfolio, performance_metrics

MODEL_DIR = Path(__file__).parent
FIGURES_DIR = MODEL_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
N_SPLITS, PURGE_DAYS = 5, 6
BATCH_SIZE = 4096
MAX_EPOCHS = 200
PATIENCE = 20
LR = 0.001
DROPOUT = 0.3


class MLP(nn.Module):
    def __init__(self, n_features: int, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_features_and_close(feature_dir: Path = FEATURE_DIR, n_stocks: int = 300):
    x_path = feature_dir / "X_matrix.csv"
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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)  # BCEWithLogitsLoss, model outputs logits
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device).float().unsqueeze(1)
            logits = model(Xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            all_labels.extend(yb.cpu().numpy().ravel())
    auc = roc_auc_score(all_labels, all_preds)
    return total_loss / max(n_batches, 1), auc


def train_fold(X_tr, y_tr, X_val, y_val, fold_idx: int):
    """训练单 fold，返回模型和 val preds。"""
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))

    tr_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                               torch.tensor(y_tr, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))
    tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2)

    model = MLP(X_tr.shape[1]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    best_auc, best_epoch, patience_counter = 0.0, 0, 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, tr_loader, optimizer, criterion, DEVICE)
        val_loss, val_auc = evaluate_model(model, val_loader, criterion, DEVICE)
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)

    _, train_auc = evaluate_model(model, tr_loader, criterion, DEVICE)
    print(f"  Fold {fold_idx}: train={len(X_tr):,}  val={len(X_val):,}  "
          f"best_epoch={best_epoch}  val_auc={best_auc:.4f}  train_auc={train_auc:.4f}")

    # Predict on validation set
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_val, dtype=torch.float32))
        val_pred = torch.sigmoid(val_logits).numpy().ravel()
    return val_pred, best_auc


def main():
    print("=" * 70)
    print("MLP 深度学习对照 — 相同标签 / CV / 评估框架")
    print("=" * 70)

    X_long, close_matrix = load_features_and_close()
    all_features = [c for c in X_long.columns]
    labels = build_labels(close_matrix, fwd_days=FWD_DAYS, top_q=0.2, bottom_q=0.2)
    aligned = align_X_y(X_long, labels).sort_index(level=0)
    valid = aligned.dropna(subset=["label"])
    # PyTorch 不能处理 NaN，用列中位数填充
    for col in all_features:
        med = valid[col].median()
        valid[col] = valid[col].fillna(med)
    print(f"训练样本: {len(valid):,}  |  特征数: {len(all_features)}")
    oof_records = []
    fold_aucs = []

    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS, purge_days=PURGE_DAYS)
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(valid), 1):
        train_df = valid.iloc[train_idx]
        val_df = valid.iloc[val_idx]

        X_tr = train_df[all_features].values.astype(np.float32)
        y_tr = train_df["label"].values.astype(np.float32)
        X_val = val_df[all_features].values.astype(np.float32)
        y_val = val_df["label"].values.astype(np.float32)

        val_pred, auc = train_fold(X_tr, y_tr, X_val, y_val, fold_idx)
        fold_aucs.append(auc)

        pred_series = pd.Series(val_pred, index=val_df.index, name="pred")
        oof_records.append(pred_series)

    # OOF 评估
    print(f"\nCV AUC: mean={np.mean(fold_aucs):.4f}  std={np.std(fold_aucs):.4f}  "
          f"folds={[f'{a:.4f}' for a in fold_aucs]}")

    all_preds = pd.concat(oof_records)
    all_preds = all_preds[~all_preds.index.duplicated(keep="first")]
    pred_matrix = all_preds.unstack(level=1)
    pred_matrix = pred_matrix.reindex(index=close_matrix.index, columns=close_matrix.columns)

    from models.evaluate import evaluate_oof
    metrics, curves = evaluate_oof(pred_matrix, close_matrix, label_days=FWD_DAYS, hold_days=1)
    print(f"\nMLP OOF 评估:")
    print(f"  Rank IC = {metrics['ic_mean']:+.4f}  IC_IR = {metrics['ic_ir']:+.4f}  "
          f"LS 夏普 = {metrics['ls_sharpe']:.3f}")

    # 组合回测
    print(f"\nMLP 组合回测 (hold_days=10, cost=0.3%):")
    port_nn = build_portfolio(pred_matrix, close_matrix, long_only=False, cost=0.003, hold_days=10)
    m_nn = performance_metrics(port_nn["port_ret"])
    print(f"  年化={m_nn['annual']:+.2%}  夏普={m_nn['sharpe']:.3f}  回撤={m_nn['mdd']:+.2%}")

    # 读取 LightGBM 对比
    lgb_path = MODEL_DIR / "walk_forward_summary.csv"
    lgb_sharpe = 8.204
    if lgb_path.exists():
        df = pd.read_csv(lgb_path)
        lgb_row = df[df["method"] == "Purged CV OOF"]
        if len(lgb_row):
            lgb_sharpe = lgb_row.iloc[0]["sharpe"]

    # 图表
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ls_nn = port_nn["cum"].dropna()
    ax.plot(ls_nn.index, ls_nn.values, linewidth=1.2, color="#9467bd", label=f"MLP (SR={m_nn['sharpe']:.2f})")

    from models.portfolio_backtest import load_data as pb_load
    pred_lgb, _, cm, _, _, _ = pb_load()
    port_lgb = build_portfolio(pred_lgb, cm, long_only=False, cost=0.003, hold_days=10)
    ax.plot(port_lgb["cum"].index, port_lgb["cum"].values, linewidth=1.2, color="#2ca02c",
            alpha=0.7, label=f"LightGBM (SR=8.75)")
    ax.axhline(1, color="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_title("累计净值（对数坐标）")
    ax.set_ylabel("净值 (log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    methods = ["LightGBM", "MLP"]
    sharpes = [8.753, m_nn["sharpe"]]
    dds = [32.41, -m_nn["mdd"] * 100]
    colors_sr = ["#2ca02c", "#9467bd"]
    bars = ax.bar(methods, sharpes, color=colors_sr, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("夏普对比")
    for bar, s in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{s:.2f}",
                ha="center", va="bottom", fontsize=10)

    ax = axes[2]
    bars2 = ax.bar(methods, dds, color=colors_sr, edgecolor="white")
    ax.set_title("最大回撤对比 (%)")
    ax.set_ylabel("回撤 %")
    for bar, d_val in zip(bars2, dds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{d_val:.1f}%",
                ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "nn_vs_lgbm_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {fig_path}")

    # 保存结果
    pd.DataFrame([{
        "model": "MLP", "cv_auc_mean": np.mean(fold_aucs), "cv_auc_std": np.std(fold_aucs),
        "ic_ir": metrics["ic_ir"], "ls_sharpe": m_nn["sharpe"],
        "ls_annual": m_nn["annual"], "ls_mdd": m_nn["mdd"],
        "lgb_sharpe_cv": 8.753, "lgb_sharpe_wf": lgb_sharpe,
    }]).to_csv(MODEL_DIR / "nn_vs_lgbm_summary.csv", index=False)

    print("\nMLP 训练完成。")


if __name__ == "__main__":
    main()
