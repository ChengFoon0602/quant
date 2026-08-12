"""
train_cv.py — 月末采样 LGBM Purged CV（方向2 ML 合成）。

方法论关键：
  - X/y 已降采样到月末（见 build_monthly_matrix.py）
  - PurgedTimeSeriesSplit.purge_days 是样本数组的索引位置数 → 此处按【月末】计，
    purge_days=2（相邻月末标签窗口基本不重叠）。绝不能传 22（会 purge 掉 22 个月末）。
  - OOF 评估也在月末截面（evaluate_oof_monthly）：pred vs close(t+22)/close(t+1)-1
    的 Rank IC/IR/t + 五分位累计收益。

用法:
    python strategies/zz500_fundamental_trial/train_cv.py
"""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (
    THIS_DIR, FEATURE_SEL_DIR, PROJECT_ROOT, INDEX, FWD_DAYS, TOP_Q, BOTTOM_Q,
    N_SPLITS, PURGE_DAYS, EARLY_STOP, MARKET_COLS,
)
from purify import month_end_dates, fwd_return

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FEATURE_SEL_DIR))

from models.cv import PurgedTimeSeriesSplit
from models.labels import build_labels, align_X_y, build_sample_weights, get_valid_samples
from models.lgbm_trainer import LGBM_PARAMS
from models.portfolio_backtest import build_portfolio, performance_metrics


def load_monthly_matrix():
    X = pd.read_csv(THIS_DIR / "X_monthly.csv", index_col=[0, 1], parse_dates=[0])
    y = pd.read_csv(THIS_DIR / "y_monthly.csv", index_col=[0, 1], parse_dates=[0]).iloc[:, 0]
    return X, y


def evaluate_oof_monthly(pred_matrix, close_matrix, min_n=30):
    """月末截面 OOF 评估：pred vs 21 日前向收益的 Rank IC / 五分位。"""
    fwd = fwd_return(close_matrix, FWD_DAYS)
    med = pred_matrix.index.intersection(fwd.index)
    p = pred_matrix.loc[med]
    r = fwd.loc[med]

    # Rank IC 序列
    joint = p.notna() & r.notna()
    ics = []
    for d in p.index:
        row_p = p.loc[d]; row_r = r.loc[d]
        m = joint.loc[d]
        if m.sum() < min_n:
            continue
        rp = row_p[m].rank(); rr = row_r[m].rank()
        ics.append(rp.corr(rr))
    ic = pd.Series(ics, index=med)
    m_ic, s_ic = ic.mean(), ic.std(ddof=0)
    ic_ir = m_ic / s_ic if s_ic > 0 else 0.0
    ic_t = ic_ir * np.sqrt(len(ic)) if s_ic > 0 else 0.0

    # 五分位累计收益（等权，不扣成本）
    quintile_cum = {}
    for q in range(5):
        rets = []
        for d in p.index:
            row_p = p.loc[d]; row_r = r.loc[d]
            m = joint.loc[d]
            if m.sum() < min_n:
                continue
            rp = row_p[m].rank()
            lo = np.quantile(rp, q / 5); hi = np.quantile(rp, (q + 1) / 5)
            sel = rp[(rp > lo) & (rp <= hi)] if q < 4 else rp[(rp > lo)]
            if len(sel) == 0:
                continue
            rets.append(row_r[m][sel.index].mean())
        quintile_cum[q] = float(np.prod([1 + x for x in rets]) - 1) if rets else np.nan

    return {
        "ic_mean": m_ic, "ic_ir": ic_ir, "ic_t": ic_t, "ic_n": len(ic),
        "quintile_cum": quintile_cum,
    }


def train_cv(X_long, close_matrix):
    """月末 Purged CV 训练 LGBM → OOF 月末预测矩阵 + 评估 + 特征重要性。"""
    print("=" * 72)
    print(f"模型：LightGBM Purged CV（{N_SPLITS} 折, purge {PURGE_DAYS} 月末）")
    print("=" * 72)
    feat_cols = [c for c in X_long.columns if c not in MARKET_COLS]
    all_feat = feat_cols + [c for c in X_long.columns if c in MARKET_COLS]

    # universe：X_monthly 的股票集合（月末成员）
    universe = pd.Series(True, index=X_long.index).unstack(fill_value=False)
    labels = build_labels(close_matrix, fwd_days=FWD_DAYS, top_q=TOP_Q, bottom_q=BOTTOM_Q,
                          universe=universe)
    aligned = align_X_y(X_long, labels).sort_index(level=0)
    print(f"  样本 {len(aligned):,} | 有效标签 {aligned['label'].notna().sum():,} "
          f"({aligned['label'].notna().mean():.1%}) | 特征 {len(all_feat)}")

    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS, purge_days=PURGE_DAYS)
    oof_records, fold_aucs = [], []
    fi = pd.DataFrame(index=all_feat)
    for k, (tr_idx, va_idx) in enumerate(cv.split(aligned), 1):
        tr, va = aligned.iloc[tr_idx], aligned.iloc[va_idx]
        X_tr, y_tr = get_valid_samples(tr[all_feat], tr["label"])
        X_va, y_va = get_valid_samples(va[all_feat], va["label"])
        if len(y_va) < 50:
            print(f"  Fold {k}: 验证样本过少 ({len(y_va)})，跳过")
            continue
        w_tr = build_sample_weights(y_tr, "balanced")
        dtr = lgb.Dataset(X_tr.values, label=y_tr.values, weight=w_tr)
        dva = lgb.Dataset(X_va.values, label=y_va.values, reference=dtr)
        model = lgb.train(LGBM_PARAMS, dtr, valid_sets=[dva], valid_names=["val"],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                     lgb.log_evaluation(0)])
        auc = model.best_score["val"]["auc"]
        fold_aucs.append(auc)
        pred = model.predict(X_va.values, num_iteration=model.best_iteration)
        oof_records.append(pd.Series(pred, index=X_va.index, name="pred"))
        fi[f"fold_{k}"] = model.feature_importance(importance_type="gain")
        print(f"  Fold {k}: train={len(X_tr):,} val={len(X_va):,} AUC={auc:.4f}")

    if not fold_aucs:
        raise SystemExit("无有效 fold，检查 X_monthly 样本量")
    print(f"  CV AUC mean={np.mean(fold_aucs):.4f} std={np.std(fold_aucs):.4f}")

    all_pred = pd.concat(oof_records)
    all_pred = all_pred[~all_pred.index.duplicated(keep="first")]
    pred_matrix = all_pred.unstack(level=1).reindex(index=labels.index, columns=labels.columns)
    pred_matrix.to_csv(THIS_DIR / "oof_predictions_monthly.csv")

    metrics = evaluate_oof_monthly(pred_matrix, close_matrix)
    fi["mean"] = fi[[f"fold_{k}" for k in range(1, len(fold_aucs) + 1)]].mean(axis=1)
    fi = fi.sort_values("mean", ascending=False)
    fi.to_csv(THIS_DIR / "feature_importance_monthly.csv")

    print(f"  OOF 月末 Rank IC mean={metrics['ic_mean']:+.4f} "
          f"IC_IR={metrics['ic_ir']:+.4f} t={metrics['ic_t']:+.2f} (n={metrics['ic_n']})")
    qc = metrics["quintile_cum"]
    print(f"  五分位累计收益 (0-4): "
          + " ".join(f"Q{i}={qc[i]:+.1%}" for i in range(5)))
    return pred_matrix, metrics, fi, float(np.mean(fold_aucs)), float(np.std(fold_aucs))


def main():
    print("=" * 72)
    print("方向2：月末 LGBM 训练")
    print("=" * 72)

    if not (THIS_DIR / "X_monthly.csv").exists():
        raise SystemExit("缺 X_monthly.csv，先跑 build_monthly_matrix.py")

    print("[1] 加载月末矩阵...")
    X, y = load_monthly_matrix()
    print(f"  X: {X.shape}")

    print("[2] 加载 PIT 面板（close 用于标签/评估）...")
    from build_pit_matrix import load_pit_panel
    close, _, _ = load_pit_panel(INDEX)
    td = close.index.intersection(X.index.get_level_values(0))
    close = close.loc[td]

    pred_matrix, metrics, fi, auc_m, auc_s = train_cv(X, close)
    print("\n完成。")


if __name__ == "__main__":
    main()
