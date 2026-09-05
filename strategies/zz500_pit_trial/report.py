"""
strategies/zz500_pit_trial/report.py — 路线②：PIT 中证 500 全链复现。

假设检验：Alpha191 量价因子在中小盘（中证 500）是否比大盘（沪深 300）更有效？
  沪深 300 PIT 天花板（models/report.md）：OOF Rank IC≈0.051，IC_IR 0.269，
  LS 净成本后 SR=-1.09（做空拖累），只有 Long-Only 超额稳健（SR 1.187，年化 +28.7%）。
  中证 500 流动性更差、散户占比更高、定价更低效 → 量价因子理论上信号更强。
  本脚本在 PIT 中证 500 universe 上重跑同一条链路，与 CSI 300 逐项对照。

方法与假设（全部 PIT，成员掩码内评估，与 CSI 300 完全对齐）：
  - 股票池 = 2010-2025 每月末中证 500 名单并集（1625 只历史成员，含退市股）
  - 每个 (date, symbol) 样本仅当该股当日在指数内才保留（月末快照前向填充）
  - IC / FM / CS_eff / 分位数 全部在"当日指数成员"掩码内计算
  - 前向收益跨越成员变更边界时用真实价格（出指数 ≠ 退市，仍可交易）
  - 摩擦成本、标签定义、CV、purge、hold 网格与 CSI 300 一致（可比性）

流程：
  1. 加载 PIT zz500 面板 → 2. 191 因子分批提纯 → 3. Rank IC 冗余剔除 →
  4. 构建本目录 X/y 矩阵（不覆盖 CSI 300） → 5. LGBM Purged CV + OOF 评估 →
  6. 组合回测 LS/LO/市场 + Bootstrap → 7. Walk-Forward 年度再训练 →
  8. 图表 + 指标 print（供 report.md 引用）

用法:
    cd D:/桌面文件/quant
    python strategies/zz500_pit_trial/report.py          # 全流程
    python strategies/zz500_pit_trial/report.py --rebuild # 强制重建 X/y 矩阵
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent.parent
FEATURE_SEL_DIR = PROJECT_ROOT / "strategies" / "feature_selection"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FEATURE_SEL_DIR))  # build_pit_matrix / purify_v2 / select_features

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from signals.alpha191 import list_factors
from signals.alpha191.calculator import compute_factor_matrix
from signals.alternative.factors import compute_residual_momentum
from build_pit_matrix import load_pit_panel, build_market_features_pit
from purify_v2 import cross_sectional_effective_ratio
from select_features import compute_rank_ic_corr_matrix, remove_redundant
from models.cv import PurgedTimeSeriesSplit
from models.labels import align_X_y, build_labels, build_sample_weights, get_valid_samples
from models.evaluate import evaluate_oof
from models.portfolio_backtest import build_portfolio, performance_metrics, block_bootstrap_sharpe
from models.lgbm_trainer import LGBM_PARAMS

# ── 配置（与 CSI 300 管道对齐，保证可比性）─────────────────
INDEX = "zz500"
DATE_START, DATE_END = "2010-01-01", "2025-12-31"
FIGURES_DIR = THIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# 提纯三维阈值
IC_IR_THRESHOLD = 0.05
FM_T_THRESHOLD = 2.0
CS_EFFECTIVE_THRESHOLD = 0.5
N_GROUPS = 5
BATCH_SIZE = 48           # 因子分批，控制内存（zz500 股票更多）

# 冗余剔除 + 池大小
RANK_IC_CORR_THRESHOLD = 0.8
MUST_KEEP = {"alpha001", "alpha055"}
MAX_POOL_SIZE = 15

# 训练 / 标签 / 组合
FWD_DAYS = 5
TOP_Q = BOTTOM_Q = 0.20
N_SPLITS = 5
PURGE_DAYS = 6
# 成本口径（2026-09 收口）：build_portfolio 默认走方向分离（买 0.026%/卖 0.076%）。
# COST_BPS 保留仅为向后兼容，新调用不传 cost 直接用默认方向分离口径。
COST_BPS = 0.00102        # 双边合计铁律 0.1%
HOLD_GRID = [1, 5, 10, 20]
WF_TEST_YEARS = list(range(2015, 2026))
WF_NUM_BOOST = 79
EARLY_STOP = 50

MARKET_COLS = ["market_vol_20d", "market_turnover_20d"]

# CSI 300 PIT 基准（models/ 已跑结果，供报告对照）
CSI300_REF = {
    "ic_mean": 0.0512, "ic_ir": 0.269, "ic_t": 15.28,
    "ls_sharpe_net": 0.036, "lo_sharpe": 1.187, "lo_annual": 0.2867,
    "wf_sharpe": 0.306, "market_sharpe": 0.249,
}


def section(title: str):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


# ── 向量化 IC / FM（数学等价于 purify_v2 的逐日循环，用矩阵一次算全历史）──
# zz500 面板 1624 宽 × 3886 天，逐日 Python 循环 ×191 因子 ≈ 3 小时；向量化后 <1 分钟。
# 已用 hs300 面板 4 因子对拍 purify_v2.compute_ic_ir/compute_fm，IC_IR/FM_t 误差 <1e-6。

def compute_ic_ir_vec(factor_df: pd.DataFrame, fwd_ret: pd.DataFrame, min_n: int = 10) -> dict:
    """向量化 Rank IC + IC_IR，等价于 purify_v2.compute_ic_ir 的逐日 Spearman。"""
    cd = factor_df.index.intersection(fwd_ret.index)
    cs = factor_df.columns.intersection(fwd_ret.columns)
    f = factor_df.loc[cd, cs]; r = fwd_ret.loc[cd, cs]
    joint = f.notna() & r.notna()
    f = f.where(joint); r = r.where(joint)               # 仅在联合有效集内排名（与循环一致）
    fr = f.rank(axis=1); rr = r.rank(axis=1)
    valid = joint.sum(axis=1) >= min_n
    fr, rr = fr[valid], rr[valid]
    fm = fr.mean(axis=1); rm = rr.mean(axis=1)           # 逐日 Pearson(rank_f, rank_r)
    fc = fr.sub(fm, axis=0); rc = rr.sub(rm, axis=0)
    cov = (fc * rc).sum(axis=1)
    denom = np.sqrt((fc ** 2).sum(axis=1) * (rc ** 2).sum(axis=1))
    ic = (cov / denom.replace(0, np.nan)).dropna()
    if len(ic) == 0:
        return {"mean": 0.0, "std": 0.0, "IR": 0.0, "t": 0.0, "n_days": 0}
    m, s = ic.mean(), ic.std(ddof=0)
    ir = m / s if s > 0 else 0.0
    return {"mean": m, "std": s, "IR": ir,
            "t": ir * np.sqrt(len(ic)) if s > 0 else 0.0, "n_days": len(ic)}


def compute_fm_vec(factor_df: pd.DataFrame, fwd_ret: pd.DataFrame, min_n: int = 10) -> dict:
    """向量化 Fama-MacBeth λ，等价于 purify_v2.compute_fm 的逐日 cov/var 单变量回归。"""
    cd = factor_df.index.intersection(fwd_ret.index)
    cs = factor_df.columns.intersection(fwd_ret.columns)
    f = factor_df.loc[cd, cs]; r = fwd_ret.loc[cd, cs]
    joint = f.notna() & r.notna()
    f = f.where(joint); r = r.where(joint)
    n_day = joint.sum(axis=1)
    valid = n_day >= min_n
    f, r, n_day = f[valid], r[valid], n_day[valid]
    fm = f.mean(axis=1); rm = r.mean(axis=1)
    fc = f.sub(fm, axis=0); rc = r.sub(rm, axis=0)
    # 严格复刻 purify_v2.compute_fm：var=np.var(ddof=0), cov=np.cov[0,1](ddof=1)
    var_x = (fc ** 2).sum(axis=1) / n_day
    cov_xy = (fc * rc).sum(axis=1) / (n_day - 1)
    lam = (cov_xy / var_x.where(var_x >= 1e-12)).dropna()
    if len(lam) == 0:
        return {"λ_annual": 0.0, "t": 0.0, "n_days": 0}
    m, s = lam.mean(), lam.std(ddof=0)
    return {"λ_annual": m * 252, "t": m / s * np.sqrt(len(lam)) if s > 0 else 0.0, "n_days": len(lam)}


def purify_and_select(close_matrix, volume_matrix, member_daily):
    """191 因子分批提纯 → Rank IC 冗余剔除 → 返回 (final_pool, purify_df, factor_tensor_final)。

    factor_tensor_final 只保留最终池的成员掩码后因子矩阵，直接喂给 build_matrix。
    """
    daily_ret = close_matrix.pct_change().fillna(0)
    fwd_ret = daily_ret.shift(-1)
    fwd_ret.iloc[-1] = 0

    section("阶段 1/3：191 因子分批提纯（PIT 中证 500 成员掩码内 IC/FM/CS）")
    all_fids = list_factors()
    print(f"因子 {len(all_fids)}，批大小 {BATCH_SIZE}；阈值 |IC_IR|>{IC_IR_THRESHOLD} "
          f"FM|t|>{FM_T_THRESHOLD} CS_eff>{CS_EFFECTIVE_THRESHOLD}")

    results = []
    keep_tensor: dict[str, pd.DataFrame] = {}  # 候选因子（通过或 must_keep）的掩码后矩阵，避免二次计算
    for b0 in range(0, len(all_fids), BATCH_SIZE):
        batch = all_fids[b0:b0 + BATCH_SIZE]
        print(f"  批次 {b0 // BATCH_SIZE + 1}: {batch[0]}~{batch[-1]}")
        _, tensor = compute_factor_matrix(
            list(close_matrix.columns), batch, start=DATE_START, end=DATE_END, verbose=False,
        )
        for fid in batch:
            fdf = tensor.pop(fid).where(member_daily)
            ic = compute_ic_ir_vec(fdf, fwd_ret)
            fm = compute_fm_vec(fdf, fwd_ret)
            p_ic = abs(ic["IR"]) > IC_IR_THRESHOLD
            p_fm = abs(fm["t"]) > FM_T_THRESHOLD
            # CS_eff（qcut 逐日）较贵：仅对 IC+FM 已通过的因子计算（三维是 AND 门，
            # 未过 IC/FM 的因子最终判定必为 False，CS 不影响结论），未算的记 NaN
            if p_ic and p_fm:
                cs = cross_sectional_effective_ratio(fdf, N_GROUPS)
            else:
                cs = np.nan
            p_cs = bool(cs > CS_EFFECTIVE_THRESHOLD) if not np.isnan(cs) else False
            passed = p_ic and p_fm and p_cs
            results.append({
                "factor": fid, "IC_IR": ic["IR"], "IC_mean": ic["mean"], "IC_t": ic["t"],
                "FM_λ_annual": fm["λ_annual"], "FM_t": fm["t"],
                "cs_effective_ratio": cs, "IC_pass": p_ic, "FM_pass": p_fm,
                "CS_pass": p_cs, "pass": passed,
            })
            if passed or fid in MUST_KEEP:
                keep_tensor[fid] = fdf

    # ── 残差动量（非 Alpha 191 因子，IC_IR=0.20）──
    print("\n  计算残差动量 (win=252, period=20)...")
    try:
        fdf_rm = compute_residual_momentum(close_matrix, window=252, momentum_period=20).where(member_daily)
        ic_rm = compute_ic_ir_vec(fdf_rm, fwd_ret)
        fm_rm = compute_fm_vec(fdf_rm, fwd_ret)
        p_ic_rm = abs(ic_rm["IR"]) > IC_IR_THRESHOLD
        p_fm_rm = abs(fm_rm["t"]) > FM_T_THRESHOLD
        cs_rm = cross_sectional_effective_ratio(fdf_rm, N_GROUPS) if p_ic_rm else np.nan
        p_cs_rm = bool(cs_rm > CS_EFFECTIVE_THRESHOLD) if not np.isnan(cs_rm) else False
        passed_rm = p_ic_rm and p_fm_rm and p_cs_rm
        results.append({
            "factor": "residual_momentum", "IC_IR": ic_rm["IR"], "IC_mean": ic_rm["mean"], "IC_t": ic_rm["t"],
            "FM_λ_annual": fm_rm["λ_annual"], "FM_t": fm_rm["t"],
            "cs_effective_ratio": cs_rm, "IC_pass": p_ic_rm, "FM_pass": p_fm_rm,
            "CS_pass": p_cs_rm, "pass": passed_rm,
        })
        keep_tensor["residual_momentum"] = fdf_rm
        print(f"    residual_momentum: IC={ic_rm['mean']:+.4f} IC_IR={ic_rm['IR']:+.3f} "
              f"FM_t={fm_rm['t']:+.2f} CS={cs_rm:.3f} → {'✓ 通过' if passed_rm else '✗'}")
    except Exception as e:
        print(f"    residual_momentum: 计算失败 — {e}")

    purify_df = pd.DataFrame(results).sort_values("IC_IR", key=abs, ascending=False)
    purify_df.to_csv(THIS_DIR / "purify_results.csv", index=False)
    n_pass = int(purify_df["pass"].sum())
    print(f"  三维通过: {n_pass} 个 | 保存 purify_results.csv")

    section("阶段 2/3：Rank IC 冗余剔除（|corr|>0.8 贪心）")
    ic_ir_map = {r["factor"]: r["IC_IR"] for _, r in purify_df.iterrows()}
    corr_mat = compute_rank_ic_corr_matrix(keep_tensor, fwd_ret)
    corr_mat.to_csv(THIS_DIR / "rank_ic_corr_matrix.csv")
    selected, redundant = remove_redundant(corr_mat, ic_ir_map, RANK_IC_CORR_THRESHOLD, MUST_KEEP)
    print(f"  候选 {len(keep_tensor)} → 冗余剔除后 {len(selected)}（剔 {len(redundant)}）")

    sel_icir = sorted(selected, key=lambda f: abs(ic_ir_map.get(f, 0)), reverse=True)
    final_pool = sel_icir[:MAX_POOL_SIZE]
    for mk in MUST_KEEP:
        if mk not in final_pool:
            final_pool.append(mk)
    print(f"  最终因子池 ({len(final_pool)}): {final_pool}")
    for i, f in enumerate(final_pool, 1):
        r = purify_df[purify_df["factor"] == f].iloc[0]
        print(f"    {i:2d}. {f:10s} IC_IR={r['IC_IR']:+.4f} FM_t={r['FM_t']:+.2f} CS={r['cs_effective_ratio']:.3f}")

    factor_tensor_final = {f: keep_tensor[f] for f in final_pool}
    return final_pool, purify_df, corr_mat, factor_tensor_final, keep_tensor

def build_matrix(close_matrix, volume_matrix, member_daily, final_pool, factor_tensor):
    """构建本目录 PIT 长格式 X/y 矩阵（不覆盖 CSI 300 的 feature_selection/）。"""
    section("阶段 3/3：构建 PIT 特征矩阵（写入本策略目录）")
    frames = [factor_tensor[f].stack().rename(f) for f in final_pool]
    X_factor = pd.concat(frames, axis=1)

    mkt = build_market_features_pit(close_matrix, volume_matrix, member_daily)
    mkt_long = mkt.loc[X_factor.index.get_level_values(0)]
    mkt_long.index = X_factor.index
    X = pd.concat([X_factor, mkt_long], axis=1)

    member_long = member_daily.stack()
    keep = member_long.reindex(X.index).fillna(False).astype(bool)
    n0 = len(X)
    X = X[keep.values]

    fwd = close_matrix.pct_change().shift(-1).fillna(0)
    y = fwd.stack().rename("fwd_return").reindex(X.index)
    X.index.names = y.index.names = ["date", "symbol"]
    X.to_csv(THIS_DIR / "X_matrix.csv")
    y.to_csv(THIS_DIR / "y_matrix.csv")
    print(f"  成员过滤 {n0:,} → {len(X):,} 行 | 股票 {X.index.get_level_values(1).nunique()} 只 | 特征 {list(X.columns)}")
    return X


def load_local_matrix():
    """从本目录读回 X_matrix.csv（字符串读入保留 symbol 格式）。"""
    xp = THIS_DIR / "X_matrix.csv"
    X_raw = pd.read_csv(xp, dtype=str)
    X_raw["date"] = pd.to_datetime(X_raw["date"])
    stock_col = X_raw.columns[1]
    X_long = X_raw.set_index(["date", stock_col]).astype(float)
    X_long.index.names = ["date", "symbol"]
    return X_long

def train_cv(X_long, close_matrix):
    """Purged CV 训练 LGBM → OOF 预测矩阵 + 评估 + 特征重要性。"""
    section("模型：LightGBM Purged CV（5 折, purge 6）")
    feat_cols = [c for c in X_long.columns if c not in MARKET_COLS]
    all_feat = feat_cols + [c for c in X_long.columns if c in MARKET_COLS]

    universe = pd.Series(True, index=X_long.index).unstack(fill_value=False)
    labels = build_labels(close_matrix, fwd_days=FWD_DAYS, top_q=TOP_Q, bottom_q=BOTTOM_Q, universe=universe)
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
        w_tr = build_sample_weights(y_tr, "balanced")
        dtr = lgb.Dataset(X_tr.values, label=y_tr.values, weight=w_tr)
        dva = lgb.Dataset(X_va.values, label=y_va.values, reference=dtr)
        model = lgb.train(LGBM_PARAMS, dtr, valid_sets=[dva], valid_names=["val"],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)])
        auc = model.best_score["val"]["auc"]
        fold_aucs.append(auc)
        pred = model.predict(X_va.values, num_iteration=model.best_iteration)
        oof_records.append(pd.Series(pred, index=X_va.index, name="pred"))
        fi[f"fold_{k}"] = model.feature_importance(importance_type="gain")
        print(f"  Fold {k}: train={len(X_tr):,} val={len(X_va):,} AUC={auc:.4f}")

    print(f"  CV AUC mean={np.mean(fold_aucs):.4f} std={np.std(fold_aucs):.4f}")

    all_pred = pd.concat(oof_records)
    all_pred = all_pred[~all_pred.index.duplicated(keep="first")]
    pred_matrix = all_pred.unstack(level=1).reindex(index=labels.index, columns=labels.columns)
    pred_matrix.to_csv(THIS_DIR / "oof_predictions.csv")

    metrics, curves = evaluate_oof(pred_matrix, close_matrix, label_days=FWD_DAYS, hold_days=1)
    fi["mean"] = fi[[f"fold_{k}" for k in range(1, len(fold_aucs) + 1)]].mean(axis=1)
    fi = fi.sort_values("mean", ascending=False)
    fi.to_csv(THIS_DIR / "feature_importance.csv")

    print(f"  OOF Rank IC mean={metrics['ic_mean']:+.4f} IC_IR={metrics['ic_ir']:+.4f} "
          f"t={metrics['ic_t']:+.2f} (n={metrics['ic_days']})")
    return pred_matrix, metrics, curves, fi, float(np.mean(fold_aucs)), float(np.std(fold_aucs))

def backtest(pred_matrix, close_matrix, X_long):
    """组合回测：hold 网格 + LS/LO/市场基准 + Bootstrap 夏普。"""
    section("组合回测：hold 网格 + LS/LO + 市场基准（双边 0.3% 成本）")

    # 市场基准：当日成员等权日收益（无重叠 tranche，直接截面均值）
    member_mask = pd.Series(True, index=X_long.index).unstack(fill_value=False)
    daily_ret = close_matrix.pct_change()
    mask_al = member_mask.reindex_like(daily_ret).fillna(False).astype(bool)
    market_ret = daily_ret.where(mask_al).mean(axis=1).dropna()
    m_mkt = performance_metrics(market_ret)

    # hold 网格（LS + LO），验证夏普对持有期的稳定性
    grid = []
    for h in HOLD_GRID:
        ls = build_portfolio(pred_matrix, close_matrix, long_only=False, hold_days=h)
        lo = build_portfolio(pred_matrix, close_matrix, long_only=True, hold_days=h)
        m_ls, m_lo = performance_metrics(ls["port_ret"]), performance_metrics(lo["port_ret"])
        grid.append({"hold_days": h, "ls_annual": m_ls["annual"], "ls_sharpe": m_ls["sharpe"],
                     "ls_mdd": m_ls["mdd"], "lo_annual": m_lo["annual"], "lo_sharpe": m_lo["sharpe"],
                     "lo_mdd": m_lo["mdd"]})
        print(f"  hold={h:2d}: LS SR={m_ls['sharpe']:+.3f} ({m_ls['annual']:+.2%}) | "
              f"LO SR={m_lo['sharpe']:+.3f} ({m_lo['annual']:+.2%})")
    grid_df = pd.DataFrame(grid)
    grid_df.to_csv(THIS_DIR / "hold_grid_summary.csv", index=False)
    print(f"  市场等权: SR={m_mkt['sharpe']:+.3f} ({m_mkt['annual']:+.2%})")

    # 主口径 hold=5：Bootstrap 夏普显著性
    ls5 = build_portfolio(pred_matrix, close_matrix, long_only=False, hold_days=5)
    lo5 = build_portfolio(pred_matrix, close_matrix, long_only=True, hold_days=5)
    _, p_ls = block_bootstrap_sharpe(ls5["port_ret"])
    _, p_lo = block_bootstrap_sharpe(lo5["port_ret"])
    m_ls5, m_lo5 = performance_metrics(ls5["port_ret"]), performance_metrics(lo5["port_ret"])
    print(f"  [hold=5] LS SR={m_ls5['sharpe']:+.3f} p={p_ls:.4f} | LO SR={m_lo5['sharpe']:+.3f} p={p_lo:.4f}")

    bt = {"grid_df": grid_df, "ls5": ls5, "lo5": lo5, "market_ret": market_ret,
          "m_ls5": m_ls5, "m_lo5": m_lo5, "m_mkt": m_mkt, "p_ls": p_ls, "p_lo": p_lo}
    return bt


def walk_forward(X_long, close_matrix):
    """年度再训练滚动回测（expanding window），hold=10。"""
    section("Walk-Forward：年度再训练（2015-2025, expanding, hold=10）")
    all_feat = [c for c in X_long.columns]
    universe = pd.Series(True, index=X_long.index).unstack(fill_value=False)
    labels = build_labels(close_matrix, fwd_days=FWD_DAYS, top_q=TOP_Q, bottom_q=BOTTOM_Q, universe=universe)
    valid = align_X_y(X_long, labels).sort_index(level=0).dropna(subset=["label"])

    yearly, ports = [], []
    for year in WF_TEST_YEARS:
        tr_end, te_start, te_end = f"{year-1}-12-31", f"{year}-01-01", f"{year}-12-31"
        d = valid.index.get_level_values(0)
        tr = valid[(d >= "2010-01-01") & (d <= tr_end)]
        te = valid[(d >= te_start) & (d <= te_end)]
        if len(tr) < 1000 or len(te) < 100:
            continue
        w = build_sample_weights(tr["label"], "balanced")
        dtr = lgb.Dataset(tr[all_feat].values, label=tr["label"].values, weight=w)
        model = lgb.train({**LGBM_PARAMS, "verbose": -1}, dtr, num_boost_round=WF_NUM_BOOST)
        pred = pd.Series(model.predict(te[all_feat].values), index=te.index).unstack(level=1)
        td = pred.index.intersection(close_matrix.index)
        ts = pred.columns.intersection(close_matrix.columns)
        pred = pred.loc[td, ts]
        if len(td) <= 12:
            continue
        pf = build_portfolio(pred, close_matrix.loc[td, ts], long_only=False, hold_days=10)
        m = performance_metrics(pf["port_ret"])
        yearly.append({"year": year, "annual": m["annual"], "sharpe": m["sharpe"], "mdd": m["mdd"], "n": m["n"]})
        ports.append(pf["port_ret"])
        print(f"  {year}: 训练={len(tr):>7,} 测试={len(te):>6,} 年化={m['annual']:>+8.2%} SR={m['sharpe']:>+7.3f}")

    yearly_df = pd.DataFrame(yearly)
    yearly_df.to_csv(THIS_DIR / "walk_forward_yearly.csv", index=False)
    all_wf = pd.concat(ports).sort_index() if ports else pd.Series(dtype=float)
    m_wf = performance_metrics(all_wf)
    print(f"  合并 WF: 年化={m_wf['annual']:+.2%} SR={m_wf['sharpe']:+.3f} 回撤={m_wf['mdd']:+.2%}")
    return yearly_df, all_wf, m_wf


def walk_forward_pit_select(close_matrix, volume_matrix, member_daily, keep_tensor):
    """WF with annual factor re-selection — 消除 selection bias。

    与 walk_forward() 的区别：每年用截至当年的数据重做三维提纯+冗余剔除，
    而非用全样本预选的固定因子池。这是路线② 落地的先决条件。
    """
    section("Walk-Forward PIT Select：年度重选因子池 (2015-2025, expanding, hold=10)")

    daily_ret = close_matrix.pct_change().fillna(0)
    fwd_ret_full = daily_ret.shift(-1)
    fwd_ret_full.iloc[-1] = 0

    # 预建 market features（全时段，后续切片）
    mkt = build_market_features_pit(close_matrix, volume_matrix, member_daily)

    yearly, ports, pools_log, preds = [], [], [], []
    for year in WF_TEST_YEARS:
        tr_end = f"{year - 1}-12-31"
        te_start, te_end = f"{year}-01-01", f"{year}-12-31"

        # ── 切片训练窗口 ──
        close_tr = close_matrix[:tr_end]
        fwd_tr = fwd_ret_full[:tr_end]
        member_tr = member_daily[:tr_end]
        mkt_tr = mkt[:tr_end]

        # ── 1. 重算 IC_IR + FM（只用训练窗口数据）──
        results = []
        for fid, fdf_full in keep_tensor.items():
            fdf = fdf_full.loc[:tr_end].where(member_tr)
            ic = compute_ic_ir_vec(fdf, fwd_tr)
            fm = compute_fm_vec(fdf, fwd_tr)
            p_ic = abs(ic["IR"]) > IC_IR_THRESHOLD
            p_fm = abs(fm["t"]) > FM_T_THRESHOLD
            cs = cross_sectional_effective_ratio(fdf, N_GROUPS) if (p_ic and p_fm) else np.nan
            p_cs = bool(cs > CS_EFFECTIVE_THRESHOLD) if not np.isnan(cs) else False
            results.append({
                "factor": fid, "IC_IR": ic["IR"], "IC_mean": ic["mean"], "IC_t": ic["t"],
                "FM_t": fm["t"], "pass": p_ic and p_fm and p_cs,
            })

        purify_yr = pd.DataFrame(results)
        passing = purify_yr[purify_yr["pass"]]
        print(f"\n  {year}: {len(passing)}/{len(keep_tensor)} 通过三维 "
              f"(top3 IC_IR: {', '.join(f'{r['factor']}={r['IC_IR']:+.3f}' for _, r in passing.head(3).iterrows())})")

        if len(passing) < 3:
            print(f"    ⚠ 通过不足，跳过")
            continue

        # ── 2. 冗余剔除（切片窗口上的 Rank IC 相关性）──
        keep_tr = {}
        for _, r in passing.iterrows():
            fid = r["factor"]
            keep_tr[fid] = keep_tensor[fid].loc[:tr_end].where(member_tr)
        for mk in MUST_KEEP:
            if mk in keep_tensor and mk not in keep_tr:
                keep_tr[mk] = keep_tensor[mk].loc[:tr_end].where(member_tr)

        ic_map_yr = {r["factor"]: r["IC_IR"] for _, r in passing.iterrows()}
        corr_yr = compute_rank_ic_corr_matrix(keep_tr, fwd_tr)
        sel, _ = remove_redundant(corr_yr, ic_map_yr, RANK_IC_CORR_THRESHOLD, MUST_KEEP)
        sel_sorted = sorted(sel, key=lambda f: abs(ic_map_yr.get(f, 0)), reverse=True)
        pool_yr = sel_sorted[:MAX_POOL_SIZE]
        for mk in MUST_KEEP:
            if mk in keep_tensor and mk not in pool_yr:
                pool_yr.append(mk)
        pools_log.append({"year": year, "pool": pool_yr, "n_pass": len(passing)})

        # ── 3. 构建训练集 X ──
        syms_tr = close_tr.columns.tolist()
        frames_tr = [keep_tensor[f].loc[:tr_end].stack().rename(f) for f in pool_yr]
        X_factor_tr = pd.concat(frames_tr, axis=1)
        mkt_align = mkt_tr.loc[X_factor_tr.index.get_level_values(0)]
        mkt_align.index = X_factor_tr.index
        X_tr = pd.concat([X_factor_tr, mkt_align], axis=1)
        # 成员过滤
        member_long = member_tr.stack()
        keep_mask = member_long.reindex(X_tr.index).fillna(False).astype(bool)
        X_tr = X_tr[keep_mask.values]
        y_tr = fwd_tr.stack().rename("fwd_return").reindex(X_tr.index)
        X_tr.index.names = y_tr.index.names = ["date", "symbol"]

        # ── 4. 标签 + 训练 ──
        universe_tr = pd.Series(True, index=X_tr.index).unstack(fill_value=False)
        labels_tr = build_labels(close_tr, fwd_days=FWD_DAYS, top_q=TOP_Q, bottom_q=BOTTOM_Q,
                                 universe=universe_tr)
        valid_tr = align_X_y(X_tr, labels_tr).sort_index(level=0).dropna(subset=["label"])
        d_tr = valid_tr.index.get_level_values(0)
        tr_data = valid_tr[(d_tr >= "2010-01-01") & (d_tr <= tr_end)]
        if len(tr_data) < 1000:
            print(f"    训练样本不足 ({len(tr_data)}), 跳过")
            continue

        feat_cols = [c for c in X_tr.columns if c not in MARKET_COLS]
        all_feat = feat_cols + [c for c in X_tr.columns if c in MARKET_COLS]
        w = build_sample_weights(tr_data["label"], "balanced")
        dtr = lgb.Dataset(tr_data[all_feat].values, label=tr_data["label"].values, weight=w)
        model = lgb.train({**LGBM_PARAMS, "verbose": -1}, dtr, num_boost_round=WF_NUM_BOOST)

        # ── 5. 预测测试年 ──
        te_dates = close_matrix.loc[te_start:te_end].index
        if len(te_dates) <= 20:
            continue

        # 测试集 X（用全时段因子值切片）
        frames_te = [keep_tensor[f].loc[te_start:te_end].stack().rename(f) for f in pool_yr]
        X_factor_te = pd.concat(frames_te, axis=1)
        mkt_te = mkt.loc[te_start:te_end]
        mkt_te_aligned = mkt_te.loc[X_factor_te.index.get_level_values(0)]
        mkt_te_aligned.index = X_factor_te.index
        X_te = pd.concat([X_factor_te, mkt_te_aligned], axis=1)
        # 测试期成员过滤
        member_te = member_daily.loc[te_start:te_end]
        member_te_long = member_te.stack()
        keep_te = member_te_long.reindex(X_te.index).fillna(False).astype(bool)
        X_te = X_te[keep_te.values]
        X_te.index.names = ["date", "symbol"]
        if len(X_te) < 100:
            continue

        pred = pd.Series(
            model.predict(X_te[all_feat].values), index=X_te.index
        ).unstack(level=1)

        td = pred.index.intersection(close_matrix.index)
        ts = pred.columns.intersection(close_matrix.columns)
        pred = pred.loc[td, ts]

        pf = build_portfolio(pred, close_matrix.loc[td, ts], long_only=False, hold_days=10)
        m = performance_metrics(pf["port_ret"])
        yearly.append({"year": year, "annual": m["annual"], "sharpe": m["sharpe"],
                       "mdd": m["mdd"], "n": m["n"], "pool_size": len(pool_yr)})
        ports.append(pf["port_ret"])
        preds.append(pred)
        print(f"    → 年化={m['annual']:>+8.2%} SR={m['sharpe']:>+7.3f} "
              f"回撤={m['mdd']:>+8.2%} 池={pool_yr}")

    yearly_df = pd.DataFrame(yearly)
    yearly_df.to_csv(THIS_DIR / "walk_forward_yearly_pit_select.csv", index=False)
    # 保存逐年因子池
    pd.DataFrame(pools_log).to_csv(THIS_DIR / "walk_forward_pools.csv", index=False)

    all_wf = pd.concat(ports).sort_index() if ports else pd.Series(dtype=float)
    m_wf = performance_metrics(all_wf)
    print(f"\n  合并 WF (PIT Select): 年化={m_wf['annual']:+.2%} "
          f"SR={m_wf['sharpe']:+.3f} 回撤={m_wf['mdd']:+.2%}")
    print(f"  对照: 固定池 WF SR=1.438（含 selection bias）")

    # ── 拼接全部年份预测 → 真正的 OOF 指标（消除 selection bias）──
    if preds:
        pred_full = pd.concat(preds).sort_index()
        pred_full.to_csv(THIS_DIR / "oof_predictions_pit_select.csv")
        print(f"\n  真实 OOF 预测: {pred_full.shape[0]} 天 × {pred_full.shape[1]} 股")

        # Rank IC（与 train_cv 的 evaluate_oof 对齐）
        daily_ret = close_matrix.pct_change().fillna(0)
        fwd_5d = daily_ret.rolling(5).sum().shift(-5)
        cd = pred_full.index.intersection(fwd_5d.index)
        cs = pred_full.columns.intersection(fwd_5d.columns)
        p = pred_full.loc[cd, cs]; r = fwd_5d.loc[cd, cs]
        joint = p.notna() & r.notna()
        pr = p.where(joint).rank(axis=1); rr = r.where(joint).rank(axis=1)
        valid = joint.sum(axis=1) >= 10
        pr, rr = pr[valid], rr[valid]
        pm = pr.mean(axis=1); rm = rr.mean(axis=1)
        pc = pr.sub(pm, axis=0); rc = rr.sub(rm, axis=0)
        cov = (pc * rc).sum(axis=1)
        denom = np.sqrt((pc ** 2).sum(axis=1) * (rc ** 2).sum(axis=1))
        ic_series = (cov / denom.replace(0, np.nan)).dropna()
        oof_ic = ic_series.mean()
        oof_ic_ir = oof_ic / ic_series.std(ddof=0) if ic_series.std() > 0 else 0
        oof_ic_t = oof_ic_ir * np.sqrt(len(ic_series))
        print(f"  真实 OOF Rank IC = {oof_ic:+.4f}  IC_IR = {oof_ic_ir:+.3f}  t = {oof_ic_t:+.1f}")
        print(f"  对照: 固定池 OOF IC = +0.095 (含 selection bias)")

    return yearly_df, all_wf, m_wf


def plot_screening(purify_df, corr_mat, final_pool, metrics, curves, fi):
    """图 1：因子提纯 + OOF 模型表现（2×3 子图）。"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    ic_ir_map = {r["factor"]: r["IC_IR"] for _, r in purify_df.iterrows()}

    ax = axes[0, 0]  # IC_IR 分布
    ax.hist(purify_df["IC_IR"].dropna(), bins=40, edgecolor="white", alpha=0.8)
    ax.axvline(IC_IR_THRESHOLD, color="green", ls="--"); ax.axvline(-IC_IR_THRESHOLD, color="green", ls="--")
    ax.set_title("191 因子 IC_IR 分布 (PIT zz500)"); ax.set_xlabel("IC_IR")

    ax = axes[0, 1]  # 最终池 IC_IR
    irs = [ic_ir_map.get(f, 0) for f in final_pool]
    ax.bar(range(len(final_pool)), irs, color=["#2ca02c" if v >= 0 else "#d62728" for v in irs], edgecolor="white")
    ax.set_xticks(range(len(final_pool))); ax.set_xticklabels(final_pool, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.5); ax.set_title(f"最终因子池 IC_IR ({len(final_pool)} 个)")

    ax = axes[0, 2]  # 相关性热力图
    im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_mat.columns))); ax.set_yticks(range(len(corr_mat.columns)))
    ax.set_xticklabels(corr_mat.columns, rotation=90, fontsize=5); ax.set_yticklabels(corr_mat.columns, fontsize=5)
    ax.set_title("候选因子 Rank IC 相关性"); plt.colorbar(im, ax=ax, shrink=0.7)

    ax = axes[1, 0]  # 累计 Rank IC
    cum_ic = curves["rank_ic"].cumsum()
    ax.plot(cum_ic.index, cum_ic.values, lw=1.0); ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"OOF 累计 Rank IC (IC_IR={metrics['ic_ir']:+.3f})"); ax.grid(alpha=0.3)

    ax = axes[1, 1]  # 五分位累计
    for c in curves["quintile_ret"].columns:
        ax.plot((1 + curves["quintile_ret"][c].fillna(0)).cumprod(), label=c, lw=1.1)
    ax.set_title("OOF 五分位累计收益"); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)

    ax = axes[1, 2]  # 特征重要性
    top = fi.head(15)
    ax.barh(range(len(top)), top["mean"].values, color="steelblue")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=8); ax.invert_yaxis()
    ax.set_title("特征重要性 (gain, top15)")

    plt.tight_layout()
    p = FIGURES_DIR / "01_screening_and_oof.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  图1: {p}")


def plot_backtest(bt, wf_yearly, wf_all, m_wf):
    """图 2：组合回测 + Walk-Forward（2×3 子图）。"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    ax = axes[0, 0]  # 净值：LS/LO/市场 (hold=5)
    ax.plot(bt["ls5"]["cum"], label=f"LS (SR={bt['m_ls5']['sharpe']:.2f})", color="#1f77b4")
    ax.plot(bt["lo5"]["cum"], label=f"LO (SR={bt['m_lo5']['sharpe']:.2f})", color="#2ca02c")
    mkt_cum = (1 + bt["market_ret"].fillna(0)).cumprod()
    ax.plot(mkt_cum, label=f"市场 (SR={bt['m_mkt']['sharpe']:.2f})", color="gray", alpha=0.7)
    ax.axhline(1, color="black", lw=0.5); ax.set_title("累计净值 hold=5（扣 0.3% 成本）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]  # hold 网格夏普
    g = bt["grid_df"]
    ax.plot(g["hold_days"], g["ls_sharpe"], "o-", label="LS", color="#1f77b4")
    ax.plot(g["hold_days"], g["lo_sharpe"], "s-", label="LO", color="#2ca02c")
    ax.axhline(0, color="black", lw=0.5); ax.set_xlabel("hold_days"); ax.set_ylabel("Sharpe")
    ax.set_title("夏普 vs 持有期（稳定性）"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 2]  # LO 月度收益热力图
    monthly = bt["lo5"]["port_ret"].dropna().resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly.index = pd.MultiIndex.from_arrays([monthly.index.year, monthly.index.month])
    piv = monthly.unstack()
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_title("Long-Only 月度收益"); plt.colorbar(im, ax=ax, shrink=0.7)

    ax = axes[1, 0]  # WF 逐年夏普
    if len(wf_yearly):
        cols = ["#2ca02c" if s >= 0 else "#d62728" for s in wf_yearly["sharpe"]]
        ax.bar(range(len(wf_yearly)), wf_yearly["sharpe"], color=cols, edgecolor="white")
        ax.set_xticks(range(len(wf_yearly))); ax.set_xticklabels(wf_yearly["year"], fontsize=8)
    ax.axhline(0, color="black", lw=0.5); ax.set_title(f"Walk-Forward 逐年夏普 (合并 SR={m_wf['sharpe']:.2f})")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]  # WF 累计净值
    if len(wf_all):
        ax.plot((1 + wf_all).cumprod(), color="#2ca02c", lw=1.3)
    ax.axhline(1, color="black", lw=0.5); ax.set_title("Walk-Forward 累计净值 (hold=10)"); ax.grid(alpha=0.3)

    ax = axes[1, 2]  # 与 CSI300 对照表
    ax.axis("off")
    rows = [
        ["指标", "zz500", "CSI300"],
        ["OOF Rank IC", f"{bt['ic_mean']:.4f}", f"{CSI300_REF['ic_mean']:.4f}"],
        ["OOF IC_IR", f"{bt['ic_ir']:.3f}", f"{CSI300_REF['ic_ir']:.3f}"],
        ["LO 夏普(h5)", f"{bt['m_lo5']['sharpe']:.3f}", f"{CSI300_REF['lo_sharpe']:.3f}"],
        ["LO 年化(h5)", f"{bt['m_lo5']['annual']:.2%}", f"{CSI300_REF['lo_annual']:.2%}"],
        ["LS 夏普(h5)", f"{bt['m_ls5']['sharpe']:.3f}", f"{CSI300_REF['ls_sharpe_net']:.3f}"],
        ["WF 夏普", f"{m_wf['sharpe']:.3f}", f"{CSI300_REF['wf_sharpe']:.3f}"],
        ["市场夏普", f"{bt['m_mkt']['sharpe']:.3f}", f"{CSI300_REF['market_sharpe']:.3f}"],
    ]
    t = ax.table(cellText=rows, cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.8)
    ax.set_title("zz500 vs CSI300 PIT 对照")

    plt.tight_layout()
    p = FIGURES_DIR / "02_backtest_and_wf.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  图2: {p}")

def main():
    rebuild = "--rebuild" in sys.argv
    section("路线②：PIT 中证 500 全链复现")

    x_exists = (THIS_DIR / "X_matrix.csv").exists()
    if x_exists and not rebuild:
        print(f"检测到已构建 X_matrix.csv，跳过提纯+构建（--rebuild 强制重建）")
        X_long = load_local_matrix()
        # close_matrix 只取 X_matrix 出现的股票
        from data.fetcher import load_daily
        syms = sorted(X_long.index.get_level_values(1).unique())
        cd = {}
        for s in syms:
            df = load_daily(s)
            if df is not None and len(df) >= 100:
                ss = df.loc[(df.index >= DATE_START) & (df.index <= DATE_END), "close"]
                if len(ss) >= 100:
                    cd[s] = ss
        close_matrix = pd.DataFrame(cd).sort_index()
        purify_df = pd.read_csv(THIS_DIR / "purify_results.csv")
        corr_mat = pd.read_csv(THIS_DIR / "rank_ic_corr_matrix.csv", index_col=0)
        final_pool = [c for c in X_long.columns if c not in MARKET_COLS]
    else:
        section("加载 PIT 中证 500 面板")
        close_matrix, volume_matrix, member_daily = load_pit_panel(INDEX)
        print(f"  股票池 {close_matrix.shape[1]} 只历史成员 | "
              f"日均成员 {(member_daily & close_matrix.notna()).sum(axis=1).mean():.1f} 只 | "
              f"{close_matrix.shape[0]} 交易日")
        final_pool, purify_df, corr_mat, factor_tensor, keep_tensor = purify_and_select(
            close_matrix, volume_matrix, member_daily)
        X_long = build_matrix(close_matrix, volume_matrix, member_daily, final_pool, factor_tensor)

    # ── 训练 + 评估 ──
    pred_matrix, metrics, curves, fi, auc_mean, auc_std = train_cv(X_long, close_matrix)

    # ── 回测 ──
    bt = backtest(pred_matrix, close_matrix, X_long)
    bt["ic_mean"] = metrics["ic_mean"]; bt["ic_ir"] = metrics["ic_ir"]

    # ── Walk-Forward（固定池，含 selection bias）──
    wf_yearly, wf_all, m_wf = walk_forward(X_long, close_matrix)

    # ── 保存 keep_tensor（供后续分阶段使用）──
    if not x_exists or rebuild:
        import pickle, gzip
        kt_path = THIS_DIR / "keep_tensor.pkl.gz"
        print(f"  保存 keep_tensor ({len(keep_tensor)} 因子) → {kt_path} ...")
        with gzip.open(kt_path, "wb") as f:
            pickle.dump(keep_tensor, f)

    # ── Walk-Forward PIT Select（年度重选池，消除 selection bias）──
    wf_pit_yearly, wf_pit_all, m_wf_pit = None, None, {"sharpe": float("nan"), "annual": float("nan"), "mdd": float("nan")}
    if not x_exists or rebuild:
        # 只在完整重跑时可用（需要 keep_tensor）
        wf_pit_yearly, wf_pit_all, m_wf_pit = walk_forward_pit_select(
            close_matrix, volume_matrix, member_daily, keep_tensor)

    # ── 图表 ──
    section("图表")
    plot_screening(purify_df, corr_mat, final_pool, metrics, curves, fi)
    plot_backtest(bt, wf_yearly, wf_all, m_wf)

    # ── 汇总 JSON ──
    summary = {
        "index": INDEX, "n_stocks": int(X_long.index.get_level_values(1).nunique()),
        "n_final_pool": len(final_pool), "final_pool": final_pool,
        "cv_auc_mean": auc_mean, "cv_auc_std": auc_std,
        "oof_ic_mean": float(metrics["ic_mean"]), "oof_ic_ir": float(metrics["ic_ir"]),
        "oof_ic_t": float(metrics["ic_t"]), "oof_ic_days": int(metrics["ic_days"]),
        "ls5_sharpe": float(bt["m_ls5"]["sharpe"]), "ls5_annual": float(bt["m_ls5"]["annual"]),
        "ls5_boot_p": float(bt["p_ls"]),
        "lo5_sharpe": float(bt["m_lo5"]["sharpe"]), "lo5_annual": float(bt["m_lo5"]["annual"]),
        "lo5_boot_p": float(bt["p_lo"]),
        "market_sharpe": float(bt["m_mkt"]["sharpe"]), "market_annual": float(bt["m_mkt"]["annual"]),
        "wf_sharpe": float(m_wf["sharpe"]), "wf_annual": float(m_wf["annual"]),
        "wf_pit_select_sharpe": float(m_wf_pit["sharpe"]), "wf_pit_select_annual": float(m_wf_pit["annual"]),
        "csi300_ref": CSI300_REF,
    }
    pd.Series(summary).to_json(THIS_DIR / "summary.json", force_ascii=False, indent=2)

    section("完成 — 关键结论")
    print(f"  zz500 OOF Rank IC = {metrics['ic_mean']:+.4f} (CSI300 {CSI300_REF['ic_mean']:+.4f})")
    print(f"  zz500 LO 夏普(h5) = {bt['m_lo5']['sharpe']:+.3f} (CSI300 {CSI300_REF['lo_sharpe']:+.3f})")
    print(f"  zz500 LS 夏普(h5) = {bt['m_ls5']['sharpe']:+.3f} (CSI300 {CSI300_REF['ls_sharpe_net']:+.3f})")
    print(f"  zz500 WF 夏普     = {m_wf['sharpe']:+.3f} (CSI300 {CSI300_REF['wf_sharpe']:+.3f})")
    if not np.isnan(m_wf_pit["sharpe"]):
        delta = m_wf["sharpe"] - m_wf_pit["sharpe"]
        print(f"  zz500 WF PIT Sel  = {m_wf_pit['sharpe']:+.3f} (消除 selection bias, Δ={delta:+.3f})")
    print(f"  汇总: summary.json")


if __name__ == "__main__":
    main()

