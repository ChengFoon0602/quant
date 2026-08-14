"""
walk_forward_pit_select.py — 年度重选因子池 + 重训（方向2 selection bias 消除）。

价量链路的教训：全样本选池让 OOF 夏普 1.906 → 去 bias 后 0.25（~87% 是选择偏差）。
方向2 从第一版就走对：每年用截至当年的数据【重新月末提纯 + 冗余剔除 + 重训】，
预测次年。同时跑一个【固定池】年度重训对照（pool 用全样本提纯结果），
两者 OOF 差异 = selection bias 的定量幅度。

月调仓语义：测试年预测在月末截面，前向填充到日频 + build_portfolio(hold=1)。

用法:
    python strategies/zz500_fundamental_trial/walk_forward_pit_select.py
"""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (
    THIS_DIR, FEATURE_SEL_DIR, PROJECT_ROOT, INDEX,
    FWD_DAYS, TOP_Q, BOTTOM_Q, WF_TEST_YEARS, WF_NUM_BOOST, EARLY_STOP,
    IC_IR_THRESHOLD, IC_T_THRESHOLD, FM_T_THRESHOLD,
    CS_EFFECTIVE_THRESHOLD, N_GROUPS, RANK_IC_CORR_THRESHOLD, MAX_POOL_SIZE,
    MUST_KEEP, MARKET_COLS, COST_BPS,
)
from purify import month_end_dates, fwd_return, _ic_ir_vec, _fm_vec

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FEATURE_SEL_DIR))

from build_pit_matrix import load_pit_panel, build_market_features_pit
from purify_v2 import cross_sectional_effective_ratio
from select_features import compute_rank_ic_corr_matrix, remove_redundant
from models.labels import build_labels, align_X_y, build_sample_weights, get_valid_samples
from models.lgbm_trainer import LGBM_PARAMS
from models.portfolio_backtest import build_portfolio, performance_metrics


def section(title: str):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def _build_X(close_tr, volume_matrix, member_daily, factor_tensor, pool, tr_end, med):
    """构建月末长表 X（截至 tr_end，月末过滤 + 成员过滤）。"""
    frames = [factor_tensor[f].loc[:tr_end].stack().rename(f) for f in pool]
    X_factor = pd.concat(frames, axis=1)
    mkt = build_market_features_pit(close_tr, volume_matrix.loc[:tr_end], member_daily.loc[:tr_end])
    mkt_align = mkt.loc[X_factor.index.get_level_values(0)]
    mkt_align.index = X_factor.index
    X = pd.concat([X_factor, mkt_align], axis=1)

    med_set = set(med)
    keep_med = X.index.get_level_values(0).isin(med_set)
    X = X[keep_med]
    member_long = member_daily.loc[:tr_end].stack()
    keep_mask = member_long.reindex(X.index).fillna(False).astype(bool)
    X = X[keep_mask.values]
    X.index.names = ["date", "symbol"]
    return X


def _predict_year(model, factor_tensor, mkt_full, member_daily, pool,
                  te_start, te_end, close_matrix):
    """预测测试年（月末截面预测矩阵）。"""
    te_dates = close_matrix.loc[te_start:te_end].index
    if len(te_dates) < 20:
        return None
    med_te = month_end_dates(te_dates)
    frames = [factor_tensor[f].loc[te_start:te_end].stack().rename(f) for f in pool]
    X_factor_te = pd.concat(frames, axis=1)
    mkt_te = mkt_full.loc[te_start:te_end]
    mkt_align = mkt_te.loc[X_factor_te.index.get_level_values(0)]
    mkt_align.index = X_factor_te.index
    X_te = pd.concat([X_factor_te, mkt_align], axis=1)
    med_set = set(med_te)
    keep_med = X_te.index.get_level_values(0).isin(med_set)
    X_te = X_te[keep_med]
    member_long = member_daily.loc[te_start:te_end].stack()
    keep_mask = member_long.reindex(X_te.index).fillna(False).astype(bool)
    X_te = X_te[keep_mask.values]
    if len(X_te) < 30:
        return None
    pred = pd.Series(model.predict(X_te.values), index=X_te.index).unstack(level=1)
    return pred


def _monthly_backtest(pred, close_matrix):
    """月末预测 → 日频 ffill + hold=1 月调仓回测。"""
    if pred is None or pred.empty:
        return None, None
    pred_daily = pred.reindex(close_matrix.index).ffill()
    td = pred_daily.index.intersection(close_matrix.index)
    ts = pred_daily.columns.intersection(close_matrix.columns)
    pred_daily = pred_daily.loc[td, ts]
    if len(td) < 20:
        return None, None
    pf = build_portfolio(pred_daily, close_matrix.loc[td, ts],
                         long_only=False, cost=COST_BPS, hold_days=1)
    return pf, pred_daily


def walk_forward_annual(close_matrix, volume_matrix, member_daily, factor_tensor,
                        pool_for_year, fixed_pool=None, label="WF"):
    """年度 expanding 重训（pool 由 pool_for_year(year) 决定，None 时用 fixed_pool）。"""
    fwd = fwd_return(close_matrix, FWD_DAYS)
    mkt_full = build_market_features_pit(close_matrix, volume_matrix, member_daily)

    yearly, ports, preds = [], [], []
    for year in WF_TEST_YEARS:
        tr_end = f"{year - 1}-12-31"
        te_start, te_end = f"{year}-01-01", f"{year}-12-31"
        close_tr = close_matrix[:tr_end]
        member_tr = member_daily[:tr_end]

        pool_yr = pool_for_year(year) if pool_for_year is not None else fixed_pool
        if not pool_yr:
            continue

        med_tr = month_end_dates(close_tr.index)
        fwd_tr = fwd.loc[:tr_end]

        # 训练 X（月末）
        X_tr = _build_X(close_tr, volume_matrix, member_daily, factor_tensor,
                        pool_yr, tr_end, med_tr)
        if len(X_tr) < 300:
            print(f"  {year}: 训练样本不足 ({len(X_tr)})，跳过")
            continue
        universe_tr = pd.Series(True, index=X_tr.index).unstack(fill_value=False)
        labels_tr = build_labels(close_tr, fwd_days=FWD_DAYS, top_q=TOP_Q, bottom_q=BOTTOM_Q,
                                 universe=universe_tr)
        valid_tr = align_X_y(X_tr, labels_tr).sort_index(level=0).dropna(subset=["label"])
        if len(valid_tr) < 300:
            continue
        feat = [c for c in X_tr.columns]
        w = build_sample_weights(valid_tr["label"], "balanced")
        dtr = lgb.Dataset(valid_tr[feat].values, label=valid_tr["label"].values, weight=w)
        model = lgb.train({**LGBM_PARAMS, "verbose": -1}, dtr, num_boost_round=WF_NUM_BOOST)

        # 预测测试年（月末）
        pred = _predict_year(model, factor_tensor, mkt_full, member_daily, pool_yr,
                             te_start, te_end, close_matrix)
        if pred is None:
            continue
        pf, pred_daily = _monthly_backtest(pred, close_matrix)
        if pf is None:
            continue
        m = performance_metrics(pf["port_ret"])
        yearly.append({"year": year, "annual": m["annual"], "sharpe": m["sharpe"],
                       "mdd": m["mdd"], "n": m["n"], "pool_size": len(pool_yr)})
        ports.append(pf["port_ret"])
        preds.append(pred)
        print(f"  {year}: 池={len(pool_yr)} 训练={len(valid_tr):,} "
              f"年化={m['annual']:+.2%} SR={m['sharpe']:+.3f}")

    yearly_df = pd.DataFrame(yearly)
    yearly_df.to_csv(THIS_DIR / f"walk_forward_yearly_{label}.csv", index=False)
    all_wf = pd.concat(ports).sort_index() if ports else pd.Series(dtype=float)
    m_wf = performance_metrics(all_wf) if len(all_wf) else {}
    pred_full = pd.concat(preds).sort_index() if preds else pd.DataFrame()
    return yearly_df, all_wf, m_wf, pred_full


def reselect_pool(close_matrix, member_daily, factor_tensor, fwd, year):
    """年度重选池：截至 year-1 的月末四维提纯 + 冗余剔除。"""
    tr_end = f"{year - 1}-12-31"
    close_tr = close_matrix[:tr_end]
    member_tr = member_daily[:tr_end]
    med_tr = month_end_dates(close_tr.index)
    fwd_tr = fwd.loc[:tr_end]

    results = []
    for field, fdf_full in factor_tensor.items():
        fdf = fdf_full.loc[:tr_end].where(member_tr)
        ic = _ic_ir_vec(fdf.loc[med_tr], fwd_tr.loc[med_tr])
        fm = _fm_vec(fdf.loc[med_tr], fwd_tr.loc[med_tr])
        p_ic = abs(ic["IR"]) > IC_IR_THRESHOLD and abs(ic["t"]) > IC_T_THRESHOLD
        p_fm = abs(fm["t"]) > FM_T_THRESHOLD
        cs = cross_sectional_effective_ratio(fdf.loc[med_tr], N_GROUPS) if (p_ic and p_fm) else np.nan
        p_cs = bool(cs > CS_EFFECTIVE_THRESHOLD) if not np.isnan(cs) else False
        results.append({"factor": field, "IC_IR": ic["IR"], "IC_t": ic["t"],
                        "FM_t": fm["t"], "cs": cs, "pass": p_ic and p_fm and p_cs})
    purify_yr = pd.DataFrame(results)
    passing = purify_yr[purify_yr["pass"]]
    if len(passing) < 3:
        return []
    keep_tr = {f: factor_tensor[f].loc[:tr_end].where(member_tr)
               for f in passing["factor"]}
    ic_map = {r["factor"]: r["IC_IR"] for _, r in passing.iterrows()}
    corr_yr = compute_rank_ic_corr_matrix(keep_tr, fwd_tr.loc[med_tr])
    sel, _ = remove_redundant(corr_yr, ic_map, RANK_IC_CORR_THRESHOLD, MUST_KEEP)
    sel_sorted = sorted(sel, key=lambda f: abs(ic_map.get(f, 0)), reverse=True)
    pool = sel_sorted[:MAX_POOL_SIZE]
    print(f"    {year} 重选池 ({len(pool)}): {pool}")
    return pool


def main():
    print("=" * 72)
    print("方向2：Walk-Forward PIT Select（年度重选池 + 重训）")
    print("=" * 72)

    print("[1] 加载数据 + 全 25 因子张量...")
    close, volume, member = load_pit_panel(INDEX)
    from signals.fundamental.factors import compute_factor_tensor, FACTOR_SPECS
    factor_tensor = compute_factor_tensor(close)
    factor_tensor = {f: df.where(member) for f, df in factor_tensor.items()}
    print(f"  因子张量: {len(factor_tensor)} 个")

    # 全样本提纯（固定池对照）
    print("\n[2] 全样本提纯（固定池，含 selection bias）...")
    from purify import purify_and_select
    final_pool, purify_df, _, _, _ = purify_and_select(close, volume, member)
    if not final_pool:
        # 四维 0 通过：退化 IC_t top-10 诊断池（含 selection bias，仅上限）。
        # PIT-Select 每年重选池仍可能选不出 → 那正是「基本面无 alpha」的全否定实证。
        print("[WARN] 全样本提纯无通过因子，退化 IC_t top-10 固定池对照")
        pur = pd.read_csv(THIS_DIR / "purify_results_monthly.csv")
        final_pool = pur.sort_values("IC_t", key=abs, ascending=False)["factor"].head(10).tolist()

    print("\n[3] 固定池 WF（对照，pool 全样本选定）...")
    fy, fwf, m_fwf, fpred = walk_forward_annual(
        close, volume, member, factor_tensor,
        pool_for_year=None, fixed_pool=final_pool, label="fixed_pool")
    if not fwf.empty:
        print(f"  固定池 WF 合并: 年化={m_fwf['annual']:+.2%} SR={m_fwf['sharpe']:+.3f}")
    else:
        m_fwf = {}

    print("\n[4] PIT-Select WF（年度重选池，消除 selection bias）...")
    fwd = fwd_return(close, FWD_DAYS)
    pools_log = []
    yearly_out, ports_out, preds_out = [], [], []
    # 手动循环（需要记录逐年池）
    from config import WF_TEST_YEARS as YEARS
    mkt_full = build_market_features_pit(close, volume, member)
    for year in YEARS:
        pool_yr = reselect_pool(close, member, factor_tensor, fwd, year)
        pools_log.append({"year": year, "pool": pool_yr})
        if not pool_yr:
            continue
        # 训练该年模型（fixed 池那套逻辑，但用 pool_yr）
        tr_end = f"{year - 1}-12-31"
        te_start, te_end = f"{year}-01-01", f"{year}-12-31"
        close_tr = close[:tr_end]
        med_tr = month_end_dates(close_tr.index)
        X_tr = _build_X(close_tr, volume, member, factor_tensor, pool_yr, tr_end, med_tr)
        if len(X_tr) < 300:
            continue
        universe_tr = pd.Series(True, index=X_tr.index).unstack(fill_value=False)
        labels_tr = build_labels(close_tr, fwd_days=FWD_DAYS, top_q=TOP_Q, bottom_q=BOTTOM_Q,
                                 universe=universe_tr)
        valid_tr = align_X_y(X_tr, labels_tr).sort_index(level=0).dropna(subset=["label"])
        if len(valid_tr) < 300:
            continue
        w = build_sample_weights(valid_tr["label"], "balanced")
        dtr = lgb.Dataset(valid_tr.values, label=valid_tr["label"].values, weight=w)
        model = lgb.train({**LGBM_PARAMS, "verbose": -1}, dtr, num_boost_round=WF_NUM_BOOST)

        pred = _predict_year(model, factor_tensor, mkt_full, member, pool_yr,
                             te_start, te_end, close)
        if pred is None:
            continue
        pf, pred_daily = _monthly_backtest(pred, close)
        if pf is None:
            continue
        m = performance_metrics(pf["port_ret"])
        yearly_out.append({"year": year, "annual": m["annual"], "sharpe": m["sharpe"],
                           "mdd": m["mdd"], "n": m["n"], "pool_size": len(pool_yr)})
        ports_out.append(pf["port_ret"])
        preds_out.append(pred)
        print(f"  {year}: 池={len(pool_yr)} 年化={m['annual']:+.2%} SR={m['sharpe']:+.3f}")

    yearly_df = pd.DataFrame(yearly_out)
    yearly_df.to_csv(THIS_DIR / "walk_forward_yearly_pit_select.csv", index=False)
    pd.DataFrame(pools_log).to_csv(THIS_DIR / "walk_forward_pools.csv", index=False)
    all_wf = pd.concat(ports_out).sort_index() if ports_out else pd.Series(dtype=float)
    m_wf = performance_metrics(all_wf) if len(all_wf) else {}
    pred_full = pd.concat(preds_out).sort_index() if preds_out else pd.DataFrame()

    print(f"\n[5] 结果对比")
    print(f"  固定池 WF: 年化={m_fwf.get('annual', float('nan')):+.2%} "
          f"SR={m_fwf.get('sharpe', float('nan')):+.3f}（含 selection bias）")
    print(f"  PIT-Select WF: 年化={m_wf.get('annual', float('nan')):+.2%} "
          f"SR={m_wf.get('sharpe', float('nan')):+.3f}（消除 selection bias）")
    if not pred_full.empty:
        pred_full.to_csv(THIS_DIR / "oof_predictions_pit_select_fund.csv")
        print(f"  去 bias OOF 预测: {pred_full.shape} 保存完成")
    print("\n完成。")


if __name__ == "__main__":
    main()
