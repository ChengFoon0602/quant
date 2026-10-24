"""models/walk_forward_pit_select.py — 年度重选因子池的 Walk-Forward（消除 selection bias）。

背景
    `models/walk_forward.py` 的 WF 只重训模型、不重选因子池——因子池是用
    2010–2025 全样本 IC_IR 选出来的，等于「看着答案出题」。
    `models/report.md` 明确写了这一点（"因子池仍是全样本选的"），但 CSI300 侧
    一直没有实现池重选。zz500 侧的教训是：池重选后 WF 夏普 1.438 → 0.295（−79%）。

本脚本做什么
    每年（测试年 Y）只用**截至 Y-1 年末**的数据：
      1. 重算 191 因子的 IC_IR / FM_t / CS_eff
      2. 三维过滤 + Rank IC 相关性去重 → 选出该年度的因子池
      3. 用该池重训 LightGBM
      4. 预测 Y 年
    最后拼接 2015–2025 的预测与组合收益，得到真正样本外的绩效。

两阶段实现（内存与耗时约束）
    Stage A：分批算 191 因子，只保留**逐日 IC / 逐日 FM λ 序列**（191×3886，
             约 6MB），丢弃因子矩阵。这一阶段约 40 分钟。
    Stage B：拿到"年度-因子"指标表后逐年选池；再只重算**历年池的并集**因子
             （通常 40–60 个），用于逐年训练。避免 191×783×3886 全矩阵驻留（≈4.6GB）。

用法
    cd D:/桌面文件/quant
    .venv/Scripts/python.exe models/walk_forward_pit_select.py
    .venv/Scripts/python.exe models/walk_forward_pit_select.py --stage-a-only   # 只算指标
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).parent
PROJECT = MODEL_DIR.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "strategies" / "feature_selection"))

from signals.alpha191 import list_factors
from signals.alpha191.calculator import compute_factor_matrix
from signals.alpha191.factors import DEGENERATE_FACTORS
from models.labels import align_X_y, build_labels, build_sample_weights
from models.lgbm_trainer import LGBM_PARAMS, FWD_DAYS
from models.portfolio_backtest import build_portfolio, performance_metrics

from build_pit_matrix import load_pit_panel, build_market_features_pit
from select_features import compute_rank_ic_corr_matrix, remove_redundant

DATE_START, DATE_END = "2010-01-01", "2025-12-31"
TEST_YEARS = list(range(2015, 2026))
NUM_BOOST = 79
HOLD_DAYS = 10
TOP_Q, BOTTOM_Q = 0.2, 0.2
COST = 0.00102            # 双边 0.102%（铁律）

IC_IR_THRESHOLD = 0.05
FM_T_THRESHOLD = 2.0
CS_EFF_THRESHOLD = 0.5
CORR_THRESHOLD = 0.8
MAX_POOL_SIZE = 15
MUST_KEEP = {"alpha001"}

IC_CACHE = MODEL_DIR / "wf_pit_ic_panel.csv"
FM_CACHE = MODEL_DIR / "wf_pit_fm_panel.csv"
POOLS_LOG = MODEL_DIR / "wf_pit_pools.csv"


def section(t: str) -> None:
    print(f"\n{'=' * 72}\n  {t}\n{'=' * 72}", flush=True)


# ── Stage A：给每个因子算逐日 IC / FM λ 序列 ─────────────────────
def _daily_ic(fac: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    out = {}
    idx = fac.index.intersection(ret.index)
    cols = fac.columns.intersection(ret.columns)
    f_all, r_all = fac.loc[idx, cols], ret.loc[idx, cols]
    for d in idx:
        f, r = f_all.loc[d], r_all.loc[d]
        m = f.notna() & r.notna()
        if m.sum() < 10:
            continue
        v = f[m].rank().corr(r[m].rank())
        if pd.notna(v):
            out[d] = v
    return pd.Series(out)


def _daily_fm(fac: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    out = {}
    idx = fac.index.intersection(ret.index)
    cols = fac.columns.intersection(ret.columns)
    f_all, r_all = fac.loc[idx, cols], ret.loc[idx, cols]
    for d in idx:
        f, r = f_all.loc[d], r_all.loc[d]
        m = f.notna() & r.notna()
        if m.sum() < 10:
            continue
        X, Y = f[m].values, r[m].values
        var = np.var(X)
        if var < 1e-12:
            continue
        out[d] = np.cov(X, Y)[0, 1] / var
    return pd.Series(out)


def stage_a(symbols: list[str], batch_size: int = 48) -> None:
    section("Stage A：逐日 IC / FM λ 序列（191 因子）")
    close_matrix, _, member_daily = load_pit_panel()
    close_matrix = close_matrix[symbols].sort_index()
    member_daily = member_daily.reindex(columns=symbols, fill_value=False)
    fwd_ret = (close_matrix.shift(-FWD_DAYS) / close_matrix - 1).loc[DATE_START:DATE_END]

    fids = [f for f in list_factors() if f not in DEGENERATE_FACTORS]
    print(f"  因子 {len(fids)} 个（已排除 {len(DEGENERATE_FACTORS)} 个退化因子）", flush=True)

    ic_rows, fm_rows = {}, {}
    for b0 in range(0, len(fids), batch_size):
        batch = fids[b0:b0 + batch_size]
        print(f"  ── 批次 {b0 // batch_size + 1}: {batch[0]} ~ {batch[-1]}", flush=True)
        _, tensor = compute_factor_matrix(symbols, batch, start=DATE_START, end=DATE_END,
                                          verbose=False, strict=False)
        for fid in batch:
            fac = tensor.pop(fid).where(member_daily)
            ic_rows[fid] = _daily_ic(fac, fwd_ret)
            fm_rows[fid] = _daily_fm(fac, fwd_ret)
        del tensor

    pd.DataFrame(ic_rows).to_csv(IC_CACHE)
    pd.DataFrame(fm_rows).to_csv(FM_CACHE)
    print(f"\n  已保存: {IC_CACHE.name} / {FM_CACHE.name}", flush=True)


# ── Stage B：逐年选池 ────────────────────────────────────────────
def _metrics_asof(series: pd.Series, tr_end: str) -> tuple[float, float, float]:
    """截至 tr_end 的 (IR, t, n)。"""
    s = series.loc[:tr_end].dropna()
    if len(s) < 60:
        return 0.0, 0.0, float(len(s))
    sd = s.std(ddof=0)
    if sd <= 0:
        return 0.0, 0.0, float(len(s))
    return float(s.mean() / sd), float(s.mean() / sd * np.sqrt(len(s))), float(len(s))


def select_pool_for_year(ic_panel: pd.DataFrame, fm_panel: pd.DataFrame,
                         member_daily: pd.DataFrame,
                         factor_tensor: dict[str, pd.DataFrame],
                         fwd_ret: pd.DataFrame, year: int,
                         cs_map: dict[str, float]) -> list[str]:
    tr_end = f"{year - 1}-12-31"
    rows = []
    for fid in ic_panel.columns:
        ir, _, n = _metrics_asof(ic_panel[fid], tr_end)
        _, t_fm, _ = _metrics_asof(fm_panel[fid], tr_end)
        rows.append({"factor": fid, "IC_IR": ir, "FM_t": t_fm, "n": n})
    df = pd.DataFrame(rows)

    cand = df[(df["IC_IR"].abs() > IC_IR_THRESHOLD) & (df["FM_t"].abs() > FM_T_THRESHOLD)]
    if len(cand) < 3:
        return []

    # CS_eff 用全样本值：它是「因子是否离散到能分组」的结构性属性，年际几乎不变。
    # 逐年在 3886 天上重算 qcut 会让 Stage B 多花 1–2 小时，收益不对等。
    keep, cs_ok = {}, []
    for _, r in cand.iterrows():
        fid = r["factor"]
        if fid not in factor_tensor or cs_map.get(fid, 0.0) <= CS_EFF_THRESHOLD:
            continue
        keep[fid] = factor_tensor[fid].loc[:tr_end].where(member_daily.loc[:tr_end])
        cs_ok.append(fid)
    if len(cs_ok) < 3:
        return []

    ic_map = {r["factor"]: r["IC_IR"] for _, r in cand.iterrows() if r["factor"] in cs_ok}
    corr = compute_rank_ic_corr_matrix(keep, fwd_ret.loc[:tr_end])
    sel, _ = remove_redundant(corr, ic_map, CORR_THRESHOLD, MUST_KEEP)
    sel = sorted([f for f in sel if f in ic_map], key=lambda f: abs(ic_map[f]), reverse=True)
    return sel[:MAX_POOL_SIZE]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a-only", action="store_true")
    args = ap.parse_args()

    close_matrix, volume_matrix, member_daily = load_pit_panel()
    symbols = list(close_matrix.columns)
    print(f"PIT 面板: {close_matrix.shape[0]} 天 × {len(symbols)} 只", flush=True)

    if not IC_CACHE.exists() or not FM_CACHE.exists():
        stage_a(symbols)
    else:
        print(f"复用已有 IC/FM 面板: {IC_CACHE.name}")
    if args.stage_a_only:
        return 0

    section("Stage B-1：逐年选池")
    pur = pd.read_csv(PROJECT / "strategies" / "feature_selection" / "purify_results.csv")
    cs_map = dict(zip(pur["factor"], pur["cs_effective_ratio"]))
    ic_panel = pd.read_csv(IC_CACHE, index_col=0, parse_dates=True)
    fm_panel = pd.read_csv(FM_CACHE, index_col=0, parse_dates=True)
    fwd_ret = (close_matrix.shift(-FWD_DAYS) / close_matrix - 1)
    print(f"  IC 面板: {ic_panel.shape}  FM 面板: {fm_panel.shape}")

    # 先粗选候选（每年 ≤ 全部），再算并集因子矩阵
    # 为避免为选池而加载 191 因子全矩阵，这里分两步：先用 IC/FM 粗筛出并集，
    # 再算并集矩阵，然后用 CS_eff + 冗余剔除定池。
    prelim: dict[int, list[str]] = {}
    for year in TEST_YEARS:
        tr_end = f"{year - 1}-12-31"
        rows = []
        for fid in ic_panel.columns:
            ir, _, n = _metrics_asof(ic_panel[fid], tr_end)
            _, t_fm, _ = _metrics_asof(fm_panel[fid], tr_end)
            rows.append((fid, ir, t_fm))
        cand = [f for f, ir, t in rows
                if abs(ir) > IC_IR_THRESHOLD and abs(t) > FM_T_THRESHOLD]
        prelim[year] = cand
        print(f"  {year}: IC+FM 粗筛 {len(cand)} 个")

    union = sorted({f for v in prelim.values() for f in v})
    print(f"  并集因子 {len(union)} 个，开始计算其因子矩阵（用于 CS_eff 与训练）...", flush=True)
    _, factor_tensor = compute_factor_matrix(symbols, union, start=DATE_START, end=DATE_END,
                                             verbose=False, strict=False)
    factor_tensor = {f: df.where(member_daily) for f, df in factor_tensor.items()}

    pools: dict[int, list[str]] = {}
    pool_rows = []
    for year in TEST_YEARS:
        pool = select_pool_for_year(ic_panel, fm_panel, member_daily,
                                    factor_tensor, fwd_ret, year, cs_map)
        pools[year] = pool
        pool_rows.append({"year": year, "pool_size": len(pool), "pool": ",".join(pool)})
        print(f"  {year}: 池 {len(pool)} 个 → {pool}")
    pd.DataFrame(pool_rows).to_csv(POOLS_LOG, index=False)

    section("Stage B-2：逐年重训 + 预测")
    mkt_full = build_market_features_pit(close_matrix, volume_matrix, member_daily)
    yearly, ports, preds = [], [], []
    for year in TEST_YEARS:
        pool = pools.get(year) or []
        if len(pool) < 3:
            print(f"  {year}: 池不足（{len(pool)}），跳过")
            continue
        tr_end = f"{year - 1}-12-31"
        te_start, te_end = f"{year}-01-01", f"{year}-12-31"

        frames = [factor_tensor[f].loc[:tr_end].stack().rename(f) for f in pool]
        X_tr = pd.concat(frames, axis=1)
        mkt_tr = mkt_full.loc[:tr_end]
        mkt_align = mkt_tr.loc[X_tr.index.get_level_values(0)]
        mkt_align.index = X_tr.index
        X_tr = pd.concat([X_tr, mkt_align], axis=1)
        mem_tr = member_daily.loc[:tr_end].stack()
        X_tr = X_tr[mem_tr.reindex(X_tr.index).fillna(False).astype(bool).values]
        X_tr.index.names = ["date", "symbol"]

        if len(X_tr) < 1000:
            print(f"  {year}: 训练样本不足 {len(X_tr)}")
            continue
        uni_tr = pd.Series(True, index=X_tr.index).unstack(fill_value=False)
        labels = build_labels(close_matrix.loc[:tr_end], fwd_days=FWD_DAYS, top_q=TOP_Q,
                              bottom_q=BOTTOM_Q, universe=uni_tr)
        valid = align_X_y(X_tr, labels).sort_index(level=0).dropna(subset=["label"])
        if len(valid) < 1000:
            print(f"  {year}: 有效样本不足 {len(valid)}")
            continue
        feat = list(X_tr.columns)
        w = build_sample_weights(valid["label"], "balanced")
        dtr = lgb.Dataset(valid[feat].values, label=valid["label"].values, weight=w)
        model = lgb.train({**LGBM_PARAMS, "verbose": -1}, dtr, num_boost_round=NUM_BOOST)

        frames_te = [factor_tensor[f].loc[te_start:te_end].stack().rename(f) for f in pool]
        X_te = pd.concat(frames_te, axis=1)
        mkt_te = mkt_full.loc[te_start:te_end]
        a = mkt_te.loc[X_te.index.get_level_values(0)]
        a.index = X_te.index
        X_te = pd.concat([X_te, a], axis=1)
        mem_te = member_daily.loc[te_start:te_end].stack()
        X_te = X_te[mem_te.reindex(X_te.index).fillna(False).astype(bool).values]
        X_te.index.names = ["date", "symbol"]
        if len(X_te) < 100:
            continue

        pred = pd.Series(model.predict(X_te[feat].values), index=X_te.index).unstack(level=1)
        td = pred.index.intersection(close_matrix.index)
        ts = pred.columns.intersection(close_matrix.columns)
        pred, cm = pred.loc[td, ts], close_matrix.loc[td, ts]
        if len(td) <= HOLD_DAYS + 2:
            continue
        pf = build_portfolio(pred, cm, long_only=False, cost=COST, hold_days=HOLD_DAYS)
        m = performance_metrics(pf["port_ret"])
        yearly.append({"year": year, "annual": m["annual"], "sharpe": m["sharpe"],
                       "mdd": m["mdd"], "n": m["n"], "pool_size": len(pool)})
        ports.append(pf["port_ret"])
        preds.append(pred)
        print(f"  {year}: 池={len(pool):2d} 训练={len(valid):>7,} 年化={m['annual']:>+8.2%} "
              f"SR={m['sharpe']:>+7.3f}", flush=True)

    if not ports:
        print("无有效窗口")
        return 1

    section("汇总")
    yearly_df = pd.DataFrame(yearly)
    yearly_df.to_csv(MODEL_DIR / "wf_pit_yearly.csv", index=False)
    all_ret = pd.concat(ports).sort_index()
    m_all = performance_metrics(all_ret)
    print(f"  PIT-Select WF（消除 selection bias）: 年化={m_all['annual']:+.2%} "
          f"SR={m_all['sharpe']:+.3f} MDD={m_all['mdd']:+.2%} n={m_all['n']}")
    print(f"\n逐年:")
    for _, r in yearly_df.iterrows():
        print(f"  {int(r['year'])}: 池={int(r['pool_size']):2d} 年化={r['annual']:>+8.2%} SR={r['sharpe']:>+7.3f}")

    pd.DataFrame([{"method": "Walk-Forward PIT-Select", "annual": m_all["annual"],
                   "sharpe": m_all["sharpe"], "mdd": m_all["mdd"], "n_days": m_all["n"]}]
                 ).to_csv(MODEL_DIR / "wf_pit_summary.csv", index=False)
    if preds:
        pd.concat(preds).sort_index().to_csv(MODEL_DIR / "oof_predictions_pit_select.csv")
    print(f"\n结果: models/wf_pit_summary.csv / wf_pit_yearly.csv / wf_pit_pools.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
