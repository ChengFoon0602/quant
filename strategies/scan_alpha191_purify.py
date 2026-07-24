"""
scan_alpha191_purify.py — Alpha 191 全库提纯扫描。

逐因子计算 IC_IR + Fama-MacBeth λ/t，按提纯管道筛选:
  |IC_IR| > 0.05 AND FM |t| > 2.0

用法: python scan_alpha191_purify.py
"""

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data.fetcher import load_daily, cache_summary
from signals.alpha191 import list_factors
from signals.alpha191.calculator import get_factor_func

# ── 配置 ─────────────────────────────────────────────────
N_STOCKS = 300  # 扫描用股票数
DATE_START, DATE_END = "2010-01-01", "2025-12-31"
IC_IR_THRESHOLD = 0.05
FM_T_THRESHOLD = 2.0


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════
section("数据加载")

cache = cache_summary()
all_symbols = sorted(cache["symbol"].tolist())
symbols = all_symbols[:N_STOCKS]
print(f"扫描股票: {len(symbols)} 只（全量缓存 {len(all_symbols)} 只）")

all_dfs = {}
for sym in symbols:
    df = load_daily(sym)
    if df is None or len(df) < 100:
        continue
    df = df.loc[(df.index >= DATE_START) & (df.index <= DATE_END)]
    if len(df) < 100:
        continue
    # 只需要 OHLCV + amount
    cols_need = ["open", "high", "low", "close", "volume"]
    if "amount" in df.columns:
        cols_need.append("amount")
    all_dfs[sym] = df[cols_need]

N_VALID = len(all_dfs)
print(f"有效股票: {N_VALID} 只")

# 构建 close 矩阵（用于收益计算）
close_data = {sym: df["close"] for sym, df in all_dfs.items()}
close_matrix = pd.DataFrame(close_data).sort_index()
daily_ret = close_matrix.pct_change().fillna(0)
fwd_ret = daily_ret.shift(-1)
fwd_ret.iloc[-1] = 0

common_dates = close_matrix.index
print(f"日期范围: {common_dates[0].date()} ~ {common_dates[-1].date()} ({len(common_dates)} 天)")

# ══════════════════════════════════════════════════════════
#  逐因子扫描
# ══════════════════════════════════════════════════════════
section("逐因子扫描 (IC_IR + Fama-MacBeth)")

ALL_FACTORS = list_factors()
print(f"因子总数: {len(ALL_FACTORS)}")
print(f"提纯阈值: |IC_IR| > {IC_IR_THRESHOLD}, FM |t| > {FM_T_THRESHOLD}\n")

results = []
n_errors = 0

for i, fid in enumerate(ALL_FACTORS):
    fn = get_factor_func(fid)

    # 计算因子矩阵
    factor_rows = {}
    for sym, df in all_dfs.items():
        try:
            result = fn(df)
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=df.index)
            factor_rows[sym] = result
        except Exception:
            factor_rows[sym] = pd.Series(np.nan, index=df.index)
    factor_df = pd.DataFrame(factor_rows).sort_index()

    # 对齐
    cd = common_dates.intersection(factor_df.index)
    cs = close_matrix.columns.intersection(factor_df.columns)
    factor_aligned = factor_df.loc[cd, cs]

    # ── IC_IR ──
    ic_list = []
    n_valid_ic = 0
    for d in cd:
        f = factor_aligned.loc[d]
        r = fwd_ret.loc[d]
        mask = f.notna() & r.notna()
        if mask.sum() < 10:
            continue
        ic = f[mask].rank().corr(r[mask].rank())
        if pd.isna(ic):
            continue
        ic_list.append(ic)
        n_valid_ic += 1

    ic_arr = np.array(ic_list)
    ic_mean = ic_arr.mean() if len(ic_arr) > 0 else 0
    ic_std = ic_arr.std() if len(ic_arr) > 0 else 1
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0

    # ── Fama-MacBeth ──
    lam_list = []
    n_valid_fm = 0
    for d in cd:
        f = factor_aligned.loc[d]
        r = fwd_ret.loc[d]
        mask = f.notna() & r.notna()
        if mask.sum() < 10:
            continue
        X = f[mask].values
        Y = r[mask].values
        var_x = np.var(X)
        if var_x < 1e-12:
            continue
        cov = np.cov(X, Y)[0, 1]
        lam = cov / var_x
        lam_list.append(lam)
        n_valid_fm += 1

    lam_arr = np.array(lam_list)
    lam_mean = lam_arr.mean() if len(lam_arr) > 0 else 0
    lam_std = lam_arr.std() if len(lam_arr) > 0 else 1
    lam_t = lam_mean / lam_std * np.sqrt(len(lam_arr)) if lam_std > 0 else 0
    lam_annual = lam_mean * 252

    pass_ic = abs(ic_ir) > IC_IR_THRESHOLD
    pass_fm = abs(lam_t) > FM_T_THRESHOLD
    passed = pass_ic and pass_fm

    results.append({
        "factor": fid,
        "IC_IR": ic_ir,
        "IC_mean": ic_mean,
        "IC_t": ic_mean / ic_std * np.sqrt(len(ic_arr)) if ic_std > 0 else 0,
        "FM_λ_annual": lam_annual,
        "FM_t": lam_t,
        "IC_pass": pass_ic,
        "FM_pass": pass_fm,
        "pass": passed,
        "n_ic_days": n_valid_ic,
        "n_fm_days": n_valid_fm,
    })

    status = "✓ 通过" if passed else ("IC" if pass_ic else ("FM" if pass_fm else "✗"))
    pct_done = (i + 1) / len(ALL_FACTORS) * 100
    print(f"  [{i+1:3d}/191 {pct_done:4.0f}%] {fid}: IC_IR={ic_ir:+.3f}, FM_t={lam_t:+.2f}  → {status}")

# ══════════════════════════════════════════════════════════
#  结果汇总
# ══════════════════════════════════════════════════════════
section("提纯结果")

df = pd.DataFrame(results)
df = df.sort_values("IC_IR", key=abs, ascending=False)

# 全部排名（按 |IC_IR|）
print(f"\n  {'因子':<10} {'IC_IR':>8} {'IC_t':>8} {'FM_λ(年化)':>12} {'FM_t':>8} {'结果'}")
print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*12} {'-'*8} {'-'*6}")
for _, r in df.iterrows():
    status = "✓ 提纯" if r["pass"] else ("仅IC" if r["IC_pass"] else ("仅FM" if r["FM_pass"] else "✗"))
    print(f"  {r['factor']:<10} {r['IC_IR']:>+8.4f} {r['IC_t']:>+8.2f} "
          f"{r['FM_λ_annual']*100:>+10.2f}%/yr {r['FM_t']:>+8.2f} {status}")

# 通过提纯的因子
passed_df = df[df["pass"]]
n_passed = len(passed_df)

print(f"\n{'─'*70}")
print(f"  提纯管道: |IC_IR| > {IC_IR_THRESHOLD} AND FM |t| > {FM_T_THRESHOLD}")
print(f"  全库 {len(ALL_FACTORS)} 个 → 通过 {n_passed} 个")

if n_passed > 0:
    print(f"\n  ★ 提纯因子列表:")
    for _, r in passed_df.iterrows():
        print(f"    {r['factor']:10s}  IC_IR={r['IC_IR']:+.4f}  FM_t={r['FM_t']:+.2f}  "
              f"FM_λ={r['FM_λ_annual']*100:+.2f}%/yr")
else:
    print("\n  ⚠ 无因子通过提纯管道。")

# 统计分布
print(f"\n  统计:")
print(f"    |IC_IR| > 0.05: {(df['IC_pass'].sum())} 个")
print(f"    FM |t| > 2.0:  {(df['FM_pass'].sum())} 个")
print(f"    同时通过:      {n_passed} 个")
print(f"    IC_IR 均值:    {df['IC_IR'].mean():.4f}")
print(f"    IC_IR 中位:    {df['IC_IR'].median():.4f}")
print(f"    IC_IR 前 10%:  {df['IC_IR'].abs().quantile(0.90):.4f}")

# Top 10 by |IC_IR|
print(f"\n  Top 10 (按 |IC_IR|):")
for i, (_, r) in enumerate(df.head(10).iterrows()):
    print(f"    {i+1}. {r['factor']:10s}  IC_IR={r['IC_IR']:+.4f}  FM_t={r['FM_t']:+.2f}")

print(f"\n  === 扫描完成 ===")
