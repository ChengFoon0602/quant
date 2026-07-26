"""
strategies/factor_discovery/bootstrap_block_sensitivity.py

点评问题 5（日频版）：Bootstrap block_size=20（≈1个月）能否覆盖 A 股收益率的
自相关结构？测试 block_size = 5 / 10 / 20 / 40 / 60 交易日下 p 值是否稳定。

对象：与 report.md 4.3 节相同的三条多空收益序列（alpha001、alpha141、合成）。
若 p 值随 block_size 剧烈变化，说明检验结果不稳健。

用法:
    cd D:/桌面文件/quant/strategies/factor_discovery
    python bootstrap_block_sensitivity.py
"""

import sys
sys.path.insert(0, "../..")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data.fetcher import load_daily, cache_summary
from data.universe import get_listing_info
from signals.alpha191.factors import factor_141, factor_001

N_STOCKS = 300
DATE_START, DATE_END = "2010-01-01", "2025-12-31"
BLOCK_SIZES = [5, 10, 20, 40, 60]
N_BOOTSTRAP = 1000

cache = cache_summary()
all_symbols = sorted(cache["symbol"].tolist())
_, delist_dates = get_listing_info()
live_symbols = [s for s in all_symbols if s not in delist_dates]
symbols = live_symbols[:N_STOCKS]

all_dfs = {}
for sym in symbols:
    df = load_daily(sym)
    if df is None or len(df) < 100:
        continue
    df = df.loc[(df.index >= DATE_START) & (df.index <= DATE_END)]
    if len(df) < 100:
        continue
    all_dfs[sym] = df

close_data = {sym: df["close"] for sym, df in all_dfs.items()}
close_matrix = pd.DataFrame(close_data).sort_index()
print(f"股票数: {len(all_dfs)}  交易日: {len(close_matrix)}")


def compute_factor_matrix_custom(fn):
    rows = {}
    for sym, df in all_dfs.items():
        try:
            result = fn(df)
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=df.index)
            rows[sym] = result
        except Exception:
            rows[sym] = pd.Series(np.nan, index=df.index)
    return pd.DataFrame(rows).sort_index()


def align(factor_df):
    cd = close_matrix.index.intersection(factor_df.index)
    cs = close_matrix.columns.intersection(factor_df.columns)
    return close_matrix.loc[cd, cs], factor_df.loc[cd, cs]


def stratified_ls(factor_aligned, close_aligned):
    daily_ret = close_aligned.pct_change().fillna(0)
    fwd_ret = daily_ret.shift(-1); fwd_ret.iloc[-1] = 0
    grp = {i: [] for i in range(5)}
    for d in factor_aligned.index:
        f = factor_aligned.loc[d].dropna()
        if len(f) < 15:
            continue
        labels = pd.qcut(f, 5, labels=False, duplicates="drop")
        if labels.nunique() < 5:
            continue
        r_next = fwd_ret.loc[d]
        for g in range(5):
            syms = labels[labels == g].index
            grp[g].append(r_next[syms].mean())
    n = min(len(grp[4]), len(grp[0]))
    ls = pd.Series([grp[4][i] - grp[0][i] for i in range(n)])
    return ls.dropna()


def block_bootstrap(returns, block, n_boot=N_BOOTSTRAP, seed=42):
    vals = returns.values
    n = len(vals)
    rng = np.random.default_rng(seed)
    srs = np.empty(n_boot)
    if n <= block:
        for b in range(n_boot):
            s = rng.choice(vals, size=n, replace=True)
            srs[b] = np.sqrt(252) * s.mean() / s.std() if s.std() > 1e-12 else 0.0
        return srs
    n_blk = n // block
    for b in range(n_boot):
        starts = rng.integers(0, n - block, n_blk)
        sample = np.concatenate([vals[s:s + block] for s in starts])[:n]
        srs[b] = np.sqrt(252) * sample.mean() / sample.std() if sample.std() > 1e-12 else 0.0
    return srs


# ── 复现 report.md 4.3 节的三条多空序列 ──
print("\n计算因子...")
f001 = compute_factor_matrix_custom(factor_001)
f141 = compute_factor_matrix_custom(factor_141)
c001, f001_a = align(f001)
c141, f141_a = align(f141)
cd = c001.index.intersection(c141.index)
cs = c001.columns.intersection(c141.columns)
c_synth = c001.loc[cd, cs]
f1 = f001_a.loc[cd, cs]
f2 = f141_a.loc[cd, cs]

ls1 = stratified_ls(f1, c_synth)
ls2 = stratified_ls(f2, c_synth)
flip1 = ls1.mean() < 0
flip2 = ls2.mean() < 0
if flip1:
    f1 = -f1
if flip2:
    f2 = -f2
ls_001 = stratified_ls(f1, c_synth)

print(f"alpha001 多空序列: {len(ls_001)} 天\n")

# ── block_size 敏感性 ──
print(f"{'block_size':>10} {'≈交易日/月':>10} {'真实SR':>8} {'p值':>8} {'95%CI':>22}")
print("-" * 62)
real_sr = np.sqrt(252) * ls_001.mean() / ls_001.std()
rows = []
for blk in BLOCK_SIZES:
    bs = block_bootstrap(ls_001, block=blk)
    p = (bs >= real_sr).mean()
    ci_lo, ci_hi = np.percentile(bs, [2.5, 97.5])
    rows.append({"block_size": blk, "real_sr": real_sr, "p_value": p, "ci_lo": ci_lo, "ci_hi": ci_hi})
    print(f"{blk:>10} {blk/20:>9.1f}月 {real_sr:>8.3f} {p:>8.4f}   [{ci_lo:+.3f}, {ci_hi:+.3f}]")

df_out = pd.DataFrame(rows)
p_range = df_out["p_value"].max() - df_out["p_value"].min()
print(f"\np 值区间: [{df_out['p_value'].min():.4f}, {df_out['p_value'].max():.4f}]  跨度={p_range:.4f}")
if p_range < 0.05:
    print("→ p 值在不同 block_size 下稳定（跨度<0.05），检验结果稳健。")
else:
    print("→ p 值随 block_size 明显变化，检验结果对 block_size 选择敏感。")

df_out.to_csv("bootstrap_block_sensitivity.csv", index=False)
print("\n结果保存: bootstrap_block_sensitivity.csv")
