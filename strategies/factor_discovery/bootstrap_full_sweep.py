"""
strategies/factor_discovery/bootstrap_full_sweep.py

点评问题 6：106 个提纯因子中，究竟有多少个能产生 Bootstrap p<0.05 的显著多空收益？
逐个因子跑 5 分组分层多空 + block_bootstrap(block=20, n=500)，系统性验证
"因子库信噪比过低"这个结论，而不是仅凭 alpha055/alpha141/alpha001 三个样本外推。

方法与 report.py / random_ortho_test.py 完全一致：
  - 方向对齐：用全时段多空 SR 符号 flip
  - 5 分组分层多空（qcut，无法分组的日子跳过）
  - Block Bootstrap（block=20 交易日，1000 次，与 report.md 一致）

用法:
    cd D:/桌面文件/quant/strategies/factor_discovery
    python bootstrap_full_sweep.py
"""

import sys
sys.path.insert(0, "../..")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data.fetcher import load_daily, cache_summary
from data.universe import get_listing_info
from signals.alpha191.calculator import get_factor_func

N_STOCKS = 300
DATE_START, DATE_END = "2010-01-01", "2025-12-31"
N_BOOTSTRAP = 1000
BLOCK_SIZE = 20
N_GROUPS = 5

PURIFIED_FACTORS = [
    "alpha141", "alpha045", "alpha041", "alpha035", "alpha142", "alpha039",
    "alpha037", "alpha144", "alpha051", "alpha001", "alpha148", "alpha033",
    "alpha011", "alpha096", "alpha079", "alpha056", "alpha087", "alpha076",
    "alpha108", "alpha080", "alpha082", "alpha077", "alpha086", "alpha061",
    "alpha081", "alpha004", "alpha071", "alpha078", "alpha070", "alpha068",
    "alpha085", "alpha083", "alpha102", "alpha146", "alpha019", "alpha176",
    "alpha169", "alpha175", "alpha174", "alpha069", "alpha055", "alpha058",
    "alpha147", "alpha088", "alpha158", "alpha067", "alpha104", "alpha065",
    "alpha105", "alpha066", "alpha015", "alpha097", "alpha106", "alpha121",
    "alpha116", "alpha166", "alpha053", "alpha159", "alpha143", "alpha060",
    "alpha057", "alpha054", "alpha162", "alpha167", "alpha042", "alpha059",
    "alpha007", "alpha064", "alpha072", "alpha016", "alpha063", "alpha126",
    "alpha089", "alpha073", "alpha021", "alpha095", "alpha145", "alpha062",
    "alpha129", "alpha168", "alpha013", "alpha131", "alpha149", "alpha170",
    "alpha135", "alpha183", "alpha090", "alpha114", "alpha018", "alpha052",
    "alpha179", "alpha026", "alpha028", "alpha101", "alpha025", "alpha128",
    "alpha182", "alpha139", "alpha022", "alpha074", "alpha180", "alpha006",
    "alpha187", "alpha125", "alpha100", "alpha031",
]
print(f"提纯因子池: {len(PURIFIED_FACTORS)} 个")

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
daily_ret = close_matrix.pct_change().fillna(0)
fwd_ret = daily_ret.shift(-1)
fwd_ret.iloc[-1] = 0
print(f"股票数: {len(all_dfs)}  交易日: {len(close_matrix)}")


def stratified_ls(factor_mat):
    grp = {i: [] for i in range(N_GROUPS)}
    for d in close_matrix.index:
        if d not in factor_mat.index:
            continue
        f = factor_mat.loc[d].dropna()
        if len(f) < N_GROUPS * 3:
            continue
        labels = pd.qcut(f, N_GROUPS, labels=False, duplicates="drop")
        if labels.nunique() < N_GROUPS:
            continue
        r_next = fwd_ret.loc[d]
        for g in range(N_GROUPS):
            syms = labels[labels == g].index
            grp[g].append(r_next[syms].mean())
    n = min(len(grp[N_GROUPS - 1]), len(grp[0]))
    ls = pd.Series([grp[N_GROUPS - 1][i] - grp[0][i] for i in range(n)])
    return ls.dropna()


def block_bootstrap(returns, n_boot=N_BOOTSTRAP, block=BLOCK_SIZE, seed=42):
    vals = returns.values
    n = len(vals)
    rng = np.random.default_rng(seed)
    srs = np.empty(n_boot)
    if n < 10:
        return srs * 0
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


print(f"\n逐个因子跑分层多空 + Bootstrap（block={BLOCK_SIZE}, n={N_BOOTSTRAP}）...\n")
print(f"{'因子':<12} {'截面有效比':>10} {'LS_SR':>8} {'Bootstrap p':>12} {'显著?':>6}")
print("-" * 54)

results = []
for i, fid in enumerate(PURIFIED_FACTORS, 1):
    try:
        fn = get_factor_func(fid)
        rows = {}
        for sym, df in all_dfs.items():
            try:
                result = fn(df)
                if not isinstance(result, pd.Series):
                    result = pd.Series(result, index=df.index)
                rows[sym] = result
            except Exception:
                rows[sym] = pd.Series(np.nan, index=df.index)
        factor_mat = pd.DataFrame(rows).sort_index()

        cd = close_matrix.index.intersection(factor_mat.index)
        cs = close_matrix.columns.intersection(factor_mat.columns)
        factor_mat = factor_mat.loc[cd, cs]

        ls = stratified_ls(factor_mat)
        eff_ratio = len(ls) / len(cd) if len(cd) > 0 else 0.0

        if len(ls) < 30 or ls.std() == 0:
            results.append({"factor": fid, "eff_ratio": eff_ratio, "ls_sr": np.nan,
                            "p_value": np.nan, "significant": False, "n_days": len(ls)})
            print(f"{fid:<12} {eff_ratio*100:>9.1f}% {'—':>8} {'—':>12} {'跳过':>6}")
            continue

        # 方向对齐
        if ls.mean() < 0:
            ls = -ls
        sr = np.sqrt(252) * ls.mean() / ls.std()
        bs = block_bootstrap(ls)
        p = (bs >= sr).mean()
        sig = p < 0.05
        results.append({"factor": fid, "eff_ratio": eff_ratio, "ls_sr": sr,
                        "p_value": p, "significant": sig, "n_days": len(ls)})
        flag = "✓ 显著" if sig else ""
        print(f"{fid:<12} {eff_ratio*100:>9.1f}% {sr:>8.3f} {p:>12.4f} {flag:>6}")
    except Exception as e:
        results.append({"factor": fid, "eff_ratio": np.nan, "ls_sr": np.nan,
                        "p_value": np.nan, "significant": False, "n_days": 0, "error": str(e)})
        print(f"{fid:<12} ERROR: {e}")

    if i % 20 == 0:
        print(f"  ... {i}/{len(PURIFIED_FACTORS)} 完成")

df_results = pd.DataFrame(results)
df_results.to_csv("bootstrap_full_sweep.csv", index=False)

n_tested = df_results["p_value"].notna().sum()
n_sig = df_results["significant"].sum()
print(f"\n{'='*54}")
print(f"汇总: {len(PURIFIED_FACTORS)} 个因子, {n_tested} 个可测(截面可分组), {n_sig} 个 p<0.05 显著")
print(f"显著比例: {n_sig}/{n_tested} = {n_sig/max(n_tested,1)*100:.1f}%")
print(f"随机期望(α=0.05): {n_tested*0.05:.1f} 个")
if n_sig > 0:
    sig_factors = df_results[df_results["significant"]].sort_values("p_value")
    print(f"\n显著因子列表:")
    for _, r in sig_factors.iterrows():
        print(f"  {r['factor']:<12} LS_SR={r['ls_sr']:.3f}  p={r['p_value']:.4f}")
print(f"\n结果保存: bootstrap_full_sweep.csv")
