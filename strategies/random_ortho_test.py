"""
random_ortho_test.py — 随机抽 3 个提纯因子 → 方向对齐 → 正交化合成 → Bootstrap 对比。

方向对齐: 用多空 SR 符号判断因子方向，负向因子取反后再正交化。
"""

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data.fetcher import load_daily, cache_summary
from signals.alpha191.calculator import get_factor_func

# ── 配置 ─────────────────────────────────────────────────
N_STOCKS = 300
DATE_START, DATE_END = "2010-01-01", "2025-12-31"
N_BOOTSTRAP = 1000
BLOCK_SIZE = 20
RANDOM_SEED = 42
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
print(f"随机种子: {RANDOM_SEED}")

rng = np.random.default_rng(RANDOM_SEED)
chosen = list(rng.choice(PURIFIED_FACTORS, size=3, replace=False))
print(f"随机抽取: {', '.join(chosen)}")

# ══════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════
print("\n加载数据...")
cache = cache_summary()
symbols = sorted(cache["symbol"].tolist())[:N_STOCKS]

all_dfs = {}
for sym in symbols:
    df = load_daily(sym)
    if df is None or len(df) < 100:
        continue
    df = df.loc[(df.index >= DATE_START) & (df.index <= DATE_END)]
    if len(df) < 100:
        continue
    cols = ["open", "high", "low", "close", "volume"]
    if "amount" in df.columns:
        cols.append("amount")
    all_dfs[sym] = df[cols]

close_data = {sym: df["close"] for sym, df in all_dfs.items()}
close_matrix = pd.DataFrame(close_data).sort_index()
daily_ret = close_matrix.pct_change().fillna(0)
fwd_ret = daily_ret.shift(-1)
fwd_ret.iloc[-1] = 0

# ══════════════════════════════════════════════════════════
#  因子计算
# ══════════════════════════════════════════════════════════
print("\n计算因子...")
factor_mats = {}
for fid in chosen:
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
    factor_mats[fid] = pd.DataFrame(rows).sort_index()

# 对齐
cd = close_matrix.index
cs = close_matrix.columns
for fid in chosen:
    cd = cd.intersection(factor_mats[fid].index)
    cs = cs.intersection(factor_mats[fid].columns)
close_matrix = close_matrix.loc[cd, cs]
for fid in chosen:
    factor_mats[fid] = factor_mats[fid].loc[cd, cs]

print(f"对齐后: {len(cd)} 天, {len(cs)} 只")

# ══════════════════════════════════════════════════════════
#  方向判定（用多空 SR 符号）
# ══════════════════════════════════════════════════════════
print("\n── 方向判定（多空 SR）──")

def get_ls_sr(factor_mat):
    """计算多空 SR，返回 (sr, sign)"""
    group_rets = {i: [] for i in range(N_GROUPS)}
    for d in cd:
        f = factor_mat.loc[d].dropna()
        if len(f) < N_GROUPS * 3:
            continue
        labels = pd.qcut(f, N_GROUPS, labels=False, duplicates="drop")
        if labels.nunique() < N_GROUPS:
            continue
        r_next = fwd_ret.loc[d]
        for g in range(N_GROUPS):
            syms_g = labels[labels == g].index
            group_rets[g].append(r_next[syms_g].mean())
    n_ls = min(len(group_rets[4]), len(group_rets[0]))
    ls_ret = pd.Series([group_rets[4][i] - group_rets[0][i] for i in range(n_ls)]).dropna()
    if len(ls_ret) < 10:
        return 0.0, 0
    sr = np.sqrt(252) * ls_ret.mean() / ls_ret.std() if ls_ret.std() > 0 else 0
    sign = 1 if ls_ret.mean() > 0 else -1
    return sr, sign

flip_map = {}  # fid → bool (是否翻转)
aligned_mats = {}
for fid in chosen:
    sr, sign = get_ls_sr(factor_mats[fid])
    if sign < 0:
        aligned_mats[fid] = -factor_mats[fid]
        flip_map[fid] = True
        print(f"  {fid}: LS_SR={sr:+.3f} → FLIP (方向为负, 取反后 SR={-sr:+.3f})")
    elif sign > 0:
        aligned_mats[fid] = factor_mats[fid]
        flip_map[fid] = False
        print(f"  {fid}: LS_SR={sr:+.3f} → 保持")
    else:
        aligned_mats[fid] = factor_mats[fid]
        flip_map[fid] = False
        print(f"  {fid}: LS_SR=nan → 无法判定, 保持")

# ══════════════════════════════════════════════════════════
#  IC_IR 对比 (对齐后)
# ══════════════════════════════════════════════════════════
print("\n── IC_IR 对比（方向对齐后）──")

def compute_ic_ir(factor_mat):
    ic_list = []
    for d in cd:
        f = factor_mat.loc[d]
        r = fwd_ret.loc[d]
        mask = f.notna() & r.notna()
        if mask.sum() < 10:
            continue
        ic = f[mask].rank().corr(r[mask].rank())
        if pd.isna(ic):
            continue
        ic_list.append(ic)
    ic_arr = np.array(ic_list)
    mean = ic_arr.mean()
    std = ic_arr.std()
    ir = mean / std if std > 0 else 0
    t = mean / std * np.sqrt(len(ic_arr)) if std > 0 else 0
    return {"mean": mean, "std": std, "IR": ir, "t": t, "pos": (ic_arr > 0).mean()}

single_ic = {}
for fid in chosen:
    single_ic[fid] = compute_ic_ir(aligned_mats[fid])

# ══════════════════════════════════════════════════════════
#  Gram-Schmidt 正交化（用方向对齐后的因子）
# ══════════════════════════════════════════════════════════
print("\n── 正交化合成 ──")

# 按 IC_IR 排序
sorted_fids = sorted(chosen, key=lambda x: abs(single_ic[x]["IR"]), reverse=True)
base_fid = sorted_fids[0]

ortho_mats = {base_fid: aligned_mats[base_fid].copy()}
print(f"  Base: {base_fid} (|IC_IR|={abs(single_ic[base_fid]['IR']):.4f})")

for fid in sorted_fids[1:]:
    regressors = [ortho_mats[bf] for bf in sorted_fids[:sorted_fids.index(fid)]]
    residual_rows = {}
    for d in cd:
        y = aligned_mats[fid].loc[d]
        X_list = [r.loc[d] for r in regressors]
        mask = y.notna()
        for X in X_list:
            mask = mask & X.notna()
        if mask.sum() < 10:
            residual_rows[d] = pd.Series(np.nan, index=cs)
            continue
        Y_vec = y[mask].values
        X_mat = np.column_stack([X[mask].values for X in X_list])
        try:
            beta = np.linalg.lstsq(X_mat, Y_vec, rcond=None)[0]
            resid = Y_vec - X_mat @ beta
        except np.linalg.LinAlgError:
            residual_rows[d] = pd.Series(np.nan, index=cs)
            continue
        res_series = pd.Series(np.nan, index=cs)
        res_series[mask] = resid
        residual_rows[d] = res_series
    ortho_mats[fid] = pd.DataFrame(residual_rows).T.sort_index()

# 等权合成
combo = sum(ortho_mats.values()) / len(ortho_mats)
combo = combo.reindex(index=cd, columns=cs)
combo_ic = compute_ic_ir(combo)

print(f"  {'因子':<12} {'IC_IR':>8} {'IC_t':>8} {'正IC占比':>10} {'翻转':>6}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")
for fid in chosen:
    ic = single_ic[fid]
    fmark = "← flip" if flip_map[fid] else ""
    print(f"  {fid:<12} {ic['IR']:>+8.4f} {ic['t']:>+8.2f} {ic['pos']*100:>9.1f}% {fmark:>6}")
print(f"  {'合成(等权)':<11} {combo_ic['IR']:>+8.4f} {combo_ic['t']:>+8.2f} {combo_ic['pos']*100:>9.1f}%")

# ══════════════════════════════════════════════════════════
#  分层回测 + Bootstrap
# ══════════════════════════════════════════════════════════
print("\n── 分层多空 + Bootstrap ──")

def stratified_ls(factor_mat):
    group_rets = {i: [] for i in range(N_GROUPS)}
    for d in cd:
        f = factor_mat.loc[d].dropna()
        if len(f) < N_GROUPS * 3:
            continue
        labels = pd.qcut(f, N_GROUPS, labels=False, duplicates="drop")
        if labels.nunique() < N_GROUPS:
            continue
        r_next = fwd_ret.loc[d]
        for g in range(N_GROUPS):
            syms_g = labels[labels == g].index
            group_rets[g].append(r_next[syms_g].mean())
    n_ls = min(len(group_rets[4]), len(group_rets[0]))
    ls_ret = pd.Series([group_rets[4][i] - group_rets[0][i] for i in range(n_ls)])
    return ls_ret.dropna()

def block_bootstrap(returns, n_boot=N_BOOTSTRAP, block=BLOCK_SIZE, seed=42):
    vals = returns.values
    n = len(vals)
    rng_bt = np.random.default_rng(seed)
    srs = np.empty(n_boot)
    if n <= block:
        for b in range(n_boot):
            sample = rng_bt.choice(vals, size=n, replace=True)
            s = sample.std()
            srs[b] = np.sqrt(252) * sample.mean() / s if s > 1e-12 else 0.0
        return srs
    n_blocks = max(1, n // block)
    for b in range(n_boot):
        starts = rng_bt.integers(0, n - block, n_blocks)
        blocks = [vals[s:s + block] for s in starts]
        sample = np.concatenate(blocks)[:n]
        s = sample.std()
        srs[b] = np.sqrt(252) * sample.mean() / s if s > 1e-12 else 0.0
    return srs

single_bs = {}
for fid in chosen:
    ls = stratified_ls(aligned_mats[fid])
    if len(ls) < 10:
        print(f"  {fid}: 多空序列太短 ({len(ls)} 天), 跳过")
        single_bs[fid] = {"sr": np.nan, "p": np.nan, "p_pos": np.nan}
        continue
    sr_real = np.sqrt(252) * ls.mean() / ls.std() if ls.std() > 0 else 0
    bs = block_bootstrap(ls)
    single_bs[fid] = {"sr": sr_real, "p": (bs >= sr_real).mean(), "p_pos": (bs > 0).mean()}
    print(f"  {fid}: LS_SR={sr_real:+.3f}, p={single_bs[fid]['p']:.4f}, P(SR>0)={single_bs[fid]['p_pos']*100:.1f}%")

combo_ls = stratified_ls(combo)
combo_sr = np.sqrt(252) * combo_ls.mean() / combo_ls.std() if combo_ls.std() > 0 else 0
combo_bs = block_bootstrap(combo_ls)
combo_p = (combo_bs >= combo_sr).mean()
combo_p_pos = (combo_bs > 0).mean()
print(f"  合成(等权): LS_SR={combo_sr:+.3f}, p={combo_p:.4f}, P(SR>0)={combo_p_pos*100:.1f}%")

# ══════════════════════════════════════════════════════════
#  汇总
# ══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  随机抽取: {', '.join(chosen)}")
print(f"  方向对齐: {sum(flip_map.values())} 个因子被翻转")
print(f"  正交化基底: {base_fid}")
print(f"")
print(f"  {'':<12} {'IC_IR':>8} {'LS_SR':>8} {'p_val':>8} {'P(SR>0)':>10}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
for fid in chosen:
    ic = single_ic[fid]
    b = single_bs[fid]
    p_str = f"{b['p']:.4f}" if not np.isnan(b['p']) else "nan"
    ppos_str = f"{b['p_pos']*100:.1f}%" if not np.isnan(b['p_pos']) else "nan"
    print(f"  {fid:<12} {ic['IR']:>+8.4f} {b['sr']:>+8.3f} {p_str:>8} {ppos_str:>10}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
print(f"  {'合成(等权)':<11} {combo_ic['IR']:>+8.4f} {combo_sr:>+8.3f} {combo_p:>8.4f} {combo_p_pos*100:>9.1f}%")

# 比较
valid_single = {fid: single_bs[fid] for fid in chosen if not np.isnan(single_bs[fid]["p"])}
if valid_single:
    best_name = max(valid_single, key=lambda x: valid_single[x]["sr"])
    best_sr = valid_single[best_name]["sr"]
    best_p = valid_single[best_name]["p"]
    p_delta = best_p - combo_p
    print(f"\n  最佳单因子: {best_name} (LS_SR={best_sr:.3f}, p={best_p:.4f})")
    print(f"  合成 p:     {combo_p:.4f}")
    print(f"  Δp:         {p_delta:+.4f}  {'✓ 合成降低了 p 值' if p_delta > 0 else '✗ 合成未改善'}")
    print(f"  合成 SR:    {combo_sr:+.3f}  vs  最佳单因子 SR: {best_sr:+.3f}  ({'↑' if combo_sr > best_sr else '↓'})")

print(f"\n  === 完成 ===")
